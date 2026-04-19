# `gen_explanations_qwenvl` 函数说明

本文档总结 `Advanced_IGOS_PP/IGOS_pp.py` 中 `gen_explanations_qwenvl`（约第 99 行起）的**工作流**与**工作原理**，面向 Qwen-VL 系列多模态模型的可解释性可视化（显著性图 / 热力图）。

---

## 1. 函数职责与返回值

| 项目 | 说明 |
|------|------|
| **输入** | `model`、`processor`（Qwen-VL）、`image`（PIL）、`text_prompt`、`tokenizer`；可选 `positions`（指定生成序列中要解释的 token 下标）、`select_word_id`（替换对应位置的词 id） |
| **输出** | `masks`：与输入图像同尺寸的显著性数组（numpy）；`superimposed_img`：原图与伪彩色热力图的叠加图（uint8 numpy，BGR 色彩空间来自 OpenCV） |

整体上，该函数在**固定 greedy 生成文本**的前提下，通过 **iGOS++（`iGOS_pp`）** 优化一张与视觉输入对齐的 mask，使模型对选定「关键词」token 的预测与**模糊基线图**之间的差异可被解释区域放大或缩小。

---

## 2. 端到端工作流（按执行顺序）

### 2.1 图像预处理

1. 保持宽高比，将长边限制在 **512** 以内。
2. 宽高对齐到 **28 的倍数**（与视觉 patch / Qwen-VL 处理习惯一致）。
3. 记录 `input_size = (H, W)`（注意 PIL 为 `(W,H)`，此处取 `(height, width)`）。

### 2.2 构造「清晰图」与「模糊基线」

1. 根据图像尺寸调用 `get_kernel_size` 得到高斯核大小，对原图做 **OpenCV 高斯模糊** 得到 `blur`。
2. **清晰图**用于正常前向与关键词筛选；**模糊图**作为「视觉信息被抹掉」的基线，用于对比哪些 token 更依赖真实图像内容。

### 2.3 Qwen-VL 多模态输入构建

1. 用 `messages` 格式组装 user：`image` + `text_prompt`。
2. `processor.apply_chat_template(..., add_generation_prompt=True)` 得到文本模板。
3. `process_vision_info` 分别处理清晰图与模糊图的 vision 张量。
4. `processor(..., return_tensors="pt")` 得到 `inputs` 与 `inputs_blur`，并 `.to(model.device)`。
5. 后续优化阶段使用的 `pixel_values` 分别记为 `image_tensor`、`blur_tensor`。

### 2.4 文本生成与 token 对齐

1. `torch.no_grad()` 下 `model.generate(..., do_sample=False, num_beams=1, max_new_tokens=50)`：**贪心解码**，保证可重复。
2. `generated_ids_trimmed`：去掉 prompt 与图像占位对应部分，只保留**新生成**的 token id。
3. `selected_token_word_id` / `target_token_position`：为每个生成 token 建立在全序列中的位置索引，供后续 `pred_probs` 取对应步的 logits。

### 2.5 关键词（要解释的 token）选择

- **`positions is None`**：调用 `find_keywords`（见第三节）。
- **`positions` 已给定**：从 `generated_ids_trimmed` 中按 `positions[0]` 解码出 `keywords` 字符串。

可选：**`select_word_id`** 不为 `None` 时，按 `(position, word_id)` 直接改写 `generated_ids_trimmed` 中对应 token，用于强制解释某个词而非模型实际生成的词。

### 2.6 组装 `pred_data` 与初始 mask

1. 将 `labels`、`keywords`（位置列表）、整图 bbox、`pred_text`、`keywords_text` 等写入 `pred_data`。
2. **`get_initial`**：在整图 bbox 内按 `diverse_k`、`init_posi` 划分子区域，生成低分辨率（默认 `size=32`）上的**初始显著性 mask**（再经 `init_val` 缩放）。本函数中 bbox 为整图，故主要是为 `iGOS_pp` 提供可优化的初值 `init_masks`。

### 2.7 iGOS++ 优化（核心）

对 `pred_data['labels']` 中每条 label（通常即一条生成序列）：

1. 调用 **`iGOS_pp`**，传入：
   - 模型、`inputs`、`generated_ids`；
   - **CLIP 归一化**的清晰图张量：`pil_to_clip_tensor_bcwh(image)`（内部会再对齐到 28 倍数尺寸等）；
   - **同一归一化**的模糊基线：`pil_to_clip_tensor_bcwh(blur)`；
   - `target_token_position`、`selected_token_word_id`；
   - `init_mask`、`keyword`（positions）、`tensor2pack` 作为 `processor`（将上采样后的 patch 排成 ViT 期望的 flatten 形式）等。
2. `iGOS_pp` 内部维护 **deletion / insertion 两路 mask**，通过 **`integrated_gradient`** 等估计对选定 token 概率的影响，并结合 L1、双边 TV、L2 等正则迭代更新（优化器配置如 `opt='NAG'`，外层 `iterations=25` 等由本函数顶部超参指定）。
3. 迭代结束后取 **`masks[0,0]`**，在 CPU 上 **min-max 归一化**（退化情况置零），resize 到当前 PIL 图像的宽高。

### 2.8 可视化与返回

1. `np.nan_to_num` 清理 NaN。
2. **`vis_gamma = 0.4`** 对 mask 做幂次变换以增强对比度。
3. 将 mask 转为伪彩色热力图（JET），与原图 **0.35 : 0.65** 混合得到 `superimposed_img`。
4. 释放部分张量并 `empty_cache()`，返回 `masks, superimposed_img`。

---

## 3. `find_keywords` 的原理（与清晰 / 模糊对比）

实现位于 `methods_helper.py`：

1. **`pred_probs`**：在 `generated_ids`（完整生成序列）与给定 `pixel_values`（清晰或模糊）下前向，取 `target_token_position` 处 logits，对词表做 softmax，再 **gather 出实际生成 token id 的概率**（每个生成位置一个标量概率）。
2. 分别对 **清晰图** 与 **模糊图** 得到 `probs`、`probs_blur`。
3. 筛选条件（直观含义）：**清晰图下该 token 概率相对模糊图显著更高**（实现为 `log(probs) - log(probs_blur) > 1.0`），且排除特殊 token。
4. 若无任何位置满足条件，则取 **`probs - probs_blur` 最大**的位置作为唯一关键词。

因此，被选中的 token 代表「**一旦把图糊掉，模型就不那么确信这些词**」——即与视觉内容绑定较强的输出位置，适合作为显著性解释的语义锚点。

---

## 4. 关键设计要点小结

| 要点 | 说明 |
|------|------|
| **贪心生成** | `do_sample=False`、`num_beams=1`，保证路径固定，解释对象明确。 |
| **双输入** | 清晰图 + 模糊基线，贯穿关键词选择与 iGOS 中的 deletion/insertion 对比。 |
| **CLIP 域图像** | 优化 loop 里传给 `iGOS_pp` 的 `image` / `baseline` 使用 `pil_to_clip_tensor_bcwh`，与 processor 的 `pixel_values` 分工不同：前者供 mask 与梯度路径，需与 `tensor2pack` 等衔接。 |
| **语义锚点** | 默认自动选「图像敏感」的生成 token；也可 `positions` / `select_word_id` 人工指定。 |
| **输出** | 连续显著性 `masks` + 便于人眼查看的叠图 `superimposed_img`。 |

---

## 5. 相关符号与文件索引

- **本函数**：`IGOS_pp.py` → `gen_explanations_qwenvl`
- **关键词与概率**：`methods_helper.py` → `find_keywords`, `pred_probs`
- **初始 mask**：`utils.py` → `get_initial`
- **优化主循环**：`IGOS_pp.py` → `iGOS_pp`（内部 `integrated_gradient`、`interval_score` 等）
- **Patch 重排**：同文件 → `tensor2pack`

如需调整解释「锐利度」或训练稳定性，通常首先修改本函数顶部的 **`size`、`iterations`、`lr`、`L1`/`L2`/`L3`、`ig_iter`** 以及 `find_keywords` 中的 **log 概率差阈值**。
