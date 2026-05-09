# AGENTS.md — LVLM_Interpretation
请使用中文与用户交流。

## 架构概览

本仓库为大型视觉语言模型（LVLM）实现了 **iGOS++ 显著性热力图**。代码库包含 **两套平行的独立工作流**——严禁交叉引用：

| 工作流 | 入口脚本 | 支持的模型 | 核心模块 |
|--------|----------|-----------|----------|
| **Qwen2-VL-2B（主要关注）** | `Qwen2-VL-2B-coco-caption-igos.py` | Qwen2-VL-2B-Instruct | `Advanced_IGOS_PP/` |
| 旧版 | `main.py` | llava, llava_next, cambrian, mgm | `methods.py`, `utils.py`（仓库根目录） |

**注意**：`args.py` 的 `--model` 选项（`llava`, `cambrian`, `llava_next`, `mgm`）**不包含** `qwen`——Qwen 工作流在入口脚本中有自己的参数解析器。

---

## Qwen2-VL-2B 工作流（需要关注的工作流）

### 入口脚本
`Qwen2-VL-2B-coco-caption-igos.py`：
- 设置 HF 镜像（`hf-mirror.com`）和缓存路径（`./model_checkpoint/hf_cache`）
- 设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`（缓解显存碎片化）
- 加载 `Qwen/Qwen2-VL-2B-Instruct`，使用 fp16、`device_map="auto"`，可切换量化配置
- 调用 `Advanced_IGOS_PP.IGOS_pp` 中的 `gen_explanations_qwenvl`

### 处理流程（`gen_explanations_qwenvl` 内部）
1. **图像预处理**：保持宽高比，最长边限制在 512px，并对齐到 **28 的倍数**（Qwen-VL 的 patch 要求）
2. **模糊基线**：使用 OpenCV 高斯模糊（核大小由 `get_kernel_size()` 根据图像宽度决定：501/301/201/101）
3. **模型前向**：贪心解码（`do_sample=False, num_beams=1, max_new_tokens=128`）
4. **关键词选择**（`Advanced_IGOS_PP/methods_helper.py` 中的 `find_keywords`）：
   - 比较 `log P(token|清晰图)` 与 `log P(token|模糊图)`
   - 阈值：`log P(清晰) - log P(模糊) > 4.0`
   - 过滤掉 `special_ids`（标点/控制 token）
   - 如果没有 token 通过阈值，则退回到选择差值最大的 token
5. **Mask 初始化**（`get_initial`）：在整图 bbox 上创建低分辨率（默认 `size=32`）mask
6. **iGOS++ 优化**（`iGOS_pp`）：迭代优化删除 mask + 插入 mask

### `tensor2pack`（Qwen2-VL 专用，关键函数）
将图像 patch 重排，适配 Qwen2-VL 的 3D patch embedding：
- 输入：`[B, C, H, W]` → 输出：`[grid_t, temporal_patch_size*C, patch_size*grid_h, patch_size*grid_w]`
- `temporal_patch_size=2`, `patch_size=14`, `merge_size=2`
- 如果 B 不能整除 temporal_patch_size，则重复最后一帧补齐
- 调用 `iGOS_pp` 时**必须**作为 `processor=` 参数传入；InternVL 则传 `None`

### `interval_score`（Qwen 专用，位于 `Advanced_IGOS_PP/IGOS_pp.py`）
- 在基线图与清晰图之间线性插值：`alpha * 清晰图 + (1-alpha) * 模糊基线图`，alpha ∈ [0,1]
- 对插值后的图像添加 2D 高斯噪声（标准差 0.2）
- 每个插值步调用一次 `pred_probs`
- 支持分块 IG 计算以节省显存

### `pred_probs`（位于 `Advanced_IGOS_PP/methods_helper.py`）
- 复制模型输入，将 `pixel_values` 替换为扰动后的图像
- 获取 `target_token_position - 1` 位置的 logits
- 当 `return_log_probs=True` 时使用 `log_softmax`（数值稳定性更好）
- 在对应位置收集 `selected_token_word_id` 的 logits

### 关键调试：NaN 处理
代码中有大量的 NaN 调试打印。如果看到 `[NaN DEBUG]`：
1. 检查 logits 是否有极端值（fp16 溢出）
2. 最可能的修复方案：在 `pred_probs` 中，softmax 之前将 logits 转为 fp32
3. 过往修复记录见 `.cursor/igos-mask-fix-implementation-summary.md`

### 关键超参数（`gen_explanations_qwenvl` 内）
```
size=32, opt='NAG', iterations=10, ig_iter=20, ig_chunks=2
L1=0.5, L2=0.1, gamma=1.0, L3=10.0, momentum=5, lr=1.0, init_val=0.4
```
这些参数与 `main.py` 工作流的默认值以及 `arg_log`（旧配置快照）**显著不同**。**切勿在工作流之间复制超参数。**

与旧配置（`.cursor/igos-mask-fix-implementation-summary.md` 记录）相比的变更：
- L2 从 60 降至 0.1（原是过度压制梯度信号）
- L3 从 30 降至 10
- lr 从 0.1 提升至 1.0
- 移除了梯度裁剪
- 移除了 pred_probs/interval_score 中的 nan_to_num

---

## 图像归一化（注意不一致）

| 工作流 | 归一化方式 |
|--------|-----------|
| Qwen2-VL（`Advanced_IGOS_PP`）| **OpenAI CLIP**：mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] |
| 旧版（根目录 `utils.py`）| **ImageNet**：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

`Advanced_IGOS_PP/IGOS_pp.py` 中的 `pil_to_clip_tensor_bcwh` 使用 CLIP 均值/方差进行归一化，并将图像对齐到 28 的倍数。此函数用于 mask 优化的图像张量。`process_vision_info` 则独立处理 Qwen 内部的 processor 归一化。

---

## Loss 函数详解（iGOS++）

优化过程中日志打印各 loss 的含义：
```
loss_comb_del | loss_comb_ins | loss_del | loss_ins | loss_l1 | loss_tv | loss_l2
```
- **loss_comb_del**：组合 mask（del*ins）在删除方向上的积分梯度 → 最大化删除效果
- **loss_comb_ins**：组合 mask 在插入方向上的积分梯度 → 最大化插入效果
- **loss_del**：仅删除 mask 的积分梯度
- **loss_ins**：仅插入 mask 的积分梯度
- **loss_l1**：`L1 * mean(|1 - mask|)` —— 鼓励 mask 接近 1
- **loss_tv**：双边 TV 范数 —— 带边缘感知的空间平滑
- **loss_l2**：`L2 * sum((1-mask)^2)` —— 按指数衰减（`exp_decay`，衰减率 `gamma`）

**所有 7 个 loss 项在迭代过程中都应有可见变化**——如果没有，说明 mask 梯度太弱。

---

## 文件布局（重要——非显而易见的细节）
```
/  （仓库根目录）
├── Qwen2-VL-2B-coco-caption-igos.py    # Qwen2-VL 入口（v2 2B）
├── Qwen25-VL-3B-coco-caption-igos.py   # Qwen2.5-VL 入口（v2.5 3B）
├── InternVL3_5-4B-coco-caption-igos.py # InternVL3.5 入口（独立的 explainer）
├── main.py                              # 旧版：llava/cambrian/mgm/llava_next
├── args.py                              # 旧版：仅 4 种旧版模型可选
├── methods.py                           # 旧版 iGOS_p / iGOS_pp 包装函数
├── methods_helper.py                    # 旧版辅助函数（与 Advanced_IGOS_PP 不同！）
├── utils.py                             # 旧版工具函数（与 Advanced_IGOS_PP 不同！）
│
├── Advanced_IGOS_PP/                    # Qwen-VL 工作流（自包含）
│   ├── IGOS_pp.py                       # gen_explanations_qwenvl, iGOS_pp, interval_score, integrated_gradient, pil_to_clip_tensor_bcwh, tensor2pack, _dump_iter_heatmap
│   ├── methods_helper.py                # pred_probs, find_keywords, phi, upscale, tv_norm, bilateral_tv_norm
│   ├── utils.py                         # get_initial, get_kernel_size, save_heatmaps 等
│   ├── gen_explanations_qwenvl.md       # gen_explanations_qwenvl 详细中文文档
│   └── __init__.py                      # 空文件
│
├── .cursor/
│   └── igos-mask-fix-implementation-summary.md  # 过往修复记录：移除梯度裁剪，超参数调优
│
├── scripts/kaggle/                       # Kaggle 实验自动化脚本（bash）
├── photoes/                              # 测试图片
├── data.json                             # 最小测试数据（2 张图片）
├── arg_log                               # 旧参数快照（不要用——L2=60, lr=0.005）
├── install.bat / install.sh              # 安装脚本：pip install -e ".[train]" + 额外依赖
```

### ⚠️ 关键：`pred_probs`、`find_keywords`、`phi`、`upscale` 各有两个版本
- **根目录** `methods_helper.py` / `utils.py`——供 `main.py`（旧版模型）使用
- **`Advanced_IGOS_PP/methods_helper.py`** / `utils.py`——供 Qwen-VL 脚本使用
- **`Advanced_IGOS_PP/IGOS_pp.py`** 中还有自有的 `interval_score`、`integrated_gradient`、`iGOS_pp`——函数签名与根目录 `methods.py` 不同
- 切勿在 Advanced_IGOS_PP 中引用根目录的函数，反之亦然

---

## 运行实验

### 本地运行
！注意！本地没有GPU环境，无法运行模型推理相关任务
```bash
# Qwen2-VL-2B（在仓库根目录执行）
python Qwen2-VL-2B-coco-caption-igos.py --Datasets datasets/coco/val2017 --eval-list datasets/Qwen2-VL-2B-coco-caption.json --save-dir ./baseline_results/Qwen2-VL-2B-coco-caption/IGOS_PP

# InternVL3.5
python InternVL3_5-4B-coco-caption-igos.py --Datasets datasets/coco/val2017 --eval-list datasets/InternVL3_5-4B-coco-caption.json

# 旧版（非 qwen）
python main.py --method iGOS++ --model llava --dataset cvbench --data_path <路径> --image_folder <路径> --output_dir <路径> --size 32 --L1 1.0 --L2 0.1 --L3 10.0 --ig_iter 10 --gamma 1.0 --iterations 5 --momentum 5
```

### Kaggle
`scripts/kaggle/` 目录下的脚本实现了推送 → 触发 → 轮询 → 获取结果的自动化流程。

---

## 数据集格式
```json
[
    {"image_path": "图片路径.jpg", "question": "描述这张图片……"},
    …
]
```
- COCO caption 格式，需包含 question 字段（非标准 COCO 格式）
- COCO 图片预期放在 `datasets/coco/val2017/` 目录
- 评测列表 JSON（如 `datasets/Qwen2-VL-2B-coco-caption.json`）建立图片与问题的映射

---

## 输出
- `{save-dir}/npy/`——原始显著性 numpy 数组（归一化到 [0,1]，再应用 gamma=0.4）
- `{save-dir}/visualization/`——热力图叠加图（BGR 色彩空间，JET 色图，35% 热力图 + 65% 原图）
- 每轮迭代可视化：文件名前缀 `_iter_{N}.jpg`（每 5 轮一次，需设置 `iter_vis_save_prefix`）

---

## 已知问题
1. **Qwen25-VL-3B-coco-caption-igos.py** 已经废除。
2. **fp16 不稳定**：fp16 下 logits/softmax 出现 NaN 是最常见的故障模式；可在 `pred_probs` 中将 logits 转为 fp32 (尝试，造成CUDA OOM)
3. **显存**：`gen_explanations_qwenvl` 处理每张图片后都会执行 `torch.cuda.empty_cache()`；使用 `ig_chunks=2` 和 `max_new_tokens=128`，Qwen2-VL-2B 建议 12-16GB 显存
4. 默认 `size=32` 的 mask 分辨率**较低**；高分辨率 mask（48+）需要更多迭代次数

---

## 速查表：关键函数及其签名

| 函数 | 所在文件 | 用途 |
|------|---------|------|
| `gen_explanations_qwenvl(model, processor, image, text_prompt, tokenizer)` | `Advanced_IGOS_PP/IGOS_pp.py:99` | Qwen 主解释流程 |
| `iGOS_pp(model, inputs, ..., init_mask, image, baseline, ..., size, iterations, ig_iter, L1, L2, L3, ...)` | `Advanced_IGOS_PP/IGOS_pp.py:713` | 核心优化循环（返回 mask_del*masks_ins + loss 列表） |
| `interval_score(model, inputs, generated_ids, images, ..., baseline, up_masks, num_iter, ..., positions, processor)` | `Advanced_IGOS_PP/IGOS_pp.py:527` | 在插值路径上计算积分梯度 |
| `integrated_gradient(...)` | `Advanced_IGOS_PP/IGOS_pp.py:595` | 封装 interval_score + backward；支持 ig_chunks |
| `pred_probs(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, need_grad, return_log_probs)` | `Advanced_IGOS_PP/methods_helper.py` | 前向推理，返回目标位置的 log prob |
| `find_keywords(model, inputs, generated_ids, output_ids, image, blur_image, target_token_position, selected_token_word_id, tokenizer)` | `Advanced_IGOS_PP/methods_helper.py` | 选择 P(清晰)-P(模糊) > 阈值的 token |
| `pil_to_clip_tensor_bcwh(img)` | `Advanced_IGOS_PP/IGOS_pp.py:15` | PIL→CLIP 归一化 torch tensor [1,3,H,W]，对齐 28 倍数 |
| `tensor2pack(patches)` | `Advanced_IGOS_PP/IGOS_pp.py:63` | Patch 重排，适配 Qwen2-VL 3D embedding |
| `upscale(masks, images)` | `Advanced_IGOS_PP/methods_helper.py:101` | 双线性上采样 mask 到图像空间尺寸 |
| `phi(img, baseline, mask)` | `Advanced_IGOS_PP/methods_helper.py` | 线性插值：mask*img + (1-mask)*baseline |
| `_dump_iter_heatmap(masks_del, masks_ins, image_tensor, i, prefix)` | `Advanced_IGOS_PP/IGOS_pp.py:673` | 每 5 轮保存中间热力图 |
| `get_initial(pred_data, k, init_posi, init_val, input_size, out_size)` | `Advanced_IGOS_PP/utils.py` | 在 bbox 子网格中创建初始低分辨率 mask |
