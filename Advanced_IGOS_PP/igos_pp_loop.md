# iGOS_pp 主优化循环详解

> 对应代码：`Advanced_IGOS_PP/IGOS_pp.py:714-992`

## 本文档修订记录
- 2026-5-11 创建

---

## 核心概念：双 Mask 架构

iGOS++ 同时优化 **两个独立的低分辨率 mask**（`[B, 1, 32, 32]`，初始值 `0.4`），从不同视角衡量像素重要性：

| Mask | 视角 | 优化方向 | 高值含义 |
|------|------|---------|---------|
| `masks_del` | 删除 (Deletion) | **最大化** interval_score | 遮掉这个区域 → 模型答错 → **重要** |
| `masks_ins` | 插入 (Insertion) | **最小化** interval_score（取负） | 只留这个区域 → 模型能答对 → **重要** |

最终显著性图 = **`masks_del × masks_ins`**（逐元素相乘），只有两个视角都认定的区域才会保留。

---

## 数据流概览

```
masks_del [B,1,32,32] ──→ upscale ──→ up_masks1 [B,1,H,W]
masks_ins [B,1,32,32] ──→ upscale ──→ up_masks2 [B,1,H,W]

up_masks1 ──────────┐
                    ├──×──→ combined_mask ──→ 积分梯度(删除+插入方向)
up_masks2 ──────────┘

up_masks1 ──→ 积分梯度(删除方向)      ──→ total_grads1
up_masks2 ──→ 积分梯度(插入方向,取负)  ──→ total_grads2
                                  └──→ + 正则化梯度(L1+TV+L2)
                                         │
                                         ▼
                                   NAG / LS 更新 masks_del, masks_ins
```

---

## 每轮迭代详细步骤

### 1. 上采样（808-809）

```python
up_masks1 = upscale(masks_del, image)  # 32×32 → H×W
up_masks2 = upscale(masks_ins, image)
```

`upscale` 用双线性插值将低分辨率 mask 映射到原图空间尺寸，使得 mask 能与原图逐像素运算。

---

### 2. 四种积分梯度计算（822-866）

每轮计算 **4 次** 积分梯度，每次都会触发反向传播，梯度累加到 `masks_del.grad` / `masks_ins.grad`：

| 调用 | 作用于 | 方向 | 对梯度的影响 |
|------|--------|------|------------|
| `loss_comb_del` (822) | `up_masks1 × up_masks2` | 删除 | `total_grads1 += grad_del`<br>`total_grads2 += grad_ins` |
| `loss_comb_ins` (843) | `up_masks1 × up_masks2` | 插入 | `total_grads1 -= grad_del`<br>`total_grads2 -= grad_ins` |
| `loss_del` (853) | `up_masks1` 单独 | 删除 | `total_grads1 += grad_del` |
| `loss_ins` (861) | `up_masks2` 单独 | 插入 | `total_grads2 -= grad_ins` |

**为什么插入方向要取负？** 因为 `integrated_gradient` 默认返回删除语义的 loss（越大越好删），插入语义相反：插入 mask 越大 → 模型应越确信 → loss 应越小。所以取负号反向优化。

**为什么组合 mask 要算两次（删除+插入）？** 同一个组合 mask，既要看"遮掉它模型会挂吗"（删除），又要看"只留它能认吗"（插入）。两者方向不同，都需要梯度信号。

**为什么还要算单独的 mask？** 组合 mask 是两个的乘积，梯度会分散到两个 mask 上。单独算一次确保每个 mask 都有独立的梯度信号。

---

### 3. 梯度平均（869-870）

```python
total_grads1 /= 2
total_grads2 /= 2
```

4 次梯度累积（2 次来自组合 + 1 次单独 + 1 次单独）后，除以 2 使组合与单独部分的贡献平衡，避免正则化项被淹没。

---

### 4. 正则化 Loss（882-898）

```python
L2 = exp_decay(L2, i, gamma)          # L2 系数指数衰减
loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks_del * masks_ins)
losses = loss_l1 + loss_tv + loss_l2
losses.sum().backward()
```

三个正则项：

| Loss | 公式 | 作用 |
|------|------|------|
| `loss_l1` | `L1 × mean(\|1 - mask\|)` | 鼓励 mask 趋近 1（不要全遮掉） |
| `loss_tv` | `L3 × bilateral_tv_norm(image, mask)` | 双边 TV 平滑，在图像边缘处允许 mask 突变 |
| `loss_l2` | `L2 × sum((1 - mask)²)` | L2 衰减，系数随迭代指数衰减（gamma=1.0） |

正则梯度追加到 `total_grads` 后统一更新。

---

### 5. 优化器更新

#### LS（Line Search，918-924）

```python
masks = cat(masks_del, masks_ins)           # [2, 1, 32, 32]
total_grads = cat(total_grads1, total_grads2)
lrs = line_search(masks, total_grads, loss_function, lr)
masks_del.data -= total_grads1 * lrs
masks_ins.data -= total_grads2 * lrs
```

用线搜索找最优步长，沿负梯度方向更新。

#### NAG（Nesterov Accelerated Gradient，926-976，默认使用）

```python
e = i / (i + momentum)          # 动量系数，随迭代从 0 趋近 1
cita_d = mask_del - lr * grad   # 梯度下降步
momentum_term = cita_d - cita_d_prev  # 一阶动量（当前步与上一步之差）
masks_del = cita_d + e * momentum_term  # Nesterov 修正
```

**NAG 核心**：先沿之前积累的动量"跳"一步，再在该位置计算梯度做修正，比标准动量收敛更快、更稳定。

---

### 6. 后处理（978-981）

```python
masks_del.grad.zero_()
masks_ins.grad.zero_()
masks_del.data.clamp_(0, 1)   # 确保 mask 在 [0,1] 范围内
masks_ins.data.clamp_(0, 1)
```

---

## Loss 打印含义

每轮输出示例：
```
iteration: 5 lr: 1.0000 loss_comb_del: -0.4321, loss_comb_ins: 0.8765, loss_del: -0.2345, loss_ins: 0.6543, loss_l1: 0.1234, loss_tv: 0.0456, loss_l2: 0.0078
```

| 字段 | 含义 | 期望趋势 |
|------|------|---------|
| `loss_comb_del` | 组合 mask 删除方向 IG | 递减（更易删除） |
| `loss_comb_ins` | 组合 mask 插入方向 IG | 递减（更易保留） |
| `loss_del` | 单独删除 mask 的 IG | 递减 |
| `loss_ins` | 单独插入 mask 的 IG | 递增（取负后，绝对值变大） |
| `loss_l1` | 稀疏正则 | 递减（mask 趋于 1） |
| `loss_tv` | 平滑正则 | 递减 |
| `loss_l2` | L2 衰减（指数衰减） | 递减 |

**所有 7 个 loss 都应有可见变化**，否则说明 mask 梯度太弱。

---

## 与旧版（main.py）的关键区别

| 特性 | Qwen 工作流 (Advanced_IGOS_PP) | 旧版 (根目录 methods.py) |
|------|-------------------------------|------------------------|
| 优化器 | NAG（默认）/ LS | LS |
| L2 超参 | 0.1 → 指数衰减 | 60（远高） |
| L3 超参 | 10 | 30 |
| 学习率 | 1.0 | 0.1 |
| 梯度裁剪 | 无 | 有 |
| `interval_score` | Qwen 专用版本（在 IGOS_pp.py 内） | 独立函数（在 methods_helper.py） |
| `processor` 参数 | 传入 `tensor2pack` 函数 | 传 `None` |

---

## 待研究疑问

### Q1： 4次integrated_gradient的调用
- 4次integrated_gradient调用得到的loss都没有使用
- 调用1和调用2的传参完全一致

### Q2：interval_score运行逻辑
