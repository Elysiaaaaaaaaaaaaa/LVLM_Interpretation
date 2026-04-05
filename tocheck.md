## 打印日志中每个 loss 的含义

在 `iGOS_pp()` 函数的打印日志中（第 875 行），各 loss 的含义如下：

```python
print(f'iteration: {i} lr: {lr:.4f} loss_comb_del: {loss_comb_del:.4f}, loss_comb_ins: {loss_comb_ins:.4f}, loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')
```

### 主要 Loss 项

1. **loss_comb_del** (Combined Deletion Loss)
   - 含义：组合 mask（mask_del * mask_ins）的删除损失
   - 计算：使用积分梯度计算删除重要区域后模型预测的下降程度
   - 目标：最小化，确保删除重要区域能显著降低模型预测

2. **loss_comb_ins** (Combined Insertion Loss)
   - 含义：组合 mask（mask_del * mask_ins）的插入损失
   - 计算：从 baseline 开始插入重要区域后模型预测的提升程度
   - 目标：最小化，确保插入重要区域能显著提升模型预测

3. **loss_del** (Deletion Loss)
   - 含义：仅删除 mask 的损失
   - 计算：单独优化删除 mask 时的积分梯度损失
   - 目标：找到需要删除的重要区域

4. **loss_ins** (Insertion Loss)
   - 含义：仅插入 mask 的损失
   - 计算：单独优化插入 mask 时的积分梯度损失
   - 目标：找到需要插入的重要区域

### 正则化 Loss 项

5. **loss_l1** (L1 Regularization)
   - 公式：`L1 * mean(|1 - mask|)`
   - 含义：鼓励 mask 接近全 1，避免过度稀疏
   - 系数：`L1 = 3.0`（可调整）

6. **loss_tv** (Total Variation Regularization)
   - 公式：`L3 * bilateral_tv_norm(image, mask, tv_beta=2, sigma=0.01)`
   - 含义：双边全变分正则化，使 mask 更平滑，同时考虑图像边界
   - 系数：`L3 = 10.0`（可调整）
   - 作用：减少噪声，使热力图区域更连续

7. **loss_l2** (L2 Regularization)
   - 公式：`L2 * sum((1 - mask)^2)`
   - 含义：L2 惩罚，鼓励 mask 接近 1
   - 系数：`L2 = 5.0`，使用指数衰减 `exp_decay(L2, iter, gamma=1.0)`
   - 作用：随迭代次数衰减，初期强约束，后期放松

---

## 各个参数控制的变量

### iGOS++ 算法参数（第 143-155 行）

```python
size=24              # mask 的初始尺寸（后续会上采样到原图尺寸）
opt = 'NAG'          # 优化器类型：NAG (Nesterov Accelerated Gradient)
diverse_k = 1        # 初始化多样性参数
init_posi = 0        # 初始化位置参数
init_val = 0.4       # mask 初始值（0-1 之间）
L1 = 2.0             # L1 正则化系数（控制 mask 稀疏度）
L2 = 5.0             # L2 正则化系数（控制 mask 接近 1 的程度）
gamma = 1.0          # L2 系数的指数衰减率
L3 = 10.0            # TV 正则化系数（控制 mask 平滑度）
momentum = 0.8       # 动量参数（用于 NAG 优化器）
ig_iter = 10         # 积分梯度采样次数
iterations=30        # iGOS++ 优化迭代次数
lr=0.02              # 学习率
```

### 参数作用详解

| 参数 | 控制内容 | 调大效果 | 调小效果 | 推荐范围 |
|------|----------|----------|----------|----------|
| **size** | mask 初始分辨率 | 更精细的热力图，计算更慢 | 更粗糙的热力图，计算更快 | 24-32 |
| **iterations** | 优化迭代次数 | mask 更精确，时间更长 | mask 较粗糙，时间更短 | 20-40 |
| **ig_iter** | 积分梯度采样数 | 梯度估计更准确 | 梯度估计较粗糙 | 10-15 |
| **lr** | 学习率 | 收敛快，可能不稳定 | 收敛慢，更稳定 | 0.01-0.1 |
| **L1** | L1 正则化强度 | mask 更稀疏 | mask 更密集 | 1.0-3.0 |
| **L2** | L2 正则化强度 | mask 更接近 1 | mask 可以更极端 | 0.05-5.0 |
| **L3** | TV 正则化强度 | mask 更平滑 | mask 可能有噪声 | 5.0-15.0 |
| **momentum** | 优化器动量 | 加速收敛，可能震荡 | 收敛稳定，速度较慢 | 0.5-0.9 |
| **gamma** | L2 衰减率 | L2 快速衰减 | L2 保持更久 | 0.5-2.0 |
| **init_val** | mask 初始值 | 从较亮区域开始 | 从较暗区域开始 | 0.1-0.5 |

### 可视化参数（第 305-311 行）

```python
vis_gamma = 0.4      # 热力图对比度增强参数（伽马校正）
# 热力图叠加权重：heatmap * 0.35 + original_image * 0.65
```

| 参数 | 控制内容 | 调大效果 | 调小效果 |
|------|----------|----------|----------|
| **vis_gamma** | 热力图对比度 | 对比度降低，颜色更均匀 | 对比度增强，重要区域更突出 |
| 热力图权重 | 叠加时热力图占比 | 颜色更重，可能遮挡原图 | 颜色更淡，原图更清晰 |
| 原图权重 | 叠加时原图占比 | 原图更清晰 | 原图被热力图覆盖 |

### 调优建议

1. **热力图颜色太重**：降低热力图权重（如 0.3-0.35），或增大 `vis_gamma`（如 0.5-0.6）
2. **热力图太分散**：增大 `L3`（TV 正则化）使区域更集中
3. **优化不稳定**：降低 `lr`，增加 `momentum`
4. **计算时间太长**：减少 `iterations` 或 `ig_iter`
