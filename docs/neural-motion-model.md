# 神经动作模型设计

状态：草案。不涉及代码变更。

---

## 1. 模型定位

模型替代 `AutoregressiveMotionBackend`，实现相同的 `MotionBackend.generate(world, seed, generated_at_ms) → MotionPlan` 接口。宿主零改动。

```
输入: WorldState (宿主采集的窗口几何 + 猫状态 + 鼠标)
输出: MotionPlan (local_rotation_deltas × H 帧 + root_translation + facial_params)
```

模型不直接控制窗口位置——`dx/dy` 继续由宿主从 plan point 中提取并做碰撞/限速处理。模型只负责**骨骼姿态和运动意图**。

---

## 2. 数据流

```
WorldState (每 50ms 一帧)
  │
  ▼
Encoder: 每帧编码为 d_model 维向量
  - pet 状态 (位置/速度/朝向/行为)
  - 候选表面 (最近的 4 个 walkable surface)
  - 鼠标状态
  - 点击事件
  - 场景状态 (全屏/暂停)
  │
  ▼
Causal Transformer
  - 输入: K=8 帧上下文 token
  - 自回归生成: H=12 帧未来姿态 (≈400ms @ 33ms/frame)
  - 因果 mask: 帧 t 只看到 ≤t 的帧
  │
  ▼
Decoder: d_model → (10 四元数 + Vec3 root + 5 面部)
  │
  ▼
宿主安全层 FK → 渲染
```

---

## 3. 输入编码

每帧编码为 48 维向量，通过 `Linear(48, d_model)` 投影到模型维度。

### 3.1 Pet 状态 (12 维)

| 字段 | 维度 | 编码 |
|---|---|---|
| foot_x, foot_y (相对屏幕中心) | 2 | float, normalized by screen width |
| vx, vy | 2 | float, clipped to [-2000, 2000] then /2000 |
| facing | 1 | -1 or 1 |
| behavior | 1 | scalar: idle=0, walk=1, jump=2, click_reaction=3, falling=4, landing=5 |
| current surface_id hash | 4 | one-hot of 16 hash buckets |
| 是否在表面上 | 1 | 0 or 1 |
| 当前表面 y 差值 | 1 | pet.foot_y - surface.y, /100 |

### 3.2 候选表面 (20 维)

取最近的 4 个 enabled, non-occluded 表面，每个编码 5 维:

| 字段 | 维度 | 编码 |
|---|---|---|
| y 差值 (相对 pet.foot_y) | 1 | float, /200 |
| x1 相对 pet.foot_x | 1 | float, /200 |
| x2 相对 pet.foot_x | 1 | float, /200 |
| 表面类型 | 1 | 0=work_area_floor, 1=window_top |
| 表面宽度 | 1 | log(width+1) / log(2000) |

不足 4 个表面时填充零向量 + 标记位 0。

### 3.3 鼠标 & 点击 (8 维)

| 字段 | 维度 | 编码 |
|---|---|---|
| cursor x, y (相对 pet.foot) | 2 | float, /200 |
| cursor over_pet | 1 | 0 or 1 |
| cursor left_down | 1 | 0 or 1 |
| pending click count | 1 | clipped to [0, 3], /3 |
| latest click age (ms) | 1 | /500 |
| latest click x, y (相对 pet) | 2 | float, /200 |

### 3.4 场景状态 (4 维)

| 字段 | 维度 |
|---|---|
| pet_allowed | 1 |
| fullscreen_active | 1 |
| generator_status | 1 (starting=0, ready=1, degraded=2) |
| 当前时间相位 | 1 (sin(time_ms/1000)) |

### 3.5 行为目标 (4 维) — 可选，训练时来自 teacher

| 字段 | 维度 |
|---|---|
| 目标行为 | 3 (one-hot: walk/jump/idle) |
| 目标 surface y | 1 (/200) |

---

## 4. 模型架构

```
CausalMotionTransformer
├── Input Projection: Linear(48, 256)
├── Learned Positional Encoding: [20, 256]
├── Transformer Blocks × 6
│   ├── Causal Self-Attention (n_heads=8, d_head=32)
│   ├── MLP (dim_feedforward=1024, GELU)
│   └── Pre-LayerNorm
├── Output Head: Linear(256, 48)
│   ├── quat_head: 40 dim → 10×4, normalize each
│   ├── root_head: 3 dim
│   └── facial_head: 5 dim
└── Total params: ~2.1M
```

**序列设计:**
- Token 0..7: 过去 8 帧的编码 (K=8 context)
- Token 8..19: 未来 12 帧 (H=12 output)
- Token 8 输入 teacher 的 "goal frame" 编码（训练时），推理时输入零向量或预测的目标
- Token 9..19: 训练时输入 shifted teacher poses，推理时使用自身预测

**训练时的 teacher forcing:**
```
Token 0..7: 真实 world state 编码
Token 8:    目标行为编码 (teacher 提供的 next behavior)
Token 9..19: 真实 teacher pose (shifted right by 1)
→ 预测: token 9..20 的 pose

推理时:
Token 0..7: 真实 world state 编码
Token 8:    预测的 goal embedding (也由模型输出)
Token 9..19: 自回归预测 (每步输出 fed 回输入)
```

---

## 5. 损失函数

### 5.1 FK 关节位置损失 (主损失)

将输出的 quaternion + root_translation 通过 FK 计算 3D 关节位置，与 teacher 的关节位置做 L2:

```python
def fk_position_loss(pred_quats, pred_root, target_joints_3d, skeleton):
    # FK forward pass with predicted quaternions
    pred_joints = fk_forward(skeleton, pred_quats, pred_root)
    # target_joints_3d is pre-computed from teacher quaternions
    return F.mse_loss(pred_joints, target_joints_3d)
```

**为什么比直接 quaternion L2 好：** FK 把旋转差异转化为关节位置的欧氏距离，对末端（手脚尾巴）的小旋转差异给出更大惩罚，对根关节的差异给出较小惩罚——符合视觉敏感度。

### 5.2 骨长一致性损失

```python
def bone_length_loss(pred_joints, skeleton):
    loss = 0
    for j, joint in enumerate(skeleton.joints):
        parent_idx = skeleton.parent_index[j]
        if parent_idx < 0: continue
        pred_len = torch.norm(pred_joints[j] - pred_joints[parent_idx])
        rest_len = skeleton.rest_length[j]
        loss += (pred_len - rest_len) ** 2
    return loss / skeleton.num_joints
```

### 5.3 速度平滑损失

```python
def velocity_smoothness_loss(poses):
    # poses: [B, H, 10, 4] — quaternion deltas over time
    vel = poses[:, 1:] - poses[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return acc.pow(2).mean()  # minimize jerk
```

### 5.4 总损失

```python
loss = (
    1.0  * fk_position_loss
    + 0.3 * bone_length_loss
    + 0.1 * velocity_smoothness_loss
)
```

---

## 6. 训练配置

| 参数 | 值 |
|---|---|
| 优化器 | AdamW (lr=3e-4, weight_decay=0.01) |
| 调度器 | Cosine warmup (1000 steps) → decay |
| Batch size | 256 |
| 训练步数 | 100,000 |
| 梯度裁剪 | 1.0 |
| 混合精度 | FP16 |
| 单次推理延迟 | < 10ms (GPU, batch=1) |
| 模型大小 | ~2.1M params, ~4.2MB FP16 |

---

## 7. 与 MotionBackend 接口集成

```python
class NeuralMotionBackend(TorchMotionBackend):
    name = "causal-transformer-v0"

    def __init__(self, checkpoint_path: str):
        self.model = CausalMotionTransformer(config)
        self.encoder = WorldStateEncoder()

    def prepare(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.cuda().eval()
        # Warmup
        dummy = torch.randn(1, 8, 48, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            self.model(dummy)

    def generate(self, world, seed, generated_at_ms) -> MotionPlan:
        # 1. Encode world state
        context = self.encoder.encode(world)  # [1, 8, 48]
        # 2. Autoregressive decode
        with torch.no_grad():
            poses = self.model.generate(context)  # [1, 12, 48]
        # 3. Convert to MotionPlan
        points = []
        for i in range(12):
            q_start = i * 48
            quats = poses[0, i, :40].reshape(10, 4)
            quats = F.normalize(quats, dim=-1)  # ensure unit quaternions
            root = poses[0, i, 40:43]
            facial = poses[0, i, 43:48]
            points.append(MotionPoint(
                t_ms=i * 33,
                dx=0, dy=0,  # host controls position via dx/dy
                vx=0, vy=0,
                facing=1,  # derived from root_rotation
                lean=0, squash=1, bob=0, expression="neutral",
                root_translation=tuple(root.tolist()),
                root_rotation=tuple(quats[0].tolist()),
                local_rotation_deltas=tuple(tuple(q.tolist()) for q in quats[1:]),
                facial_params={...},
            ))
        return MotionPlan(...)
```

---

## 8. 最低可运行验证

在没有任何训练数据的情况下，可以先验证模型管线：

1. 定义模型结构 → 跑一次随机前向 → 确认输出 shape 和数值范围正确
2. 用随机权重创建一个 `MotionPlan` → 通过协议发送给真实宿主 → FK 渲染器绘制（姿态会是垃圾，但管线通了）
3. 用模拟器生成 100 个样本 → 过拟合到一个样本 → 确认 loss 下降 → 确认能复现训练样本的动作
4. 扩展到完整训练集

---

## 9. 待确认

| 问题 | 当前假设 |
|---|---|
| 模型文件放在哪里？ | `services/generator/pet_generator/neural_motion/` |
| PyTorch checkpoint 格式？ | `model.pt` (state_dict only, no optimizer) |
| 是否需要 behavior 分类头？ | 是，作为辅助任务 |
| 训练是否需要 wandb？ | 可选 |
| 推理用 GPU 还是 CPU？ | GPU 优先 (CUDA), CPU fallback |
