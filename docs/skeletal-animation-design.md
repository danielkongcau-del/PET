# 骨骼动画 & 端到端动作模型：设计文档

状态：已评审 v2，并于 2026-07-21 更新为通用角色架构。猫是首个验证资产，不是协议特例。

**通用角色决议 (2026-07-21，覆盖本文中更早的猫专用示例)：**

- 每个具体角色拥有独立的 character rig manifest、归一化统计和 checkpoint。
- 同一套数据、FK、渲染、训练与推理代码必须支持不同物种、拓扑和 driven joint 数量。
- checkpoint 不跨角色复用；加载时必须精确匹配 `characterId`、`rigFingerprint` 和 `drivenJointOrder`。
- `cat-skeleton*.json` 仅作为旧格式兼容输入。当前权威入口是所选角色的 `character-rig-manifest-v1`。
- 模型输出采用 manifest 驱动的 N 个 local quaternion，不允许在代码、文档或 checkpoint 中写死 10、20 或 73。
- 通用性指角色物种、拓扑和 driven joint 数量可变；首版运行时输出画布统一归一化为 48×48，并固定 2 倍显示为 96 DIP，不把角色素材原始分辨率暴露为物理碰撞尺寸。

**历史评审记录 (2026-07-20)：** 下列条目描述最初的 2D 猫方案；凡与 2026-07-21 通用角色决议冲突之处，均以后者为准。
- 分层精灵：优先搜索现有免费 CC0 资产，搜索期间先写不依赖精灵的预备代码（协议层、Schema、能力协商）
- 能力协商：通过 `hello/ready` 的 `capabilities` 字段宣告 `"skeletal_motion"`
- 骨骼版本：生成器在 `ready` 中发送完整骨骼定义的 SHA-256，宿主对比本地 `cat-skeleton.json`，不匹配时拒绝 + 错误提示
- 尾巴物理：模型输出 tail_base 角度，tail_tip 由渲染器施加惯性跟随 + 阻尼二级物理
- 步态协调：模型从数据中自主学习（数据驱动），渲染器不做约束
- 表情系统：用连续面部参数（eye_scale, mouth_open, ear_angle 等），不再仅依赖离散 expression 字段
- **角色 manifest 为唯一真相源 (2026-07-21)**：任何代码不得硬编码角色名、骨骼数量或索引。当前选择的 character rig manifest 定义骨骼层级、rest transform、DOF/mask、精确 driven joint order、源资产映射和 checkpoint 身份；所有模块必须从同一 manifest 读取。

---

## 目录

1. [骨骼层级定义](#1-骨骼层级定义)
2. [FK 渲染管线](#2-fk-渲染管线)
3. [协议与宿主变更](#3-协议与宿主变更)
4. [桌面场景模拟器](#4-桌面场景模拟器)
5. [Causal Transformer 行为克隆](#5-causal-transformer-行为克隆)
6. [1D Flow Matching（Phase 3 预留）](#6-1d-flow-matchingphase-3-预留)
7. [实施路线与里程碑](#7-实施路线与里程碑)
8. [待确认项](#8-待确认项)

---

## 1. 骨骼层级定义

> §1.1-§1.3 是早期 2D 猫原型的兼容格式说明，不是新角色或神经模型的实现规范。当前规范从 §1.4 开始，以 character rig manifest 为准。

### 1.1 骨骼树（10 根）

```
root (pelvis, anchor at foot_x/foot_y)
├── spine (L1)
│   └── head (L1)
│       ├── ear_left (L0.5, no sprite)
│       └── ear_right (L0.5, no sprite)
├── upper_arm_left (L1)    ← FK: root → upper_arm
├── upper_arm_right (L1)
├── upper_leg_left (L1)
├── upper_leg_right (L1)
└── tail (L2, 2-segment chain)
    └── tail_tip
```

- `L{n}` = 从 parent joint 到自身 joint 的骨骼长度（单位：精灵像素）
- `ear_left/right` 不挂独立 sprite，仅影响 head sprite 的局部变形（stretch）
- `tail` 为 2 段链：`tail_base → tail_tip`，每段独立旋转

### 1.2 角色骨骼入口：`character-rig-manifest-v1`

下方 `cat-skeleton.json` 片段保留为 legacy 2D 格式说明。新角色必须通过 character rig manifest 声明可变长度 joint graph、精确 `drivenJointOrder`、DOF/mask、source mapping 和 checkpoint identity；Cat 默认 manifest 只是第一份实例。

```jsonc
{
  "schema": "pet-skeleton-v1",
  "canvas": [48, 48],
  "displayScale": 2,
  // 脚底锚点位于 skeleton space 中的位置
  "footAnchor": [24, 46],
  // 源素材的默认朝向（-1 = 面朝左）
  "sourceFacing": -1,
  "bones": [
    {
      "id": "root",
      "name": "骨盆/根骨",
      "parent": null,
      // rest pose 下 joint 在 canvas 中的位置
      "joint": [24, 34],
      "length": 8,
      // 相对 parent 的默认角度（弧度，0 = parent 方向）
      "restAngle": 0,
      // 该骨骼对应的 sprite layer（用于渲染）
      "sprite": null,
      // 运动范围限制 [min, max]（弧度）
      "limits": { "rotation": [-0.15, 0.15] }
    },
    {
      "id": "spine",
      "name": "脊柱",
      "parent": "root",
      "joint": [24, 26],
      "length": 10,
      "restAngle": 0,
      "sprite": "body",
      "limits": { "rotation": [-0.25, 0.25] }
    },
    {
      "id": "head",
      "name": "头部",
      "parent": "spine",
      "joint": [24, 16],
      "length": 10,
      "restAngle": 0,
      "sprite": "head",
      "limits": { "rotation": [-0.45, 0.45] }
    },
    {
      "id": "ear_left",
      "name": "左耳",
      "parent": "head",
      "joint": [18, 8],
      "length": 5,
      "restAngle": -0.5,
      "sprite": null,
      "limits": { "rotation": [-0.1, 0.1] }
    },
    {
      "id": "ear_right",
      "name": "右耳",
      "parent": "head",
      "joint": [30, 8],
      "length": 5,
      "restAngle": 0.5,
      "sprite": null,
      "limits": { "rotation": [-0.1, 0.1] }
    },
    {
      "id": "upper_arm_left",
      "name": "左上臂",
      "parent": "spine",
      "joint": [18, 26],
      "length": 8,
      "restAngle": 0.5,
      "sprite": "upper_arm_left",
      "limits": { "rotation": [-0.3, 0.6] }
    },
    {
      "id": "upper_arm_right",
      "name": "右上臂",
      "parent": "spine",
      "joint": [30, 26],
      "length": 8,
      "restAngle": -0.5,
      "sprite": "upper_arm_right",
      "limits": { "rotation": [-0.6, 0.3] }
    },
    {
      "id": "upper_leg_left",
      "name": "左腿",
      "parent": "root",
      "joint": [20, 38],
      "length": 9,
      "restAngle": 0.3,
      "sprite": "upper_leg_left",
      "limits": { "rotation": [-0.35, 0.5] }
    },
    {
      "id": "upper_leg_right",
      "name": "右腿",
      "parent": "root",
      "joint": [28, 38],
      "length": 9,
      "restAngle": -0.3,
      "sprite": "upper_leg_right",
      "limits": { "rotation": [-0.5, 0.35] }
    },
    {
      "id": "tail_base",
      "name": "尾根",
      "parent": "root",
      "joint": [24, 38],
      "length": 10,
      "restAngle": 0.6,
      "sprite": "tail_base",
      "limits": { "rotation": [-0.7, 0.7] }
    },
    {
      "id": "tail_tip",
      "name": "尾尖",
      "parent": "tail_base",
      "joint": [30, 44],
      "length": 8,
      "restAngle": 0.3,
      "sprite": "tail_tip",
      "limits": { "rotation": [-0.5, 0.5] }
    }
  ],
  // 渲染层顺序（从后往前画）
  "drawOrder": [
    "tail_tip",
    "tail_base",
    "upper_leg_left",
    "upper_leg_right",
    "upper_arm_left",
    "upper_arm_right",
    "body",
    "head"
  ]
}
```

### 1.3 骨骼姿态表示

生成器输出的姿态向量（模型负责 7 个关节旋转 + 面部参数；tail_tip 由渲染器物理驱动）：

```python
# 模型输出: [root_dx, root_dy, θ_spine, θ_head, θ_arm_l, θ_arm_r, θ_leg_l, θ_leg_r, θ_tail_base,
#            eye_scale, eye_squint, mouth_open, ear_angle, brow_tilt]
# 共 14 维
pose_dim = 14

# 渲染器内部额外计算:
# tail_tip = tail_physics(tail_base)  ← 不在模型输出中

class BonePose:
    rotations: tuple[float, ...]    # 7 个关节旋转角（弧度），不含 tail_tip
    root_offset: tuple[float, float]  # root 相对 foot anchor 的偏移
    facial: FacialParams             # 5 个连续面部参数
```

当前 `PlanPoint` 的 `lean/squash/bob/expression` 继续保留。新增 `bone_rotations` 数组（7 个值）和 `facial_params` 对象。

### 1.4 角色 manifest 为唯一真相源

所选角色的 character rig manifest 是骨骼层级、父子关系、可驱动关节顺序和 checkpoint 身份的**唯一权威来源**。任何模块不得硬编码角色名称、骨骼数量、骨骼名称或骨骼索引。旧 `cat-skeleton*.json` 只能由兼容适配器转换，不能再作为新训练数据的身份来源。

**数据来源规则：**

```
character-rig-manifest-v1
  ├── rig.drivenJointOrder                         → 模型输出的精确关节顺序与 N
  ├── rig.joints[*].parentIndex                    → FK 父子拓扑
  ├── rig.joints[*].restLocal                      → authored rest transform
  ├── rig.joints[*].dofMask / masks                → model/secondary/static 与监督 mask
  ├── rig.joints[*].semanticRole                   → 可选语义，不决定数组长度
  ├── source / trainingClips                       → mesh、skin、IBM、动画及 provenance
  └── checkpoint                                   → characterId + rigFingerprint 绑定
  └── bones[*].limits                             → 运行时角度裁剪
```

**各模块读取义务：**

| 模块 | 从 character manifest 读取 | 当前状态（2026-07-21） |
|---|---|---|
| `character-rig.ts` / `character_rig.py` | 严格校验 rig fingerprint、层级、driven order、mask 与 checkpoint 身份 | ✅ 已实现，保留 legacy 适配器 |
| `generator-bridge.ts` | 将同一 rig fingerprint 与生成器协商，并把精确 driven order 交给宿主安全层 | ✅ 已实现 |
| `protocol.ts` / Python protocol | 验证 3D 原子字段组与最多 128 个 local quaternion | ✅ 已实现 |
| `motion-controller.ts` | 按所选角色的 `drivenJointOrder.length` 校验每个 PlanPoint | ✅ 已实现 |
| `pet-window.ts` | 将同一份已验证 manifest 和角色资源发送给 renderer | ✅ 已实现 |
| `renderer.js` | 任意拓扑四元数 FK、正交侧视投影、pose-aware sprite/debug fallback | ✅ 自动 2D joint warp 已实现；真正的 skinned-mesh 绘制仍未实现 |
| generator planner | 从所选 manifest 读取 N 并输出 N 个 local quaternion | ✅ 已实现；运行时 procedural backend 仍只是 identity 姿态基线 |
| training data teacher | 将角色动画转换为 rest-local delta，并与程序化桌面行为轨迹合成 | ✅ 已实现；循环接缝、Schema、clip/order/fingerprint 与逐样本 provenance 均 fail closed |
| neural checkpoint loader | 精确校验 `characterId + rigFingerprint + drivenJointOrder` 后推理 | ⏳ 尚未实现；当前只有 bundle/路径契约 |

**关键不变量：**

1. `modelDrivenBoneCount = drivenJointOrder.length`；所有模块读取同一 manifest，不得各自猜测或按 role 重新筛选。
2. `local_rotation_deltas[i]` 精确对应 `drivenJointOrder[i]`；顺序是 checkpoint ABI 的一部分。
3. 每个角色 checkpoint 必须精确匹配 `characterId + rigFingerprint + drivenJointOrder`，任何差异都拒绝加载。
3. 宿主加载 skeleton 后立即计算 `modelDrivenBoneIds: string[]`，用于校验和日志

**Legacy 示例：早期 10 骨 skeleton 中 model_driven 骨骼共 7 个。** 新实现不得根据此表推导 N；例如当前 Cat 源资产是 74 个完整 joint、30 个 driven joint，其他角色可以不同。

| 索引 | bone ID | sprite |
|---|---|---|
| 0 | spine | body |
| 1 | head | head |
| 2 | upper_arm_left | upper_arm_left |
| 3 | upper_arm_right | upper_arm_right |
| 4 | upper_leg_left | upper_leg_left |
| 5 | upper_leg_right | upper_leg_right |
| 6 | tail_base | tail_base |

`tail_tip` 为 secondary，`root`、`ear_left`、`ear_right` 不计入模型输出。

---

## 2. FK 渲染管线

### 2.1 渲染流程（Canvas 2D）

```
for each world_state tick:
    1. 从 PlanPoint 读取 bone_rotations[0..7]
    2. FK 正向传递（root → spine → head → ears, arms, legs → tail_base → tail_tip）
       - 每个 joint 世界位置 = parent_joint + rotate(length, parent_world_angle + restAngle + rotation)
    3. 按 drawOrder 逐层绘制：
       - context.save()
       - context.translate(joint_world_x, joint_world_y)
       - context.rotate(parent_world_angle + restAngle + rotation)
       - 如果该骨骼有 sprite：drawImage(sprite, -pivot_x, -pivot_y)
       - 如果没有 sprite：跳过（如 ear_left/right 仅影响 head 的变换）
       - context.restore()
    4. 叠加 deck 变换（整体 facing mirror、全局 squash/bob）
```

### 2.2 与现有 `PetVisualState` 的集成

现有 `PetVisualState`（`pet-window.ts:34-44`）：

```typescript
interface PetVisualState {
  facing: -1 | 1;
  lean: number;
  squash: number;
  bob: number;
  expression: string;
  behavior: Behavior;
  generatorStatus: GeneratorStatus;
  debug: boolean;
}
```

**变更：** 新增 `boneRotations: Float64Array`（8 个值）：

```typescript
interface PetVisualState {
  // ...现有字段不变...
  boneRotations?: Float64Array;  // 8 个关节旋转角
  // 回退兼容：当 boneRotations 为空时，使用 lean/squash/bob 做旧的整体变形
}
```

### 2.3 精灵获取策略

**决策：优先搜索现有 CC0 分层猫精灵，程序化几何作为 fallback。**

#### 搜索目标规格

| 要求 | 值 |
|---|---|
| 画布尺寸 | ≥ 48×48（可缩小，不可放大） |
| 层数 | ≥ 6：头、身体、左臂、右臂、左腿、右腿、尾巴 |
| 层间重叠 | ≥ 2px（关节处有额外像素覆盖接缝） |
| 格式 | PNG with alpha |
| 授权 | CC0 / Public Domain / MIT |

#### 搜索渠道（按优先级）

1. **[itch.io](https://itch.io/game-assets/tag-pixel-art/tag-cat/tag-sprites?license=cc0) — Game Assets → tag: pixel-art, cat, sprites, CC0**
2. **[OpenGameArt](https://opengameart.org/art-search-advanced?field_art_tags=pixel+cat&field_art_licenses%5B%5D=4) — 搜索 pixel+cat, CC0/Public Domain**
3. **craftpix.net — 搜索 "cat sprite sheet free"（注意授权条款）**
4. **GitHub — 搜索 "cat sprite sheet" repo（MIT/GPL）**

#### 评估标准

对每个候选精灵检查：
- [ ] 独立图层 or 足够空间分离裁切
- [ ] 行走/跳跃/待机多 frame
- [ ] 关节 pivot 位置清晰
- [ ] 风格与现有参考图（`cat-reference.png`）可辨识一致
- [ ] 授权明确允许修改和再分发

#### Fallback：程序化几何

如果 48 小时内找不到合适精灵，启动 §2.4 的程序化几何管线。渲染器代码不变——`drawSpriteCat()` 和 `drawProceduralCat()` 是同一渲染循环的两个绘制策略，切换只需改一行配置。

详细程序化几何方案见下节。

### 2.4 尾巴二级物理

模型输出 `tail_base` 的旋转角。`tail_tip` 由渲染器施加惯性跟随 + 阻尼效果。

```javascript
// renderer.js — tail physics (per-frame update)
const tailPhysics = {
  tipAngle: 0,           // 当前 tail_tip 实际角度
  tipAngularVelocity: 0, // 角速度
};

function updateTailPhysics(baseAngle, dt) {
  const DAMPING = 0.82;       // 阻尼系数（0-1，越小跟随越快）
  const STIFFNESS = 0.12;     // 刚度（回正力度）
  const INERTIA_FACTOR = 0.25; // 惯性系数

  // 弹簧-阻尼模型：tail_tip 跟随 tail_base，但有延迟和过冲
  const targetAngle = baseAngle * 1.3; // 尾尖相对尾根放大
  const springForce = (targetAngle - tailPhysics.tipAngle) * STIFFNESS;
  const dampingForce = -tailPhysics.tipAngularVelocity * DAMPING;

  tailPhysics.tipAngularVelocity += (springForce + dampingForce) * dt * 60;
  tailPhysics.tipAngularVelocity += (baseAngle - prevBaseAngle) * INERTIA_FACTOR; // 惯性
  tailPhysics.tipAngle += tailPhysics.tipAngularVelocity * dt * 60;

  // 运动范围限制
  tailPhysics.tipAngle = clamp(tailPhysics.tipAngle, -0.9, 0.9);

  return tailPhysics.tipAngle;
}
```

**效果：** 猫快速转身时尾巴会先被惯性甩向反方向，然后逐渐跟随。静止时尾巴会轻微摆动（如果 base 有小幅振动）。所有这些都是纯渲染器行为——模型不需要预测尾尖位置。

### 2.5 表情系统：连续面部参数

从离散 `expression` 迁移到连续面部参数，与骨骼旋转角一起由模型输出。

```typescript
// 新增：PlanPoint 中的面部参数字段
interface FacialParams {
  eye_scale?: number;      // 眼睛大小 [0.5, 1.5]  → surprised=大, sleepy=小
  eye_squint?: number;     // 眯眼程度 [0, 1]      → annoyed/ curious
  mouth_open?: number;     // 张嘴程度 [0, 1]      → surprised/ happy
  ear_angle?: number;      // 耳朵旋转角 [-0.5, 0.5] → scared=flat, curious=perked
  brow_tilt?: number;      // 眉毛倾斜 [-1, 1]     → annoyed=down, sad=up
}
```

**现有 `expression` 字符串保留作为高层标签**（"neutral", "surprised", "happy", "annoyed", "curious", "focused", "sleepy", "relieved"），但渲染时优先使用 `facial_params`。映射关系：

```javascript
// renderer.js — expression → facial_params fallback
const EXPRESSION_DEFAULTS = {
  neutral:    { eye_scale: 1.0, eye_squint: 0, mouth_open: 0, ear_angle: 0, brow_tilt: 0 },
  surprised:  { eye_scale: 1.35, eye_squint: 0, mouth_open: 0.7, ear_angle: -0.2, brow_tilt: 0.3 },
  happy:      { eye_scale: 1.15, eye_squint: 0.3, mouth_open: 0.4, ear_angle: 0, brow_tilt: 0.1 },
  annoyed:    { eye_scale: 0.9, eye_squint: 0.6, mouth_open: 0, ear_angle: 0.3, brow_tilt: -0.5 },
  curious:    { eye_scale: 1.1, eye_squint: 0, mouth_open: 0.1, ear_angle: 0, brow_tilt: 0.2 },
  focused:    { eye_scale: 1.0, eye_squint: 0.3, mouth_open: 0, ear_angle: 0, brow_tilt: -0.15 },
  sleepy:     { eye_scale: 0.65, eye_squint: 0.8, mouth_open: 0, ear_angle: -0.4, brow_tilt: 0 },
  relieved:   { eye_scale: 1.05, eye_squint: 0, mouth_open: 0.2, ear_angle: 0, brow_tilt: 0.15 },
};
```

这样向后兼容：旧生成器只发 `expression` 字符串时，渲染器查表得到默认参数；一旦新生成器显式发送稀疏 `facial_params` 对象，未出现的通道一律取中性值，而不是继承上一帧或继续叠加 `expression`。下一条 visual state 未携带 `facialParams` 时，renderer 会清除旧覆盖并重新使用当前 `expression` 默认值。

当前角色 manifest 还没有角色专属的眼、嘴、耳和眉形变绑定。因此 renderer 提供一个明确的 **generic facial fallback**：把五个连续通道确定性地映射为以脚底锚点为中心的细微整体缩放、旋转和垂直起伏，并把最终缩放限制在 `[0.95, 1.06]`、旋转限制在 `±0.035 rad`、垂直偏移限制在 `±0.75 px`。这层 fallback 包裹普通 sprite、骨骼驱动的 pose-aware sprite 和 debug skeleton 三条绘制路径，所以协议字段不会只停留在 IPC；它不冒充真正的角色面部绑定。角色以后提供专属 face rig/mesh controls 时，应由专属绑定替换这层整体变换，避免重复应用。

### 2.6 程序化几何原型（fallback，仅在无分层精灵时激活）

当 `cat-48.png` 不存在或 sprite 为空时，每个骨骼用 Canvas 直接绘制几何形状：

| 骨骼 | 形状 | 颜色 |
|---|---|---|
| root/pelvis | 椭圆 (6×4) | #2d2d2d |
| spine/body | 圆角矩形 (10×14) | #f5f0e8 |
| head | 椭圆 (14×12) + 三角耳 | #f5f0e8 / #2d2d2d |
| upper_arm ×2 | 圆角矩形 (5×9) | #f5f0e8 |
| upper_leg ×2 | 圆角矩形 (6×10) | #f5f0e8 |
| tail_base/tail_tip | 曲线 (3px 宽贝塞尔) | #2d2d2d |

这保证在没有任何外部资产的情况下，管线可以完整跑通并生成可辨识的猫。

### 2.4 渲染器文件变更

```
desktop/src/renderer/renderer.js    ← 主要变更：FK + 逐层绘制
desktop/src/renderer/styles.css     ← 不变
desktop/src/renderer/index.html     ← 不变
```

新增：

```
assets/pet/runtime/cat-skeleton.json   ← 骨骼定义
assets/pet/runtime/cat-skeleton.schema.json  ← JSON Schema
```

---

## 3. 协议与宿主变更

### 3.1 协议版本策略

**v1 不变。** 骨骼姿态通过扩展现有字段承载：

```typescript
// PlanPoint 新增可选字段（v1 向后兼容）
interface PlanPoint {
  // ...现有字段不变...
  bone_rotations?: number[];  // 8 个 float，范围 [-π, π]
}
```

- 旧宿主忽略未知字段 → 回退到 lean/squash/bob 整体变形
- 新宿主读取 `bone_rotations` → FK 渲染
- 两个路径共存，平滑过渡

**将来 v2（如果需要）：**
- 增加独立的 `PosePlan` 消息类型
- 扩展 `capabilities` 协商（`"skeletal_motion"`）

### 3.2 骨骼能力协商

为满足「双方不匹配时应有 warning」的需求，通过 `capabilities` 字段显式协商。

**握手流程：**

```
Host hello:
  capabilities: [..., "skeletal_motion"]   ← 宿主宣告支持骨骼渲染

Generator ready:
  capabilities: [..., "skeletal_motion"]   ← 生成器宣告能输出骨骼姿态
```

**匹配矩阵：**

| 宿主 \\ 生成器 | 有 `skeletal_motion` | 无 `skeletal_motion` |
|---|---|---|
| 有 `skeletal_motion` | ✅ FK 渲染 | ⚠️ 降级为整体变形 + WARN 日志 |
| 无 `skeletal_motion` | ⚠️ 忽略 `bone_rotations` + WARN 日志 | ✅ 整体变形 |

**实现细节：**

```typescript
// generator-bridge.ts — 收到 ready 后
function detectSkeletalMode(helloCaps: string[], readyCaps: string[]): SkeletalMode {
  const hostSkeletal = helloCaps.includes("skeletal_motion");
  const genSkeletal = readyCaps.includes("skeletal_motion");

  if (hostSkeletal && genSkeletal) return "full";
  if (hostSkeletal && !genSkeletal) {
    warn("protocol", "Host supports skeletal motion but generator does not; falling back to whole-sprite deformation.");
    return "host_only";
  }
  if (!hostSkeletal && genSkeletal) {
    warn("protocol", "Generator outputs bone rotations but host renderer will ignore them.");
    return "generator_only";
  }
  return "none";
}
```

生成器侧同步检查：

```python
# service.py — _handle_hello 中
def _handle_hello(self, envelope, writer):
    ...
    self._skeletal_enabled = "skeletal_motion" in envelope.payload.get("capabilities", [])
    if not self._skeletal_enabled:
        LOGGER.warning("Host does not advertise skeletal_motion; bone rotations will not be rendered.")
```

- **`full` 模式：** 生成器输出 `bone_rotations`，渲染器执行 FK。
- **降级模式：** 生成器跳过 `bone_rotations`，只输出 `lean/squash/bob`；宿主只用整体变形。
- 所有不匹配情况都会在宿主日志 + 生成器 stderr 中产生 **WARN** 级别日志，托盘状态不变（不弹窗干扰用户）。

### 3.3 骨骼版本协商（完整哈希校验）

**生成器在 `ready` 消息中携带它训练时使用的骨骼定义 SHA-256。**

```typescript
// ready 消息扩展
interface ReadyPayload {
  // ...现有字段...
  generator: RuntimeInfo & {
    // ...existing...
    skeleton_sha256?: string;  // 新增
  };
}
```

**宿主侧校验逻辑：**

```typescript
// generator-bridge.ts — onReady
function verifySkeletonCompatibility(ready: ReadyPayload): SkeletonVerification {
  const genSkeletonHash = ready.generator.skeleton_sha256;
  const localSkeletonHash = loadSkeletonSha256();  // 对 cat-skeleton.json 计算 SHA-256

  if (!genSkeletonHash && !localSkeletonHash) {
    // 双方都没有骨骼定义 → 整体变形模式，no warning
    return { compatible: true, mode: "whole_sprite" };
  }
  if (!genSkeletonHash && localSkeletonHash) {
    warn("skeleton", "Host has skeleton definition but generator was not trained with one; FK disabled.");
    return { compatible: false, mode: "whole_sprite" };
  }
  if (genSkeletonHash && !localSkeletonHash) {
    warn("skeleton", "Generator expects skeleton but host has no cat-skeleton.json; install the asset or update the generator.");
    return { compatible: false, mode: "whole_sprite" };
  }
  if (genSkeletonHash !== localSkeletonHash) {
    warn("skeleton", `Skeleton hash mismatch: generator=${genSkeletonHash.slice(0,12)}..., host=${localSkeletonHash.slice(0,12)}...`);
    // 不阻断运行，但骨骼姿态将被忽略
    return { compatible: false, mode: "whole_sprite" };
  }
  return { compatible: true, mode: "skeletal" };
}
```

**关键设计考量：**
- 哈希对 `cat-skeleton.json` 的**规范化 JSON**（sorted keys, no whitespace）计算，避免格式差异导致误判。
- 不匹配时**不阻断运行**——降级到整体变形模式并打 WARN 日志。这保证了开发迭代中不会因为骨架微调而频繁崩溃。
- 生产环境中，生成器 checkpoint 和 `cat-skeleton.json` 应一起分发。

### 3.4 宿主变更汇总

| 文件 | 变更 | 风险 |
|---|---|---|
| `pet-window.ts` | `PetVisualState` 新增 `boneRotations` | 低：新增可选字段 |
| `motion-controller.ts` | `#tick()` 从 PlanPoint 提取 bone_rotations 传入 visual state | 低：只读新字段 |
| `protocol.ts` | `isPlanPoint()` 放行 `bone_rotations` 字段 | 低：新增允许字段 |
| `renderer.js` | 完整的 FK 渲染管线 | 中：核心变更，需回退路径 |
| `preload.cjs` | 不变 | — |
| `cat-skeleton.json` | 新增 | — |
| `cat-parts.json` | 新增 `skeleton` 引用 | 低 |

---

## 4. 桌面场景模拟器

### 4.1 设计目标

- 纯 Python，不依赖 Electron/Windows
- 确定性物理（镜像 `geometry.ts` 的碰撞/重力/表面支撑逻辑）
- 输出与真实 `WorldState` / `HorizonPlanPayload` 结构一致的训练样本
- 速度 > 1000× 实时（每个 episode 在 100ms 内完成）

### 4.2 架构

```python
# services/generator/pet_generator/simulator.py

class DesktopSimulator:
    """Offline desktop environment for trajectory data generation."""

    def __init__(
        self,
        seed: int,
        world_state_dt_ms: int = 50,
        plan_dt_ms: int = 33,
        motion_tick_ms: int = 16,
    ):
        self.rng = random.Random(seed)
        self.world_state_dt_ms = world_state_dt_ms
        self.plan_dt_ms = plan_dt_ms
        self.motion_tick_ms = motion_tick_ms
        self.displays: list[SimDisplay] = []
        self.windows: list[SimWindow] = []
        self.surfaces: list[SimSurface] = []
        self.pet: SimPet = ...
        self.cursor: SimCursor = ...
        self.time_ms: int = 0

    def reset(self, scenario: ScenarioConfig) -> WorldState:
        """随机生成一个桌面场景。返回初始 WorldState。"""
        self.displays = self._random_displays(scenario)
        self.windows = self._random_window_layout(scenario)
        self.surfaces = self._compute_surfaces()
        self.pet = self._random_pet_placement()
        return self._to_world_state()

    def advance_plan(self, plan: MotionPlan) -> WorldState:
        """将计划执行到下一个 WorldState 采样点并返回该状态。
        物理模拟包括：
        - 重力 (980 px/s²)
        - 碰撞检测 (与 geometry.ts 的 findCrossedSurface 等价)
        - 表面支撑
        - 工作区边界限制
        """
        elapsed_ms = 0
        while elapsed_ms < self.world_state_dt_ms:
            dt_ms = min(self.motion_tick_ms, self.world_state_dt_ms - elapsed_ms)
            elapsed_ms += dt_ms
            # generated_at_ms 是计划时间原点；关键帧之间按墙钟时间线性插值。
            action = sample_plan_linear(plan, elapsed_ms)
            self.pet.apply(action, dt_ms)
            self.pet = self._physics_step(self.pet, dt_ms)
        # 窗口可能移动
        self._maybe_move_windows()
        # 重新计算表面
        self.surfaces = self._compute_surfaces()
        self.time_ms += self.world_state_dt_ms
        return self._to_world_state()

    def _physics_step(self, pet: SimPet, dt_ms: int) -> SimPet:
        """确定性物理：镜像 motion-controller.ts 的 applyFallback 逻辑。"""
        ...

    def _compute_surfaces(self) -> list[SimSurface]:
        """镜像 surface-tracker.ts 的 buildSurfaceSnapshot 逻辑。"""
        ...
```

模拟器明确区分三个时钟：`world_state_dt_ms=50` 是环境观测/重规划节拍，
`plan_dt_ms=33` 是计划关键帧间隔，`motion_tick_ms=16` 是计划执行和物理推进的
确定性子步长。数据集 manifest 必须分别记录 `worldStateDtMs`、`planDtMs`
和 `executionClock`；修改任一时钟或采样语义都会改变数据集 ABI。

### 4.3 场景生成器（ScenarioConfig）

```python
@dataclass
class ScenarioConfig:
    num_displays: tuple[int, int] = (1, 3)       # 显示器数量范围
    display_scales: list[float] = (1.0, 1.25, 1.5, 2.0)
    num_windows: tuple[int, int] = (3, 12)        # 窗口数量范围
    window_size_range: tuple[int, int] = (300, 1900)  # 窗口尺寸范围 (px)
    pet_start: Literal["floor", "random_window", "mixed"] = "mixed"
    events: list[SimEvent] = []                   # 点击、窗口移动、全屏等事件
    duration_ms: int = 30_000                     # 单 episode 时长
```

### 4.4 训练数据生成策略

使用当前的 `AutoregressiveMotionBackend` 作为 **teacher**，生成 ground-truth 轨迹：

```python
def generate_training_sample(sim: DesktopSimulator, backend: MotionBackend, seed: int):
    """生成一个 (condition, target) 训练样本。"""
    sim.reset(scenario)
    condition_frames = []  # 过去 K 帧 world state
    target_poses = []       # 未来 H 帧骨骼姿态

    # Warmup: 跑 K 帧收集条件
    for _ in range(K):
        plan = backend.generate(sim.to_world_state(), seed)
        state = sim.advance_plan(plan)
        condition_frames.append(state)

    # 生成目标
    for _ in range(H):
        plan = backend.generate(sim.to_world_state(), seed)
        state = sim.advance_plan(plan)
        # 监督目标保留 teacher 的完整原始 horizon，不是 50 ms 环境采样快照。
        target_poses.append(plan_to_bone_poses(plan))

    return TrainingSample(
        condition=encode_condition(condition_frames),
        target=torch.tensor(target_poses, dtype=torch.float32),
        metadata={"scenario": sim.scenario, "seed": seed},
    )
```

### 4.5 骨骼姿态映射（action → bone pose）

当前 procedural planner 输出的是 `dx/dy/vx/vy/lean/squash/bob`。需要一个确定性的映射函数将其转为骨骼旋转角。这个映射可以在原型阶段手工设计：

```python
def action_to_bone_pose(plan_point: PlanPoint, skeleton: Skeleton) -> BonePose:
    """将高层运动参数映射到骨骼旋转角。"""
    return BonePose(
        rotations=(
            # spine: lean 影响身体倾斜
            plan_point.lean * 0.3,
            # head: 看向移动方向
            plan_point.lean * 0.15,
            # arms: 行走时前后摆臂
            math.sin(plan_point.bob * 0.5) * 0.4,
            -math.sin(plan_point.bob * 0.5) * 0.4,
            # legs: 行走步态
            math.sin(plan_point.bob) * 0.45,
            -math.sin(plan_point.bob) * 0.45,
            # tail_base: 身体倾斜 + 惯性
            -plan_point.lean * 0.3 + plan_point.vx * 0.0003,
            # tail_tip: 尾部跟随 + 延迟
            -plan_point.lean * 0.15 + plan_point.vx * 0.0005,
        ),
        root_offset=(plan_point.dx, plan_point.dy)
    )
```

这个映射函数只在 **训练数据生成阶段** 使用。训练好的神经网络会直接输出骨骼旋转角，不再需要这个手工映射。

---

## 5. Causal Transformer 行为克隆

### 5.1 模型架构

```
                    ┌─────────────────────────┐
Condition (K frames)│  Causal Transformer      │
┌──────────────────┤                          ├──→ Future Poses (H frames)
│ past_poses       │  d_model = 256            │   [dx, dy, θ₀..θ₇]
│ surface_feats    │  n_layers = 6             │
│ cursor_feat      │  n_heads = 8              │   每帧 10 维
│ event_feat       │  dim_ff = 1024            │
│ style_embed      │  dropout = 0.1            │
└──────────────────┤                          │
                   │  ~2.1M params             │
                   │  FP16: ~4.2 MB            │
                   └─────────────────────────┘

输入:  [K, d_input]   where K=8, d_input≈64
输出:  [H, d_pose]    where H=12, d_pose=14 (2 位置 + 7 旋转 + 5 面部)
```

### 5.2 输入特征编码

```python
class WorldStateEncoder:
    """将世界状态编码为固定维度向量。"""

    def encode(self, state: WorldState) -> Tensor:  # [d_enc]
        # 猫自身状态（10 维）
        pet_feat = [state.pet.vx, vy, facing, surface_id_hash, ...]

        # 候选表面编码（取最近 4 个表面，每个 6 维）
        surface_feat = [surface.y_diff, x1_relative, x2_relative,
                        kind_onehot, window_height, surface_width]

        # 鼠标/事件（4 维）
        cursor_feat = [cursor_x_rel, cursor_y_rel, left_down, over_pet]

        # 点击事件（4 维）
        click_feat = [has_pending_click, click_age_ms, click_x_rel, click_y_rel]

        return concat([pet_feat, surface_feat, cursor_feat, click_feat])
```

### 5.3 训练配置

```python
@dataclass
class TrainConfig:
    # 数据
    num_episodes: int = 50_000
    episode_duration_ms: int = 30_000
    context_frames: int = 8        # K
    horizon_frames: int = 12       # H，对齐现有 plan_horizon_ms=400, plan_dt_ms=33

    # 模型
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # 训练
    batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0

    # 损失
    position_loss_weight: float = 1.0
    rotation_loss_weight: float = 0.5
    velocity_consistency_weight: float = 0.1
```

### 5.4 损失函数

```python
def compute_loss(pred: Tensor, target: Tensor) -> Tensor:
    # pred, target: [B, H, 10]

    # L2 位置损失 (dx, dy)
    pos_loss = F.mse_loss(pred[..., :2], target[..., :2])

    # 角度损失（用 circular loss 避免 -π/+π 边界问题）
    rotation_diff = pred[..., 2:] - target[..., 2:]
    rotation_loss = (1 - torch.cos(rotation_diff)).mean()

    # 速度连续性损失（相邻帧之间）
    velocity = pred[:, 1:, :2] - pred[:, :-1, :2]
    target_velocity = target[:, 1:, :2] - target[:, :-1, :2]
    velocity_loss = F.mse_loss(velocity, target_velocity)

    return pos_loss + 0.5 * rotation_loss + 0.1 * velocity_loss
```

### 5.5 与现有 MotionBackend 接口的集成

```python
class NeuralMotionBackend(TorchMotionBackend):
    name = "causal-transformer-v0"

    def __init__(self, checkpoint_path: str):
        self.model = CausalMotionTransformer(config)
        self.encoder = WorldStateEncoder()
        self.context_buffer = deque(maxlen=8)  # 滑动窗口

    def prepare(self):
        """加载权重 + GPU warmup。在 hello 阶段调用。"""
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.cuda().eval()
        # Warmup inference
        dummy = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            self.model(dummy)

    def generate(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        context = self._build_context(world)  # [1, 8, d_enc]
        with torch.no_grad():
            poses = self.model(context)  # [1, 12, 10]

        # 将预测的 delta 姿态转换为 PlanPoint 列表
        points = []
        for i in range(12):
            dx, dy = poses[0, i, 0].item(), poses[0, i, 1].item()
            rotations = poses[0, i, 2:].tolist()
            # 反推 lean/squash/bob 用于回退兼容
            lean = rotations[0] / 0.3  # spine rotation → lean
            points.append(MotionPoint(
                t_ms=i * 33,
                dx=dx, dy=dy,
                vx=..., vy=...,
                facing=...,
                lean=lean,
                squash=1.0,
                bob=abs(rotations[4]),
                expression="neutral",
            ))

        return MotionPlan(
            plan_id=f"nn-{world.seq}-{seed:016x}",
            based_on_seq=world.seq,
            behavior="walk",  # 模型输出中可包含 behavior 分类头
            generated_at_ms=generated_at_ms,
            valid_until_ms=generated_at_ms + 400,
            dt_ms=33,
            confidence=0.85,
            seed=seed,
            points=tuple(points),
        )

    def cancel(self, plan_id: str | None = None) -> bool:
        self.context_buffer.clear()
        return True
```

关键点：**`NeuralMotionBackend` 的输出结构和 `AutoregressiveMotionBackend` 完全一致**，宿主的 `SafePlanExecutor` 和 `MotionController` 不需要任何修改。

---

## 6. 1D Flow Matching（Phase 3 预留）

Phase 2 的 causal transformer 验证闭环可行性后，Phase 3 替换为 flow matching：

```python
class FlowMatchingMotionBackend(TorchMotionBackend):
    name = "flow-matching-v0"

    def generate(self, world, seed, generated_at_ms):
        # 1. 编码条件
        context = self.encoder.encode(world)

        # 2. 采样噪声
        z = torch.randn(1, H, pose_dim, device="cuda")

        # 3. 多步 ODE 积分（DDIM 4 步或 Euler 8 步）
        trajectory = self.solver.integrate(
            self.model, z, context, num_steps=4
        )

        # 4. 转换为 MotionPlan（同 causal transformer）
        return self._trajectory_to_plan(trajectory, world, seed, generated_at_ms)
```

训练时用条件 flow matching loss：

```python
def flow_matching_loss(model, x0, x1, context):
    t = torch.rand(batch_size, 1, 1)
    xt = (1 - t) * x0 + t * x1  # 线性插值路径
    vt_pred = model(xt, t, context)
    vt_target = x1 - x0
    return F.mse_loss(vt_pred, vt_target)
```

---

## 7. 当前实施路线与里程碑

| 阶段 | 当前状态 | 完成定义 |
|---|---|---|
| 2A 通用角色契约 | ✅ 已完成首个可用版本 | manifest/schema/TS/Python 校验一致；任意 joint 名称与可变 N；Cat 只是 fixture |
| 2B 源资产提取 | ✅ Cat 首个资产已提取 | rest TRS、skin/IBM、mesh 引用、30Hz 非 identity 动画和 provenance 可复现 |
| 2C 宿主与 FK | ✅ FK 与自动 2D warp 完成；真蒙皮未完成 | capability 协商、精确 rig fingerprint、四元数 FK、侧视投影、正常 sprite 最终像素变化均有测试 |
| 2D 模拟器与数据 ABI | ✅ 已完成首个版本 | 固定原点、K/H/dt、每角色 rig/order、动画 teacher、原子写入与 dataset manifest 全部通过反例测试 |
| 2E 数据质量基线 | 🚧 已有非 identity 冒烟，完整统计待做 | 扩大数据；检查关节角速度/jerk、接触滑移、行为/clip 覆盖和确定性 |
| 2F 每角色行为克隆模型 | ⏳ 未开始 | 共享代码按 manifest 实例化 N；每个角色单独训练 checkpoint；严格 bundle 校验和闭环 rollout |
| 2G 真实角色蒙皮显示 | 🚧 通用自动 warp 已完成，高质量蒙皮待做 | 当前已证明模型骨姿态改变最终角色像素；正式角色仍需真实 2D 权重/分层或 mesh/skin/IBM/texture |
| 2H Flow Matching/风格控制 | ⏳ 后续 | 在行为克隆基线稳定后再评估，不阻塞首个 checkpoint |

当前阶段不承诺已经存在可训练或可推理的神经 checkpoint。manifest 中的 checkpoint 路径是每角色产物约定，不代表文件已经生成。自动 2D warp 已经证明非 identity 骨姿态会改变正常 sprite 的最终像素，但真正训练前仍需通过 2E 的数据质量门槛；正式角色发布前还应完成角色专属权重/分层或真实 mesh skinning，以解决复杂肢体交界和自遮挡质量。

---

## 8. 已确定边界与剩余决策

| 问题 | 决议/状态 |
|---|---|
| 是否让一个 checkpoint 跨物种或跨骨架？ | 否。每个具体角色单独 checkpoint；代码与契约通用。 |
| 模型关节数如何确定？ | `N = manifest.rig.drivenJointOrder.length`，上限受协议约束为 128。 |
| 不同角色能否使用不同原始素材尺寸？ | 可以，但导入阶段必须归一化到 48×48 runtime canvas；首版 manifest 固定 `canvas=[48,48]`、`displayScale=2`。 |
| 训练姿态是什么？ | `root_translation + root_rotation + N 个 rest-local quaternion delta`；不训练父关节相对位置数组。 |
| 3D→2D 如何投影？ | 由 manifest 的 up/forward/handedness 决定的正交侧视，深度用于绘制排序。 |
| 旧 `bone_rotations` 如何处理？ | 仅 legacy 2D capability 使用；3D 原子字段组出现时互斥。 |
| checkpoint 身份如何绑定？ | bundle 必须精确包含并匹配 `characterId + rigFingerprint + drivenJointOrder + dataset schema + normalization`。 |
| per-joint 动画 translation 怎么办？ | 当前 PlanPoint 不表达；训练 clip 可保留用于审计，但首版 teacher 只消费旋转。需要伸缩/平移骨骼的角色必须另起协议版本。 |
| 真正蒙皮何时实现？ | 当前自动 2D joint warp 已贯通正常像素主路，可用于模型闭环验证；高质量桌面角色仍应补角色专属 2D 权重/分层或真实 mesh skinning。 |
| 推理设备与训练日志？ | 尚未锁定；先以正确性、确定性和 checkpoint ABI 为门槛。 |

> **下一步：** 完成数据质量报告与首个 Cat 数据集，然后实现通用模型/训练器并只产出 Cat 自己的 checkpoint；其他每个具体角色重复数据准备与独立训练，不共享权重文件。
