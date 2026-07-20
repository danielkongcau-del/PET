# 骨骼动画 & 端到端动作模型：设计文档

状态：已评审 v1，已更新至 v2。预备代码阶段，不涉及核心逻辑变更。

**评审决议 (2026-07-20)：**
- 分层精灵：优先搜索现有免费 CC0 资产，搜索期间先写不依赖精灵的预备代码（协议层、Schema、能力协商）
- 能力协商：通过 `hello/ready` 的 `capabilities` 字段宣告 `"skeletal_motion"`
- 骨骼版本：生成器在 `ready` 中发送完整骨骼定义的 SHA-256，宿主对比本地 `cat-skeleton.json`，不匹配时拒绝 + 错误提示
- 尾巴物理：模型输出 tail_base 角度，tail_tip 由渲染器施加惯性跟随 + 阻尼二级物理
- 步态协调：模型从数据中自主学习（数据驱动），渲染器不做约束
- 表情系统：用连续面部参数（eye_scale, mouth_open, ear_angle 等），不再仅依赖离散 expression 字段
- **骨骼为唯一真相源 (2026-07-20)**：任何代码不得硬编码骨骼数量。`cat-skeleton.json` 是骨骼层级、父子关系、可驱动骨骼列表的唯一权威来源。所有模块——宿主协商、协议校验、渲染器 FK、生成器后端——必须在运行时从此文件动态读取骨骼结构。

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

### 1.2 骨骼定义文件：`assets/pet/runtime/cat-skeleton.json`

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

### 1.4 骨骼为唯一真相源

`cat-skeleton.json` 是骨骼层级、父子关系、可驱动骨骼列表的**唯一权威来源**。任何模块不得硬编码骨骼数量、骨骼名称或骨骼索引。

**数据来源规则：**

```
cat-skeleton.json
  ├── bones[*].physics.mode === "model_driven"  → 可驱动骨骼列表
  ├── bones[*].physics.mode === "secondary"      → 渲染器物理驱动骨骼
  ├── bones[*].physics.mode === "static"         → 纯锚点骨骼（不移动）
  ├── bones[*].parent                             → FK 父子拓扑
  ├── drawOrder                                   → 渲染层级
  └── bones[*].limits                             → 运行时角度裁剪
```

**各模块读取义务：**

| 模块 | 从 skeleton 读取 | 当前状态 |
|---|---|---|
| `generator-bridge.ts` | 加载 skeleton JSON，计算 SHA-256，提取 model_driven 骨骼数量和名称列表 | ❌ 仅算 hash |
| `protocol.ts` | `isBoneRotations` 接收期望长度参数，校验 `bone_rotations.length === modelDrivenCount` | ❌ 允许 1-32 |
| `motion-controller.ts` | 校验 PlanPoint.bone_rotations 长度与 skeleton 一致 | ❌ 不做长度校验 |
| `pet-window.ts` | 加载 skeleton 并传入 renderer | ❌ 不传 |
| `renderer.js` | 根据 bones + drawOrder 递归 FK，无需硬编码 | ❌ 待实现 |
| `backend.py` (生成器) | 读取同一个 `cat-skeleton.json`，model_driven 骨骼数 = 姿态输出维度 | ❌ 不自省 |

**关键不变量：**

1. `modelDrivenBoneCount` = `count(bones where physics.mode === "model_driven")` — 所有模块各自从 skeleton 计算
2. `bone_rotations[i]` 对应第 i 个 model_driven 骨骼，按 `bones` 数组出现顺序
3. 宿主加载 skeleton 后立即计算 `modelDrivenBoneIds: string[]`，用于校验和日志

**示例：当前 10 骨 skeleton 中 model_driven 骨骼共 7 个：**

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
  eye_scale: number;      // 眼睛大小 [0.5, 1.5]  → surprised=大, sleepy=小
  eye_squint: number;     // 眯眼程度 [0, 1]      → annoyed/ curious
  mouth_open: number;     // 张嘴程度 [0, 1]      → surprised/ happy
  ear_angle: number;      // 耳朵旋转角 [-0.5, 0.5] → scared=flat, curious=perked
  brow_tilt: number;      // 眉毛倾斜 [-1, 1]     → annoyed=down, sad=up
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

这样向后兼容：旧生成器只发 `expression` 字符串，渲染器查表转成默认参数；新生成器直接发 `facial_params`，渲染器原样使用。

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

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
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

    def step(self, action: PoseDelta) -> WorldState:
        """执行一步动作，返回下一帧 WorldState。
        物理模拟包括：
        - 重力 (980 px/s²)
        - 碰撞检测 (与 geometry.ts 的 findCrossedSurface 等价)
        - 表面支撑
        - 工作区边界限制
        """
        dt = 33  # ms
        # 应用动作到 pet 状态
        self.pet.apply(action, dt)
        # 重力 + 碰撞
        self.pet = self._physics_step(self.pet, dt)
        # 窗口可能移动
        self._maybe_move_windows()
        # 重新计算表面
        self.surfaces = self._compute_surfaces()
        self.time_ms += dt
        return self._to_world_state()

    def _physics_step(self, pet: SimPet, dt_ms: int) -> SimPet:
        """确定性物理：镜像 motion-controller.ts 的 applyFallback 逻辑。"""
        ...

    def _compute_surfaces(self) -> list[SimSurface]:
        """镜像 surface-tracker.ts 的 buildSurfaceSnapshot 逻辑。"""
        ...
```

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
        action = sample_plan(plan, sim.time_ms)
        state = sim.step(action)
        condition_frames.append(state)

    # 生成目标
    for _ in range(H):
        plan = backend.generate(sim.to_world_state(), seed)
        action = sample_plan(plan, sim.time_ms)
        state = sim.step(action)
        # 将 action 映射为骨骼姿态
        target_poses.append(action_to_bone_pose(action))

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

## 7. 实施路线与里程碑

### Milestone 2A：骨骼渲染管线（目标：Day 1-3）

| 任务 | 产出 | 验证方式 |
|---|---|---|
| 2A.1 骨骼 JSON Schema | `cat-skeleton.schema.json` | JSON Schema 校验通过 |
| 2A.2 骨骼默认定义 | `cat-skeleton.json`（10 骨） | 加载到宿主不报错 |
| 2A.3 程序化几何猫 | `renderer.js` 修改：`drawProceduralCat(ctx, skeleton, poses)` | 不依赖资产，看到可辨识的猫 |
| 2A.4 FK 渲染器 | `renderer.js` 完整 FK 管线 | 手动设置一组关节角，验证视觉正确 |
| 2A.5 `bone_rotations` 协议兼容 | `protocol.ts` 放行新字段 | TS 类型检查 + 协议测试通过 |
| 2A.6 回退兼容 | 空 `bone_rotations` 时使用旧的整体变形 | 旧 plan 仍可渲染 |

**交付物：** 一只用程序化几何画的猫，能通过 FK 骨骼摆出不同姿势。分层精灵到后只需替换 `drawProceduralCat` → `drawSpriteCat`。

### Milestone 2B：桌面模拟器 + 训练数据（目标：Day 4-10）

| 任务 | 产出 | 验证方式 |
|---|---|---|
| 2B.1 模拟器核心 | `simulator.py`：场景生成 + 物理 + 碰撞 | 1000 episodes 中 0 crash，物理无穿模 |
| 2B.2 场景随机化 | `ScenarioConfig` 完整参数空间 | 覆盖率：1-3 显示器、3-12 窗口、多 DPI |
| 2B.3 Teacher 轨迹生成 | `generate_data.py` 用当前 planner 生成数据 | 10K samples 通过协议校验 |
| 2B.4 姿态映射函数 | `action_to_bone_pose()` | 生成 100 个随机 pose 人工肉眼检查 |
| 2B.5 数据 loader | PyTorch Dataset + DataLoader | 加载速度 > 10K samples/s |
| 2B.6 数据质量报告 | 轨迹分布统计 | velocity jerk、position jump、fallback 率与真实宿主一致 |

**交付物：** 10K-50K 训练样本（`.pt` 或 `.h5` 格式），`TrainingSample` 结构确定。

### Milestone 2C：Causal Transformer 训练（目标：Day 11-21）

| 任务 | 产出 | 验证方式 |
|---|---|---|
| 2C.1 模型定义 | `neural_motion/model.py` | 前向通过 + 参数统计（~2M） |
| 2C.2 环境编码器 | `neural_motion/encoder.py` | 编码器输出维度正确 |
| 2C.3 训练脚本 | `neural_motion/train.py` | 单 GPU 训练不 OOM |
| 2C.4 验证指标 | 位置 L2、角度 MAE、jerk、落地成功率 | 对比 teacher 基线 |
| 2C.5 闭环模拟测试 | 在模拟器中用模型 rollout 30s | 不掉落、不穿模、80% 落地率 |
| 2C.6 真实宿主集成 | `NeuralMotionBackend` 替换 planner | `pnpm dev` 启动，猫正常运动 |
| 2C.7 A/B 对比 | 模型 vs teacher 人工对比 | 运动自然度评分 |

**交付物：** 一个 `.pt` checkpoint，加载后可通过 `pnpm dev` 在真实桌面上运行。

### Milestone 2D：Flow Matching（目标：后续，不阻塞 2A-2C）

| 任务 | 产出 |
|---|---|
| 2D.1 Flow Matching 模型 | `neural_motion/flow_model.py` |
| 2D.2 训练 + 蒸馏 | 4 步 → 2 步采样 |
| 2D.3 风格控制 | CFG / style embedding |
| 2D.4 闭环评估 | 与 causal transformer 对比 |

---

## 8. 待确认项

| 问题 | 优先级 | 状态 |
|---|---|---|
| 分层精灵何时到位？ | P0 | 🔴 搜索中（itch.io / OpenGameArt / craftpix），48h 评估 |
| 骨骼层级 joint 位置是否合理？ | P0 | 🟡 待精灵到位后校准 |
| ~~能力协商方案~~ | ~~P0~~ | 🟢 `capabilities` + 骨骼 SHA-256 |
| ~~尾巴物理方案~~ | ~~P0~~ | 🟢 渲染器二级物理 |
| ~~骨骼版本协商~~ | ~~P0~~ | 🟢 完整哈希校验，不匹配降级+WARN |
| ~~步态协调~~ | ~~P0~~ | 🟢 数据驱动，模型自主学习 |
| ~~表情系统~~ | ~~P0~~ | 🟢 连续面部参数 + expression 查表 fallback |
| 训练数据量目标？ | P1 | 50K episodes |
| 是否需要 behavior 分类头？ | P1 | 是 |
| 姿态映射函数 `action_to_bone_pose()` 是否需要校准？ | P1 | 精灵到位后校准 |
| 模型推理 GPU/CPU？ | P2 | GPU 优先，支持 CPU fallback |
| 训练日志工具？ | P2 | TensorBoard（默认） |

---

> **下一步：** 请评审本设计文档，尤其关注 §1（骨骼层级）、§2.3（程序化几何外观）、§5.1（模型架构）。确认后我开始实施 Milestone 2A。
