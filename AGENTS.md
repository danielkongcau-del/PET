# PET 项目 Agent 行为规范

## 核心理念

本项目对代码正确性要求极高：多进程通信、物理安全层、跨语言协议、实时渲染。任何 bug 在桌宠运行时都会直接可见。因此采用 **"问 → 写 → 审 → 修 → 再审"** 的强制性工作流，而非一次写完靠运气。

---

## 强制工作流

### 每次代码任务启动前（必选，不可跳过）

**在写任何代码之前，必须持续向用户提问，直至确认所有技术细节。** 此步骤不可省略。

提问规则：
1. 识别任务中所有存在多种合理实现方式的技术决策点
2. 在获得用户的明确选择前，不得落代码
3. 当用户说「开始写代码」但仍有未确认细节时，**必须指出剩余模糊点并继续提问**
4. 可附带建议，但最终选择权在用户

典型提问领域：
- 数据结构设计（字段名、类型、默认值、可选/必填）
- API / 接口签名（函数参数、返回值、IPC 通道名）
- 向后兼容策略（新字段是可选还是必填、旧路径是否保留）
- 错误处理策略（静默降级 vs 崩溃 vs 日志警告）
- 性能约束（目标 FPS、内存预算、延迟上限）
- 代码组织方式（新文件 vs 追加到现有文件）

### 每次代码变更必须执行以下步骤：

```
0. Ask   ──→  确认所有技术细节（不可跳过）
1. Write  ──→  产出变更
2. Review ──→  至少 2 个并行、只读的子 Agent 独立审查
3. Fix    ──→  P0 必须修复；P1 必须修复，或由用户明确批准并记录缓解措施后转为债务；P2 可记录为已知债务
4. Re-review ──→ 至少 2 个 Agent 独立复审修复增量；P0 必须关闭，P1 必须关闭或满足上述债务例外，否则返回 Fix
5. Verify ──→  按变更路径执行强制测试矩阵，必须全部通过
6. Report ──→  向用户汇报：变更了什么、审查发现了什么、修了什么、验证结果
```

### 子 Agent 审查规则

- **数量**：每批变更启动 **至少 2 个** 独立审查 Agent，从不同角度检查
- **模式**：给普通子 Agent 下达只读探索/审查任务，并明确禁止修改文件；不得依赖当前工具不存在的 agent type
- **指令**：每个审查 Agent 必须被告知：
  - 审查哪些文件
  - 特别关注哪些方面（安全、兼容性、边界条件、类型正确性、与其他模块的交互）
  - 不允许笼统审查——必须指出具体行号和问题描述
- **并行**：所有审查 Agent 同时启动，不串行等待

### Bug 优先级

| 级别 | 含义 | 处理方式 |
|---|---|---|
| P0 / BUG / Defect | 编译错误、运行时崩溃、逻辑错误 | 必须立即修复 |
| P1 | 设计缺陷、潜在安全/性能问题 | 必须修复；只有用户明确批准并记录缓解措施后，才能转为已知债务 |
| P2 | 风格不一致、文档缺失、可优化但不紧急 | 可记录，不阻塞当前任务 |

---

## 系统化代码审查方法论

代码审查不得只检查单个函数是否“看起来合理”，必须沿着数据和状态的完整生命周期验证。默认采用以下 **契约 → 生产 → 传输 → 消费 → 生命周期 → 验证** 方法。

### 1. 先确定真实变更范围

1. 先记录当前 `HEAD` 和本次审查采用的基线 SHA；基线不明确且会改变审查结论时，必须向用户确认。
2. 工作区变更必须同时检查：`git status --short`、`git diff --stat`、`git diff`、`git diff --cached --stat`、`git diff --cached`，以及 `git ls-files --others --exclude-standard` 列出的未跟踪文件。
3. 已提交分支必须检查 `git diff --stat <base>...HEAD` 和 `git diff <base>...HEAD`，不能只看未提交工作区。
4. 搜索所有新增字段、类型、capability、Schema 名称和配置文件的引用位置。
5. 如果 Git 元数据不可用，必须明确记录审查范围受限；可用文件修改时间和符号搜索辅助定位，但不得声称已经完成精确 diff 审查。
6. 区分源代码、生成产物、缓存和第三方代码，避免把构建输出当成权威实现。

### 2. 建立跨层契约矩阵

每个新增或修改字段都必须逐项核对以下位置：

| 层 | 必查内容 |
|---|---|
| 权威 Schema | required/optional、范围、互斥、原子字段组、`additionalProperties` |
| TypeScript 类型 | 字段类型、可选性、枚举和数组顺序 |
| Python 类型 | 与 TypeScript 相同的必填/可选语义 |
| 双端运行时校验 | 有限值、长度、范围、结构和错误策略 |
| 序列化/反序列化 | 字段是否遗漏、重命名、变形或重复编码 |
| capability/版本 | 发送方是否真的实现所宣告的能力 |

必须验证三类一致性：

- **结构一致性**：字段名、类型、必填性、数组长度和互斥关系一致。
- **语义一致性**：坐标空间、单位、轴方向、相对参考系、默认值一致。
- **行为一致性**：同一合法/非法消息在 Schema、TypeScript 和 Python 中得到相同结论。

不得仅凭类型声明推定运行时安全；必须检查实际边界校验器。

原子字段组不能被“所有新增字段都可选”的规则拆散：扩展分支或容器整体可以向后兼容地缺席；一旦判别字段或组内任意成员出现，组内字段可被条件性要求完整。TypeScript 应优先使用可辨识联合，Schema 应使用 `oneOf` 或 `if/then` 表达同一约束，Python 运行时校验必须给出相同结论。

### 3. 沿端到端数据流逐字段追踪

对每个字段画出或写出实际 source/sink 路径。下列链路是端到端功能的检查模板，不要求每个字段机械经过所有站点；不适用或有意提前终止的站点必须标为 `N/A` 并说明理由：

```text
生产者
→ 内部数据类
→ 序列化
→ 协议解码与校验
→ 安全层/插值器
→ 控制器
→ IPC/preload
→ renderer
→ 轨迹记录与确定性回放
```

逐站确认：

1. 字段没有在对象重建、插值、浅拷贝或降级过程中丢失。
2. 同一语义没有被两套字段重复应用，例如世界位移和局部 root 位移。
3. 消费者确实使用了已宣告支持的字段；仅透传但不消费不算能力完成。
4. 字段消失时旧状态会被清除，不会因浅合并冻结在上一帧。
5. 插值方法符合数据类型：标量可 lerp，角度需最短弧，四元数需归一化 SLERP，离散状态不得数值插值。

### 4. 检查数学与坐标不变量

涉及几何、骨骼、物理或模型输出时，必须显式核对：

- 坐标空间、手性、up/forward/depth 轴和屏幕 Y 方向。
- 单位及换算关系，例如源素材像素、DIP、物理像素和模型单位。
- root、body root、support/contact anchor 的职责是否分离。
- rest transform、局部 delta、FK 组合顺序和旋转轴是否正确。
- 四元数的分量顺序（`[x,y,z,w]` 或 `[w,x,y,z]`）、局部/父空间、`rest·delta` 或 `delta·rest` 的组合约定是否在双端一致。
- 四元数是否有限、接近单位范数，零四元数是否被拒绝；还必须定义双端共享的单位范数容差、拒绝或归一化策略、`q/-q` 半球规则及近共线插值退化策略。
- 骨长、关节限制、DOF 和模型输出维度是否一致。
- 哈希是否使用跨语言一致的规范化字节；必须覆盖非 ASCII、浮点格式和属性顺序。

静态 Rig 不能只做 JSON 结构校验，还必须做图语义校验：joint ID 唯一、motion root 唯一且有效、parent 存在、无环、所有 joint 可达、draw order 引用有效、可驱动关节顺序确定、协议能表达所有声明的 DOF。

数值容差、四元数策略和哈希规范化算法不能只写成“足够接近”或“规范化”；必须以带版本的常量或协议规范固定下来，并由 TypeScript/Python 共用的固定输入输出向量验证。修改这些常量或算法视为协议变更。

### 5. 把兼容性当作状态机审查

不得只测试一次理想握手。至少检查以下序列：

```text
旧 host ↔ 新 generator
新 host ↔ 旧 generator
3D → 2D 降级
3D → 无骨骼降级
重新 hello / 新 session
generator 重启
renderer reload
骨架 hash 缺失或不匹配
计划从新编码切回旧编码
旧 child 的异步加载/协商在新 child ready 后才完成
连续两个 ready 或加载结果乱序完成
重启期间旧 promise 尝试 flush
```

把所支持的编码视为状态集合（当前至少为 `{none, 2D, 3D}`）：测试必须覆盖每个允许的有向转换、同状态重复握手，以及至少一次往返转换，不能只覆盖从新能力向旧能力降级。

确认 capability、缓存、计数器和 renderer 状态在每次转换后都被正确设置或清除。异步握手、Rig 加载和 flush 必须使用 generation token、取消机制或等价方案，阻止旧会话完成结果覆盖新会话。双方必须在发送首个计划前对 selected encoding、Rig hash 和可驱动关节顺序达成同一结果。若两种编码互斥，Schema 和运行时必须共同强制；不得依靠“发送方应该不会这么做”。

### 6. 用最小反例主动证伪

现有测试通过不等于新路径正确。审查者必须为关键假设构造最小、只读复现，例如：

- 包含中文名称的对象在两种语言中计算相同哈希。
- `[0, 0, 0, 0]`、非单位四元数、极大有限数是否被拒绝。
- 只提供原子字段组的一部分是否被拒绝。
- legacy 与新编码同时出现时双端是否一致拒绝。
- 两个关键帧经过插值后，所有姿态字段仍存在且数值正确。
- 全部 local delta 为 identity 时，必须精确还原 authored rest pose。
- 单关节已知轴旋转时，TypeScript/Python FK 得到相同姿势；legacy `restAngle`/限位转换到 3D 后保持相同世界姿势。
- capability 降级或重新握手后，不再输出上一会话的编码。
- renderer reload 后，配置和姿态是否恢复且不存在旧状态残留。

每个反例先提供一个被 Schema、TypeScript、Python 全部接受的合法对照，再只改变一个条件触发失败；用结果表记录三端的真实返回值或错误。复现应优先调用真实 serializer、validator、interpolator 和 consumer，而不是复制一份相同逻辑到测试脚本中。

### 7. 反查测试覆盖，而非只运行测试

完成常规 typecheck、单元测试和集成测试后，还必须：

1. 搜索新增字段和 capability 是否出现在测试中。
2. 确认测试断言最终可见行为，而不是只断言消息被发送或函数未抛错。
3. 检查跨语言 fixture 是否覆盖新增合法消息和关键非法消息。
4. 检查插值中点、精确关键帧、边界长度、重启和降级等状态转换。
5. 核对新增测试位于 `package.json` 标准脚本或 CI 的测试发现范围内，并从标准命令输出确认测试用例实际被收集和执行。
6. 如果测试全绿但新增符号没有测试引用，或新增测试没有被标准命令收集，报告中必须明确说明该绿灯不覆盖新功能。

### 8. 并行审查的默认分工

涉及跨进程或骨骼功能时，至少两个审查 Agent 应从不同角度独立工作；推荐拆成三个方向：

1. **协议与兼容性**：Schema、TypeScript/Python 镜像、validator、capability、旧版本。
2. **数学与数据模型**：坐标、单位、Rig、rest pose、旋转、FK、物理不变量。
3. **运行时消费链**：插值、安全层、IPC、renderer、reload、回放和测试覆盖。

主 Agent 必须自己核实关键发现，不能仅转述子 Agent 结论。

### 9. 审查报告格式

每个问题必须包含：

- 严重级别（P0/P1/P2）。
- 可点击的文件路径和精确起始行号。
- 触发条件。
- 实际后果，而非笼统描述“可能有问题”。
- 必要时给出最小复现结果或跨语言对照证据。

报告中必须区分：确定性缺陷、尚未贯通的脚手架、测试空白和长期设计债务。按严重程度排序，并先报告会阻断主路径的问题。

---

## 代码质量标准

### 不得硬编码

任何可能在别处变化的数字、名称、结构，必须从单一真相源获取。

本项目中已确定的真相源：
- `cat-skeleton.json` → 骨骼层级、数量、可驱动骨骼列表
- `packages/protocol/schemas/v1/pet-motion.schema.json` → 协议结构
- `@pet/protocol` 包 → TypeScript/Python 类型定义

### 跨语言一致性

当 TypeScript 和 Python 两端都有同一概念时，两边的逻辑必须同步变更：
- `packages/protocol/src/v1.ts` ↔ `packages/protocol/python/pet_protocol/v1.py`
- `desktop/src/protocol.ts`（运行时校验）↔ `services/generator/pet_generator/protocol.py`
- `packages/protocol/schemas/v1/pet-motion.schema.json` 是权威 Schema，双方都应引用它

### 破坏性变更禁区

以下行为不允许：
- 改变现有 `PlanPoint`、`WorldStatePayload` 等接口的必填字段
- 修改 `additionalProperties: false` 的 Schema 而不增加新字段
- 改变 `hello/ready` 握手顺序或必需的 capability
- 使现有测试（`pnpm test`）失败

### 向后兼容规则

新增向后兼容扩展必须：

1. 扩展分支或容器整体可选；一旦判别字段或原子字段组开始出现，其内部字段可被条件性要求完整，三端必须表达相同约束。
2. 在 Schema 的 `properties` 中显式定义，不依赖 `additionalProperties` 放行未知字段。
3. 不得假设带 `additionalProperties: false` 的旧消费者会忽略新字段。只有在 capability 协商确认对端支持后才能发送扩展字段；无法协商时必须使用新协议版本或新消息类型。
4. 旧对端未协商新能力时仍能走已验证的降级路径，且不会保留上一会话的新编码状态。

---

## 测试门禁

以下命令必须在每次代码变更后全部通过：

```powershell
pnpm typecheck          # TypeScript 全 workspace 类型检查
pnpm test:desktop       # 宿主安全层 + 协议 + 表面 + 渲染器测试
```

此外按变更路径执行以下强制矩阵；同时命中多行时取并集，跨层改动优先运行 `pnpm check`：

| 变更路径/内容 | 追加必跑命令 |
|---|---|
| `packages/protocol/**`、Schema、跨语言 fixture、协议 validator | `pnpm test:protocol`、`pnpm test:generator`、`pnpm test:integration` |
| `services/generator/**` | `pnpm test:generator`；涉及 stdio、握手、协议或进程生命周期时再跑 `pnpm test:integration` |
| `desktop/src/**` | `pnpm test:desktop`；涉及 IPC、bridge、握手或 generator 生命周期时再跑 `pnpm test:integration` |
| `assets/**`、Rig、骨骼转换或资源校验 | `pnpm test:assets`；涉及协议姿态时再跑 `pnpm test:protocol` 和 `pnpm test:integration` |
| 多层端到端功能或无法可靠判定影响范围 | `pnpm check` |

这些命令不是“可选测试”；命中对应条件时必须执行：

```powershell
pnpm test:generator     # 生成器单元测试
pnpm test:protocol      # 跨语言协议测试
pnpm test:integration  # 真实 Python 子进程集成测试
pnpm test:assets        # 资源、Rig 与静态语义校验
```

---

## 文件变更记录

设计文档在 `docs/` 中。重大设计决策必须在对应 `docs/*.md` 中记录。

当前活跃设计文档：
- `docs/skeletal-animation-design.md` — 骨骼动画 & 端到端动作模型
- `docs/architecture.md` — 系统架构
- `docs/milestone-1-acceptance.md` — 里程碑验收清单
