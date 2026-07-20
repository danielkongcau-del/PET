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
2. Review ──→  至少 2 个并行子 Agent 独立审查（Agent Type: Explore）
3. Fix    ──→  修复审查发现的所有 bug（P0/P1 必须修，P2 可记录为已知债务）
4. Verify ──→  typecheck + test suite，必须全部通过
5. Report ──→  向用户汇报：变更了什么、审查发现了什么、修了什么、验证结果
```

### 子 Agent 审查规则

- **数量**：每批变更启动 **至少 2 个** 独立审查 Agent，从不同角度检查
- **类型**：使用 `Explore` agent type
- **指令**：每个审查 Agent 必须被告知：
  - 审查哪些文件
  - 特别关注哪些方面（安全、兼容性、边界条件、类型正确性、与其他模块的交互）
  - 不允许笼统审查——必须指出具体行号和问题描述
- **并行**：所有审查 Agent 同时启动，不串行等待

### Bug 优先级

| 级别 | 含义 | 处理方式 |
|---|---|---|
| P0 / BUG / Defect | 编译错误、运行时崩溃、逻辑错误 | 必须立即修复 |
| P1 | 设计缺陷、潜在安全/性能问题 | 必须修复，或记录为已知债务并说明缓解措施 |
| P2 | 风格不一致、文档缺失、可优化但不紧急 | 可记录，不阻塞当前任务 |

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

新增字段必须：
1. 标记为可选（`?` / `Optional` / `total=False`）
2. 在 Schema 的 `properties` 中显式定义（不依赖 `additionalProperties`）
3. 旧消费者忽略它时系统正常运行（降级路径已验证）

---

## 测试门禁

以下命令必须在每次代码变更后全部通过：

```powershell
pnpm typecheck          # TypeScript 全 workspace 类型检查
pnpm test:desktop       # 宿主安全层 + 协议 + 表面 + 渲染器测试
```

可选（Python 侧变更时执行）：
```powershell
pnpm test:generator     # 生成器单元测试
pnpm test:protocol      # 跨语言协议测试
pnpm test:integration  # 真实 Python 子进程集成测试
```

---

## 文件变更记录

设计文档在 `docs/` 中。重大设计决策必须在对应 `docs/*.md` 中记录。

当前活跃设计文档：
- `docs/skeletal-animation-design.md` — 骨骼动画 & 端到端动作模型
- `docs/architecture.md` — 系统架构
- `docs/milestone-1-acceptance.md` — 里程碑验收清单
