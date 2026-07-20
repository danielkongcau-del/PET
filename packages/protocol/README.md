# PET Motion Protocol v1

这个包是 Electron 桌面宿主与 Python 动作生成器之间唯一的进程边界契约。传输采用宿主创建的私有 `stdin/stdout` 管道；每一行是一个完整 UTF-8 JSON 对象，行尾为 `\n`。标准输出只允许协议消息，诊断文字必须写入标准错误。

## 权威文件

- JSON Schema：`schemas/v1/pet-motion.schema.json`
- TypeScript 类型：`src/v1.ts`
- Python 类型与严格编解码：`python/pet_protocol/`
- 两种语言共用的固定会话：`fixtures/v1/session.ndjson`

Schema 是线格式的最终权威。TypeScript 的 `decodeNdjsonLine` 只检查 envelope；宿主边界需要用 Schema 校验不可信 payload。Python 的 `decode_ndjson_line` 不依赖第三方库并执行完整结构及关键语义校验。

## Envelope

```json
{
  "protocol": "pet-motion",
  "version": 1,
  "type": "world_state",
  "seq": 12,
  "timestamp_ms": 1784491200030,
  "payload": {}
}
```

- `seq` 由每个发送方独立维护，进程存活期间严格递增；宿主序列和生成器序列不互相比较。
- 所有 `*_ms` 绝对时间均为 Unix epoch 毫秒；`PlanPoint.t_ms` 是相对该 horizon 起点的时间。
- 所有跨 JavaScript/Python 的整数必须在 `0..Number.MAX_SAFE_INTEGER`；随机种子进一步限制为 uint32。
- v1 进程边界统一使用 `physical_px`。Electron DIP 与物理像素的转换只能发生在宿主的传感/执行边界。
- 未知 `protocol`、`version`、消息类型或额外字段均应拒绝，并以可恢复 `error` 回复；解析失败时不得把原始输入写入日志。

## 时序与方向

| 消息 | 方向 | 规则 |
|---|---|---|
| `hello` | 宿主 → 生成器 | 子进程启动后的第一条消息，声明 session、能力和运行参数 |
| `ready` | 生成器 → 宿主 | 接受同一 session 后回复；宿主收到前不发送状态流 |
| `world_state` | 宿主 → 生成器 | 最新状态队列容量为 1，旧状态允许被覆盖 |
| `horizon_plan` | 生成器 → 宿主 | `based_on_seq` 指向其依据的 `world_state.seq` |
| `cancel` | 宿主 → 生成器 | 可针对一个 plan；省略 `plan_id` 表示取消全部 |
| `ping` / `pong` | 任一方向 | `nonce` 原样返回；首版主要由宿主探活 |
| `metrics` | 任一方向 | 仅数值指标和有限标签，不携带窗口标题、按键或截图 |
| `error` | 任一方向 | 协议或运行错误；`recoverable=false` 后应重启会话 |

正常握手为 `hello → ready`，其后才进入 `world_state → horizon_plan` 循环。生成器可以在响应之间发 `metrics`，接收方不能假设严格的一问一答。

## 轨迹语义

`horizon_plan.points[].dx/dy` 是相对于 `based_on_seq` 世界状态中 `pet.foot_x/foot_y` 的位移，不是绝对窗口坐标。`vx/vy` 以物理像素每秒表示。姿态字段 `lean`、`squash`、`bob` 和 `expression` 供渲染器使用。

桌面宿主始终拥有最终位置与碰撞权威。下列 plan 必须被拒绝而不能执行：

- `based_on_seq` 来自未来，或已落后于宿主允许的状态窗口；
- 当前 Unix 时间不早于 `valid_until_ms`；
- 点时间不从 0 开始、不严格递增，或间隔不等于 `dt_ms`；
- 数值不是有限数、越界，或目标 surface 已失效；
- plan 已被 `cancel`，或者优先级更高的拖拽、拓扑变化、安全动作已发生。

安全优先级固定为：用户拖拽 > 窗口拓扑变化 > 安全层 > 生成 plan > 宿主 fallback。

## 版本策略

v1 对额外字段采取严格拒绝，避免两个进程悄悄产生不同解释。新增可选业务字段也需要升级协议版本；修正文档或收紧不改变合法消息集合的语义约束可以保持 v1。握手无法接受版本时返回 `error.code = "UNSUPPORTED_VERSION"` 且 `recoverable = false`。

## 本地检查

从仓库根目录运行：

```powershell
$null = pnpm --filter @pet/protocol build
$env:PYTHONPATH = (Resolve-Path packages/protocol/python).Path
python -m unittest discover -s tests/protocol -v
```

Python 测试包可通过 `pip install -e packages/protocol/python[test]` 安装可选的 Draft 2020-12 Schema 元验证器；即使未安装 `jsonschema`，内置严格 codec 与 Schema 结构门禁仍会执行。
