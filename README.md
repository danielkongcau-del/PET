# PET：实时生成动作的 Windows 像素桌宠

PET 是一个只面向 Windows 11 的本地开发原型。角色不是播放预先录制的动作片段：Electron 宿主持续采集鼠标与可见窗口的几何状态，Python 生成器在线生成约 400 ms 的短时运动轨迹，宿主再经过碰撞、速度和屏幕边界安全层后渲染 48×48 像素猫。

当前里程碑的自动化链路已经打通，使用轻量随机自回归规划器验证交互和运行时架构；它是后续条件扩散/自回归神经网络的可运行基线，不代表最终训练模型。人工桌面交互、连续 30 分钟稳定性和端到端延迟验收仍待完成，验收状态以 [`docs/milestone-1-acceptance.md`](docs/milestone-1-acceptance.md) 中的勾选结果为准。

## 已实现

- 透明、无边框、置顶的 96×96 Electron 桌宠窗口，内部以 48×48 最近邻放大渲染；
- 把附近可见窗口顶部的水平边缘识别为可行走表面；
- 在线生成行走、跳跃、下落和点击反应，不播放动作精灵表；
- 猫的不透明像素拦截左键，透明像素把点击传给下方窗口；
- 全屏应用、开始菜单及系统/安全界面出现时隐藏；
- Python 子进程握手、心跳、超时、重启和宿主安全待机；
- Windows 物理像素协议、多显示器/DPI 边界、安全限速与窗口移动重规划；
- 托盘暂停/继续、调试层开关、重启生成器和退出；
- 无网络端口、无遥测，不采集屏幕内容和键盘输入。

## 快速启动

前置条件：Windows 11、Node.js 20+、pnpm 11，以及已配置的 `pet-core` Conda 环境。当前固定 Python 为 3.10.20，PyTorch 为 2.10.0+cu130；详见 [`environment/README.md`](environment/README.md)。

如果普通 PowerShell 尚不能识别 `pnpm`，可用 Node 自带的 Corepack 将项目锁定版本安装到已在用户 `PATH` 中的 npm 目录：

```powershell
corepack install -g pnpm@11.9.0
corepack enable pnpm --install-directory "$env:APPDATA\npm"
pnpm --version
```

```powershell
conda activate pet-core
python environment/smoke_test.py
pnpm install
pnpm dev
```

`pnpm dev` 会先构建共享协议，再启动 Electron；Electron 会自动运行：

```powershell
D:\Anaconda\envs\pet-core\python.exe services/generator/run.py
```

开发时可设置：

```powershell
$env:PET_DEBUG_OVERLAY = "1"       # 启动时显示表面、轨迹与安全状态
$env:PET_DISABLE_GENERATOR = "1"   # 只运行宿主的安全待机路径
$env:PET_PYTHON = "...\python.exe" # 覆盖 Python 路径
pnpm dev
```

退出请使用系统托盘中的“退出”；关闭可见窗口不会结束托盘宿主。

## 验证

```powershell
pnpm check
pnpm build
```

`pnpm check` 包含 TypeScript 类型检查、桌面安全层测试、Python 生成器测试、跨语言协议 Schema 测试、真实 stdio 子进程集成测试和资产确定性测试。Python/CUDA/研究仓库依赖另由下面的命令验证：

```powershell
conda activate pet-core
python environment/smoke_test.py
```

人工验收项目在 [`docs/milestone-1-acceptance.md`](docs/milestone-1-acceptance.md)，本次自动化与真实进程故障注入证据见 [`docs/verification-2026-07-20.md`](docs/verification-2026-07-20.md)。

## 运行链路

```mermaid
flowchart LR
    A["Windows 窗口与鼠标几何"] --> B["Electron 状态采集"]
    B -->|"world_state / NDJSON"| C["Python 在线动作生成器"]
    C -->|"约 400 ms horizon_plan"| D["宿主运动安全层"]
    D --> E["像素姿态渲染"]
    E --> F["透明置顶桌宠窗口"]
    D -->|"失效、越界或超时"| G["安全待机/下落/重规划"]
```

宿主始终是位置的唯一写入者。模型只提出脚底锚点的相对轨迹和 `lean/squash/bob/expression` 姿态参数，不能直接移动真实窗口、绕过全屏隐藏策略或读取窗口内容。

## 目录

| 路径 | 作用 |
|---|---|
| `desktop/` | 基于 OpenPets 设计缩减的 Electron/Windows 宿主 |
| `services/generator/` | Python 在线随机自回归轨迹生成器及 PyTorch 后端接口 |
| `packages/protocol/` | JSON Schema、TypeScript/Python 类型和固定协议样本 |
| `assets/pet/` | 用户参考图、生成概念图、48×48 运行资产和部件元数据 |
| `environment/` | 单一 `pet-core` 环境约束、锁定快照和离线冒烟测试 |
| `tests/` | 跨语言协议及真实生成器 stdio 集成测试 |
| `third_party/` | 保持不改的宿主、动作生成和像素生成参考仓库 |

桌面宿主主要复用 OpenPets 的透明窗口、命中测试、窗口跟踪、单写入运动循环和托盘生命周期设计；具体来源边界见 [`desktop/UPSTREAM.md`](desktop/UPSTREAM.md)。`third_party` 中的研究项目是后续模型实验的参考快照，不应把它们各自冲突的依赖整体安装进统一环境。

## 当前生成器与下一阶段

当前 `AutoregressiveMotionBackend` 会根据最新表面、脚底位置、速度和点击边沿事件，以可复现随机种子连续生成短轨迹。它已经具备真实模型所需的服务接口、滚动重规划、点击优先级和故障降级，因此替换模型时不需要重写桌面宿主。

下一阶段建议先建立程序化窗口场景模拟器和轨迹数据集，再训练“小型条件动作模型”，而不是直接逐像素生成 RGBA 帧：

1. 条件输入：最近状态历史、候选表面、目标、点击事件和当前姿态；
2. 输出：未来 12～24 个脚底位移/速度/姿态点；
3. 第一模型：小型 causal Transformer 或 MLP-Mixer，建立延迟和稳定性基线；
4. 第二模型：1D conditional diffusion/flow matching，生成更多样的运动；
5. 蒸馏或少步采样，把桌面端单次规划延迟稳定在 100 ms 内；
6. 继续复用宿主的确定性碰撞和安全层，模型永远不拥有最终执行权。

更完整的边界与训练路线见 [`docs/architecture.md`](docs/architecture.md)。

## 角色资产

运行时使用 [`assets/pet/runtime/cat-48.png`](assets/pet/runtime/cat-48.png)，按 2 倍最近邻显示。资产由用户截图作为参考，通过内置图像生成得到色键概念图、移除背景后再确定性缩放；原始提示词与生成方式保存在 [`assets/pet/source/IMAGEGEN.md`](assets/pet/source/IMAGEGEN.md)。

首版刻意不包括安装包、开机启动、声音、拖拽、跨显示器跳跃、屏幕内容理解和完整设置页。
