# 2026-07-20 开发切片验证记录

本记录区分自动化/进程级验证与尚未完成的人工桌面验收。人工功能状态仍以 `docs/milestone-1-acceptance.md` 的勾选项为准。

## 自动化结果

在 `E:\CodeSpace\PET`、Windows 11、`D:\Anaconda\envs\pet-core\python.exe` 下执行：

```powershell
pnpm check
pnpm build
& 'D:\Anaconda\envs\pet-core\python.exe' environment\smoke_test.py
```

结果：

- TypeScript 协议包与 Electron 宿主类型检查、构建通过；
- 桌面宿主 19/19；
- Python 生成器 26/26；
- 跨语言协议 10/10；
- 真实 Python stdio 子进程集成 3/3；
- 角色资产与色键重建 6/6；
- 共 64 项项目测试通过；
- 环境 smoke 全部必需检查通过，包括 PyTorch 2.10.0+cu130、RTX 5070 Ti `sm_120` CUDA、TorchVision/TorchAudio ABI、ONNX Runtime 以及所选研究仓库轻量前向/导入。

## 真实 Electron 进程验证

以根目录 `pnpm dev` 启动最终代码，宿主日志记录：

- Electron 42.7.0 完成启动；
- Python PID 28160 完成 `hello/ready`；
- 终止该生成器后，宿主进入 safe idle，约 259 ms 后创建 PID 28800，约 366 ms 后重新 `ready`；
- 终止桌宠 renderer PID 8616 后，宿主记录 `render-process-gone`，随后以相同 `renderer-client-id=4` 创建替代 PID 38024；
- 在 renderer 已恢复的状态下再次终止 Python PID 28800，宿主约 259 ms 后创建 PID 31328，约 366 ms 后重新 `ready`；
- 验证结束后按已核对的根 PID 清理整棵测试进程树，PET Electron/Python 遗留进程数为 0。

这些检查验证了真实 Electron 二进制、`get-windows` 加载、Python 进程握手、renderer 恢复以及生成器安全降级/重启链路；没有使用 mock 进程。

## 尚待人工验收

由于本轮没有取得可持续的桌面视觉控制许可，以下项目没有标记为完成：

- 猫不透明像素实际拦截点击、透明像素实际透传；
- 在真实窗口顶边连续行走、从工作区底边跳到不同高度窗口并正确落地；
- 前台全屏视频/游戏、开始菜单与系统安全界面的实际隐藏/恢复；
- 100%/150% 等真实混合 DPI 显示器边界；
- 30 分钟常驻内存/句柄稳定性；
- 真实交互点击响应与 plan 生成延迟的 p95。

上述逻辑均有自动化回归覆盖，但仍需要按 `docs/milestone-1-acceptance.md` 在真实桌面逐项观察后勾选。
