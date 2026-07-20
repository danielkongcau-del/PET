# 上游项目源码

本目录保存桌宠项目调研阶段筛选出的上游仓库。它们于 2026-07-19 以浅克隆方式取得，并保留各自的 `.git`，便于后续直接实验、比较以及追踪上游。

精确远端、分支和提交记录见 [REPOSITORIES.lock.tsv](./REPOSITORIES.lock.tsv)。

## 目录

### `desktop/`：桌面外壳与窗口互动

- `dororo`：当前首选工程底座；Godot/C#、透明窗口、鼠标与 Win32 交互。
- `clawd-on-desk`：完成度较高的 Electron 像素桌宠外壳。
- `esheep-desktop-pet`：窗口、任务栏和多屏碰撞逻辑参考。
- `bongo-cat-next`：Tauri/PixiJS 轻量桌面窗口方案。
- `openpets`：插件式桌宠平台和扩展接口参考。

### `motion/`：在线动作与轨迹生成

- `camdm`：当前首选动作扩散基底。
- `diffusion-policy`：滚动时域动作序列生成参考。
- `dart`：自回归动作 primitive 扩散参考。
- `motionlcm`：少步动作一致性蒸馏参考。
- `diffusion-forcing`：长时序生成和自回归稳定性参考。

### `pixel/`：像素帧、透明图像与实时扩散

- `diamond`：当前首选动作条件下一帧扩散基底。
- `iris`：离散图像 token 自回归世界模型参考。
- `sprite-sheet-diffusion`：角色条件像素动作及训练数据生成参考。
- `pixel-vqvae-fusion-dance`：像素画 VQ-VAE 表示参考。
- `streamdiffusion`：少步实时扩散和流水线优化参考。
- `live2diff`：因果视频扩散与时序缓存参考。
- `layerdiffuse`：RGBA/透明图层生成参考。
- `maskgit-pytorch`：帧内并行离散 token 解码参考。
- `taesd`：极小型 latent 编解码器参考。

## 克隆状态

- 所有 19 个仓库均为 shallow clone，工作树在克隆后保持干净。
- 克隆时设置了 `GIT_LFS_SKIP_SMUDGE=1`，没有主动下载 Git LFS 内容、外部模型权重或依赖。
- `live2diff` 包含 `live2diff/MiDaS` 子模块，目前未初始化；只有实际启用其深度条件功能时才需要补拉。
- `camdm` 自身以普通 Git 文件提交了 4 个 ONNX checkpoint，`taesd` 自身提交了 12 个 `.pth`；这些文件属于仓库当前提交，已随 checkout 正常取得。
- 如需完整历史，可在具体仓库中执行 `git fetch --unshallow`；不要在整个 `third_party` 上批量执行。

后续正式实现建议放在独立项目目录中，并从这些仓库选择性移植或建立明确的 fork；不要直接把 19 套依赖混装到同一个环境。
