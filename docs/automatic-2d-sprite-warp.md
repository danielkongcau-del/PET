# 自动 2D 骨骼像素形变

当前 Canvas renderer 的正常 `sprite` 路径会消费角色 manifest 中的通用 3D
joint graph，但它不是 glTF mesh skinning，也不读取顶点权重。其用途是让尚未准备好
2D 分层素材或 mesh buffer 的角色，在 48×48 像素桌宠中仍能看到骨骼动作。

## 契约

- 输入姿态使用 manifest 声明的任意 joint 拓扑和 `drivenJointOrder`，不假定猫的
  关节名称或数量。
- 3D 路径直接使用 `rootTranslation`、`rootRotation` 与 local quaternion deltas。
- legacy 2D 路径把每个标量角解释为正交侧视平面内的旋转：先将投影深度轴从
  model space 变换到该 joint 的 rest-local space，再生成 local delta quaternion，
  最终复用同一套 FK 和像素形变。
- 角色或能力降级传入 `null` rig 时，renderer 必须清除 FK、rest pose 和 warp cache；
  旧姿态不能继续作用于图像。

## 形变流程

1. 使用 rest pose 建立一次固定的 model-to-canvas 投影。运动中不重新 fit，因此
   root 和关节位移不会被动态缩放抵消。
2. 为源图每个像素预计算最多四个最近 joint 的确定性反距离权重。
3. 每帧由 FK 得到各 joint 的投影位置和朝向。像素通过绑定 joint 的二维刚体逆变换
   回采样源图，并使用最近邻采样保持像素边缘。
4. 每个 joint 都提供位置和旋转控制；因此即使 leaf joint 的原点不移动，其 local
   rotation 也会改变附近的最终像素。
5. identity pose 走精确复制分支。资源提取、绑定、形变或上传任一步失败时，骨骼
   sprite 不得静态显示，而是明确降级到 pose-aware debug skeleton。

## 已知限制

- 自动最近关节绑定不知道角色的真实蒙皮权重，肢体交界处可能出现拉伸或接缝。
- 多 joint 混合是二维刚体近似，不能还原透视、自遮挡、材质或真实 3D mesh 体积。
- 深度排序只用于 skeleton fallback；单张 raster warp 无法把身体部件重新分层。
- 高质量角色仍应提供正式的 2D 分层/权重资产，或在 renderer 完成 glTF
  vertex/index/weight/texture 消费后使用真正 mesh skinning。

