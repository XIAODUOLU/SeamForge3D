# 项目名
SeamForge3D
含义比较贴合整个项目：

Seam：焊缝/接缝，是最终目标
Forge：工业制造、金属加工，同时也有“构建”的含义
3D：明确是三维点云项目

仓库名可以直接：

seamforge3d

项目标题：

SeamForge3D: Learning 3D Weld Seam Detection from Point Clouds

内部模型可以叫：

SeamFieldNet

# 3D Weld Seam Detection — Architecture & Development Roadmap (Rev. 2)

> **目标**：仅使用融合后的 PLY 点云（优先 `XYZ + Normal`），自动检测多个物理工件之间当前需要焊接的 3D 接触轨迹，并输出一条或多条有序、平滑的 `K×3` XYZ 焊缝轨迹。  
> **V1 推荐方案**：`PTv3 / Utonia backbone + Seam Field + Part Embedding + 拓扑感知图重建 + 原始高密度点云精修`。  
> **Rev. 2 重点**：正式覆盖直线、圆弧、闭环、空间自由曲线、卷圆/折弯/冲压类异形件，以及多焊缝和局部 junction 场景。

---

# 1. 问题定义

## 1.1 需要解决的问题

输入是一整块装配体的融合点云：

```text
P = {p_i}, i = 1...N_raw
p_i = [x, y, z, nx, ny, nz]
```

其中可能包含：

- 大板、加强板、方管、圆管、折弯件等多个物理工件；
- 卷圆壳体、锥壳、自由弯曲板、冲压/拉伸件、曲面法兰等异形零件；
- 工件自身折角、卷边、翻边、加强筋；
- 外轮廓、孔洞、长圆孔和开槽边界；
- 接触但不需要焊接的结构；
- 已焊接结构；
- 直线、圆弧、闭环、空间自由曲线等不同焊缝拓扑；
- 多条焊缝同时存在，局部甚至可能形成端点、交汇或分叉结构；
- 遮挡、残缺、噪声和多帧融合误差。

最终输出不是“几何边”，而是：

> **两个物理工件之间，当前工艺步骤需要焊接的 3D 接触/界面中心轨迹。**

应用级输出：

```python
Trajectory {
    trajectory_id: int
    confidence: float
    topology: "open" | "closed" | "segment"
    part_pair: tuple[int, int] | None
    points: float[K, 3]
}
```

其中 `topology` 用于区分：

```text
open    → 普通开放曲线
closed  → 环焊缝/闭合空间曲线
segment → 从 junction graph 中拆出的单条可执行轨迹段
```

最终机器人仍只消费有序 `K×3` XYZ；拓扑字段主要用于算法内部和调试。

下游焊枪姿态、焊接角度、接近/离开动作不属于本模型范围。

---

## 1.2 为什么不能只做法向/曲率边缘检测

以下结构都可能具有类似局部几何：

```text
自由外轮廓
同一工件折角
倒角
折弯板棱线
两工件接触边
真实焊缝
```

因此：

```text
normal discontinuity != weld seam
geometric edge != weld seam
```

任务必须引入“工件归属”和“焊缝语义”，而不是继续把焊缝定义成高曲率边。

---

## 1.3 推荐任务定义

不要让网络直接输出可变长度的 `K×3` 曲线。

定义为两个联合任务：

1. **Class-agnostic workpiece representation**
2. **3D seam centerline field estimation**

网络在降采样点云上逐点预测：

```text
seam probability
seam centerline offset
seam tangent
part embedding
```

再由确定性的图算法恢复最终曲线。

更严格地说，系统预测的最终几何对象不是永远只有一条 open curve，而是一个 **3D seam curve graph**：

```text
SeamGraph = {nodes, curve segments}
```

它可以退化为：

```text
一条 open curve
一个 closed loop
多条互相独立的 curve
带 junction 的 curve graph
```

因此网络保持局部逐点输出，拓扑由后处理显式恢复。

---

# 2. 输入与尺度

## 2.1 原始点云

真实 PLY 可能有：

```text
N_raw = 5e5 ~ 5e6+
```

禁止直接把全部点送入网络。

## 2.2 网络点云

先做 voxel downsample：

```text
voxel = 2 ~ 5 mm
N_net ≈ 30k ~ 100k
```

定义：

```text
N_raw = 原始高密度点数
N_net = 网络输入点数
```

网络负责粗语义/结构判断；原始 PLY 保留给最后毫米级精修。

## 2.3 网络输入

推荐：

```python
coord  # [N,3] = XYZ
feat   # [N,3] = Normal
```

概念上：

```text
Input = N × 6
```

同时必须做 `XYZ-only` ablation，防止模型过度依赖不稳定法向。

---

# 3. 模型总体结构

```text
XYZ + Normal
     │
     ▼
Voxel / PTv3 preprocessing
     │
     ▼
PTv3 / Utonia Encoder
     │
     ├─────────────→ Seam Probability Head  → [N,1]
     ├─────────────→ Seam Offset Head       → [N,3]
     ├─────────────→ Seam Tangent Head      → [N,3]
     └─────────────→ Part Embedding Head    → [N,8]
```

推荐 backbone：

```text
Point Transformer V3
```

优先尝试：

```text
Utonia / PTv3-compatible pretrained encoder
```

V1 不建议先实现完整 `N×Q` query-mask instance decoder；先用低维 part embedding 验证任务。

---

# 4. 网络输出定义

## 4.1 Seam probability

```python
seam_logit: [N,1]
```

经过 sigmoid：

```text
h_i ∈ [0,1]
```

含义：

> 点 `p_i` 是否位于目标焊缝附近。

---

## 4.2 Seam offset

```python
seam_offset: [N,3]
```

定义：

```text
q_i = p_i + Δ_i
```

`q_i` 应指向最近 GT 焊缝中心线。

---

## 4.3 Seam tangent

```python
seam_tangent: [N,3]
```

归一化：

```text
||t_i|| = 1
```

并认为：

```text
t_i == -t_i
```

因为焊缝切线没有正反方向。

---

## 4.4 Part embedding

```python
part_embed: [N,8]
```

目标：

```text
同一物理工件 → embedding 接近
不同物理工件 → embedding 分离
```

V1 推荐 8 维。

---

## 4.5 完整输出维度

```python
output = {
    "seam_logit":   Tensor[N,1],
    "seam_offset":  Tensor[N,3],
    "seam_tangent": Tensor[N,3],
    "part_embed":   Tensor[N,8],
}
```

概念总维度：

```text
N × (1 + 3 + 3 + 8)
= N × 15
```

注意 `N` 是 `N_net`，不是百万级 `N_raw`。

## 4.6 可选 V1.1：Topology auxiliary heads

对真实异形结构，后续可增加两个非常轻量的逐点辅助头：

```python
endpoint_logit: [N,1]
junction_logit: [N,1]
```

此时概念总输出为：

```text
N × 17
```

它们分别表示：

```text
endpoint → 附近是否为开放焊缝端点
junction → 附近是否存在多个 seam branch 交汇
```

**V1 默认不启用。**

原因：

- open/closed topology 首先可以从预测 graph 的节点度恢复；
- endpoint/junction GT 虽然容易从 synthetic seam graph 自动生成，但会增加多任务调参；
- 应先证明 `heat + offset + tangent + embedding` 本身有效。

当 V1 在异形件上出现明显的端点错连、闭环断裂或 junction 误处理时，再启用 V1.1。

---

# 5. 为什么不直接输出焊缝轨迹

直接回归曲线会遇到：

- 焊缝条数不固定；
- 每条长度不固定；
- 点数不固定；
- 多条曲线无固定顺序；
- 可能有闭环；
- 可能有遮挡和断裂；
- 轨迹方向正反等价。

因此 V1 采用：

```text
dense seam field
    ↓
centerline voting
    ↓
graph clustering
    ↓
curve ordering
    ↓
B-spline
    ↓
K×3
```

网络负责判断“哪里是焊缝、中心在哪、方向如何”；后处理负责拓扑和有序轨迹。

---

# 6. GT 标签格式

每个训练场景建议保存：

```python
coord           # float32 [N,3]
normal          # float32 [N,3]

instance_id     # int32   [N]

seam_distance   # float32 [N]
seam_heat       # float32 [N]
seam_offset     # float32 [N,3]
seam_tangent    # float32 [N,3]
```

可选：

```python
nearest_joint   # int32 [N]
sensor_conf     # float32 [N]
```

训练文件推荐 `.npz`，PLY 主要用于可视化。

---

# 7. 焊缝 GT 的定义

## 7.1 不使用“一点宽 binary label”

焊缝是 1D 曲线，点云是表面采样。直接标：

```text
exact seam point = 1
else = 0
```

会导致极端类别不平衡。

## 7.2 GT seam curve

在生成 assembly 时直接保存焊缝曲线：

```text
C_j(s)
```

建议表示成密集 polyline。

对每个点计算：

```text
d_i = distance(p_i, nearest GT seam)
```

## 7.3 Seam heatmap

推荐：

```text
h_i* = exp(-d_i² / (2σ²))
```

初值：

```text
sigma = 3 ~ 5 mm
supervision_radius = 10 ~ 15 mm
```

## 7.4 Offset GT

对 `d_i < supervision_radius`：

```text
q_i* = nearest point on seam
Δ_i* = q_i* - p_i
```

## 7.5 Tangent GT

在 `q_i*` 对应的曲线位置求单位切线：

```text
t_i*
```

## 7.6 Curve topology GT

每条 GT seam 必须同时保存 topology：

```yaml
topology: open | closed
```

对于复杂 joint，可把整体表示成 seam graph：

```yaml
nodes:
  - id: 0
    type: endpoint
  - id: 1
    type: junction

segments:
  - id: 0
    start_node: 0
    end_node: 1
    polyline: ...
```

最常见的三类：

### Open curve

```text
endpoint ───────── endpoint
```

例如：

- 加强板一条边；
- 搭接焊的一段；
- 异形板局部自由曲线。

### Closed loop

```text
      ╭──────╮
      │      │
      ╰──────╯
```

例如：

- 圆管与法兰；
- 圆筒与底板；
- 椭圆/自由曲线闭合环焊缝。

闭环 GT 必须使用周期参数化或首尾一致的 dense polyline。

### Branched / junction graph

```text
──────┬──────
      │
```

如果工艺上本质应执行为多条独立 seam segment，则 synthetic metadata 中直接保存为多个 segment，并共享 junction node。

可选 V1.1 的 `endpoint` / `junction` labels 可由该 graph 自动生成软 heatmap。

---

# 8. Loss

总损失：

```text
L =
λh L_heat
+ λo L_offset
+ λt L_tangent
+ λe L_embed
```

初始权重：

```yaml
lambda_heat: 2.0
lambda_offset: 1.0
lambda_tangent: 0.2
lambda_embed: 0.5
```

## 8.1 Heatmap

推荐 Focal Loss。

## 8.2 Offset

只在焊缝邻域监督：

```text
d_i < R_seam
```

使用：

```text
SmoothL1
```

## 8.3 Tangent

使用：

```text
L_tangent = 1 - |dot(t_pred, t_gt)|
```

自动处理 `t` 与 `-t` 等价。

## 8.4 Embedding

使用 discriminative instance embedding loss：

```text
same instance → pull
different instance → push
```

即：

```text
L_embed = L_pull + L_push + L_reg
```

---

# 9. Blender 合成场景生成

## 9.1 原则

不要：

```text
先随机摆 mesh → 最后再猜焊缝 GT
```

要：

```text
joint generator 在生成装配关系时同时返回 GT seam
```

推荐接口：

```python
part_a, part_b, joint = make_t_joint(...)
```

其中：

```python
joint = {
    "part_a": id_a,
    "part_b": id_b,
    "state": "pending",
    "seam_curves": [...]
}
```

---

# 10. 工件与异形零件生成库

不要只建立“标准 primitive 库”，而应建立 **parameterized workpiece generator library**。

至少包含标准类：

```text
plate
thick plate
square tube
rectangular tube
round tube
angle steel
channel
H/I beam
gusset
triangular plate
flange
custom extrusion
```

以及异形类：

```text
rolled cylindrical shell
conical shell
freeform bent plate
curved flange
swept profile
variable-radius tube
stamped-like sheet
ear / tab / bracket
flanged sheet
multi-bend sheet-metal part
curved rib
annular/ring component
```

## 10.1 推荐 Blender/bpy 建模操作

优先程序化组合：

```text
extrude
sweep
bevel
solidify
simple deform / bend
curve modifier
lattice deform
boolean holes
loft-like section interpolation
skin/profile along curve
```

目标不是复刻每一种真实产品，而是扩大：

```text
surface curvature distribution
part topology
boundary topology
cross-section family
local feature diversity
```

## 10.2 Freeform curve driven part

建议大量使用随机 B-spline / Bezier centerline：

```text
C(u)
```

沿 `C(u)` sweep：

```text
rectangle
circle
custom profile
thin plate strip
```

可快速生成现实中很难枚举的自由曲面/空间弯曲件。

## 10.3 Sheet-metal style generator

对薄板零件增加：

```text
随机折弯次数
折弯角
折弯半径
局部翻边
开孔/槽
圆角
局部加强筋
```

并保证：

> 无论一个零件内部有多少平面、曲面和折弯，它始终只有一个 `instance_id`。

这是防止模型把“一块复杂零件”错误拆成多个工件的核心 GT 约束。

## 10.4 参数随机化

随机化：

```text
长度
宽度
厚度
直径
截面
圆角
倒角
曲率
曲率变化率
扭转
孔洞数量
开槽数量
姿态
尺寸比例
```
# 11. Joint / Interface Generator

V1 可以保留常见 joint helper：

```text
T joint
lap joint
butt joint
corner joint
plate-tube
tube-tube
gusset-plate
oblique T
curved contact
flange contact
```

但长期接口不要绑定具体 joint 名称，而应抽象为：

```python
joint = connect(
    part_a,
    interface_a,
    part_b,
    interface_b,
    weld_policy=...,
)
```

每个 workpiece generator 输出若干：

```text
interface curve
interface loop
interface surface
```

`connect()` 负责：

1. 对齐两个 interface；
2. 施加随机装配公差；
3. 决定哪些 interface boundary 是待焊 seam；
4. 输出精确 world-space seam graph；
5. 保存 `part_pair` 与 joint state。

这样圆筒、法兰、自由曲面板、异形冲压件都可以使用同一套装配逻辑。

## 11.1 Joint angle

不要只训练 90°。

在物理合理范围内随机：

```text
20° ~ 160°
```

并加入：

```text
局部相切
斜交
曲面-平面
曲面-曲面
环形接触
非恒定夹角接触
```

## 11.2 Contact policy 与 Weld policy 分离

必须显式区分：

```text
contact relation
weld-required relation
```

因为：

```text
接触 ≠ 一定需要焊接
```

建议 joint metadata：

```yaml
contact: true
weld_required: true
state: pending
```

Hard negative 可生成：

```yaml
contact: true
weld_required: false
```

这会迫使模型学习工件关系和训练分布，而不是把所有接触边都当 seam。
# 12. 焊缝 GT 与 Curve Generator

尽量不要依赖 Boolean 后重新寻找交线。

更推荐：

```text
part local interface
    ↓
joint generator 明确知道 seam
    ↓
transform 到 world
    ↓
保存 GT curve / seam graph
```

例如：

```text
加强板底边
圆管端部 loop
搭接板边缘
曲面法兰边界
自由弯曲板 interface curve
```

在生成时就是已知曲线。

## 12.1 Seam curve family

Synthetic generator 必须覆盖：

```text
straight line
polyline with corner
circular arc
ellipse
closed circle
closed ellipse
planar B-spline
3D B-spline
periodic B-spline
spatial helical-like curve
piecewise smooth curve
```

网络不需要知道 curve type；curve type 只用于数据生成和 evaluation stratification。

## 12.2 Curvature randomization

对于自由曲线 `C(s)`，随机化并约束：

```text
total length
mean curvature
maximum curvature
minimum curvature radius
curvature change rate
torsion / out-of-plane deformation
closed/open probability
```

训练集要覆盖宽广的曲率分布。

不要只随机“零件尺寸”，否则模型可能只学到标准 T-joint / 环焊缝模板。

## 12.3 Closed-loop GT

闭环曲线必须保证：

```text
C(0) == C(1)
t(0) == t(1)
```

或使用足够密的 periodic polyline。

用于 nearest point / tangent GT 时必须处理首尾周期邻接。

## 12.4 Junction GT

如果 seam graph 存在 junction：

```text
一个 junction node
+
多个 curve segments
```

训练时仍按所有 segment 的距离构造 seam field。

最终机器人执行层可以把 graph 拆成多条 segment。
# 13. Scene metadata

每个 scene 保存：

```yaml
scene_id: 000123

parts:
  - id: 0
    type: plate
  - id: 1
    type: tube

joints:
  - id: 0
    part_a: 0
    part_b: 1
    state: pending
    seam_file: seam_000.npy
```

物理工件 instance 定义始终按原始零件，而不是按“是否已经焊死”。

---

# 14. 已焊接问题

即使两块工件已经焊好，也仍然保持：

```text
instance A
instance B
```

不要合并成一个 instance，否则 instance GT 会失去稳定定义。

Joint state 单独保存：

```text
pending
done
ignore
```

如果焊前/焊后几何完全相同，纯 PLY 无法推断工艺状态。

V1 推荐由流程层维护：

```text
done joint blacklist
```

如果真实扫描可以明显看到 weld bead，后续可增加“已焊焊道”合成几何和状态分类。

---

# 15. Hard Negatives

这是数据集最重要的部分之一。

必须大量生成：

```text
同一工件锐角
同一工件折弯
卷边/翻边
冲压加强筋
倒角
外轮廓
孔洞/长圆孔/槽边界
圆管/型材棱线
自由曲面自身高曲率 ridge
几乎接触但未接触
夹具接触
接触但不焊
已完成/忽略 joint
```

全部：

```text
seam = 0
```

否则网络会退化成新的 edge detector。

---

# 16. Mesh → 点云：不要只 uniform sample

不要把：

```python
mesh.sample_points_uniformly()
```

作为主训练数据。

真实融合点云具有：

```text
遮挡
视角缺失
密度变化
边缘飞点
深度噪声
融合重影
配准误差
局部缺失
```

因此必须模拟扫描过程。

---

# 17. Open3D RaycastingScene 采集

推荐：

```text
Blender
  ↓
生成独立 part meshes + seam metadata
  ↓
Open3D RaycastingScene
  ↓
多虚拟相机 depth raycast
  ↓
世界坐标融合
  ↓
模拟噪声
  ↓
voxel
  ↓
normal estimation
```

每个物理工件作为独立 geometry 加入 raycasting scene。

这样 ray hit 的 `geometry_id` 天然就是：

```text
instance_id
```

无需人工标 point instance。

---

# 18. 虚拟相机

随机：

```text
camera count
azimuth
elevation
distance
FOV
coverage
```

初值：

```text
6 ~ 30 views / scene
```

并故意制造部分 scene 的非 360° 覆盖。

针对异形结构还要随机制造：

```text
深凹区域不可见
环件内外侧只看到一部分
曲面自遮挡
耳板/翻边遮挡 seam
闭环焊缝局部缺失
```

模型必须学会在不完整可见性下保持 seam continuity，而不是依赖完整 CAD 表面。

---

# 19. 多视角融合

每个视角：

```text
raycast
→ visible XYZ
→ transform to world
→ merge
```

然后再模拟：

```text
pose noise
fusion error
voxel downsample
normal recomputation
```

最终 synthetic PLY 应尽可能像真实重建结果，而不是 CAD 表面。

---

# 20. 数据增强

随机删 20% 点可以保留，但必须增加结构化增强。

## 20.1 Point dropout

```text
0 ~ 20%
```

## 20.2 View dropout

```text
删除 0 ~ 50% 虚拟视角
```

比随机删点更接近真实缺失。

## 20.3 Patch dropout

随机移除 3D sphere/box/局部 patch。

## 20.4 Grazing-angle dropout

视线与表面接近平行时提高丢点率。

## 20.5 Edge noise

几何深度边缘提高：

```text
jitter
dropout
flying points
```

## 20.6 XYZ noise

初值：

```text
0.2 ~ 1.0 mm
```

之后按真实传感器测量。

## 20.7 Pose noise

对每个虚拟 view 的位姿加入小扰动，再融合。

## 20.8 Normal noise

```text
2° ~ 8°
```

## 20.9 Normal sign augmentation

随机：

```text
n → -n
```

用于适应真实法向方向不一致。

## 20.10 装配公差

随机：

```text
gap            0 ~ 3 mm
translation    ±1 ~ 3 mm
angle error    ±0 ~ 3°
```

按真实工艺调整。

---

# 21. 数据集规模

## Phase 1

```text
3,000 ~ 5,000 scenes
```

目的：

- 验证 heatmap；
- 验证 offset；
- 验证 tangent；
- 验证 embedding；
- 验证曲线重建。

## Phase 2

```text
30,000 ~ 100,000+ scenes
```

扩大几何、尺寸、joint、噪声和扫描策略。

---

# 22. Dataset split

禁止只做随机 90/10。

至少增加：

```text
shape holdout
freeform-family holdout
dimension holdout
joint topology holdout
curve-topology holdout
curvature-range holdout
noise-domain holdout
```

例如：

```text
tube + oblique plate
```

组合完全不出现在 train，只在 test 出现；

或者：

```text
train: open/低曲率 seam
test: closed/freeform 高曲率 seam
```

用于判断模型是否真的学到局部 seam field，而不是背诵模板。

目的是确认模型学到“工件关系 + seam centerline field”，而不是背固定产品形状。

---

# 23. 推荐训练阶段

## Stage A — Seam Field only

先训练：

```text
heat + offset + tangent
```

暂时关闭 embedding。

目标：

> synthetic validation 上 centerline voting 能稳定回到 GT。

## Stage B — Part Embedding

加入：

```text
[N,8] part embedding
```

观察是否能压制 same-part crease hard negative。

## Stage C — Synthetic scale-up

扩大 procedural data 和 sensor randomization。

## Stage D — Real fine-tuning

准备：

```text
50 ~ 200 个经过人工确认的真实 scene
```

用于小学习率 fine-tune。

---

# 24. 推理流程

```text
Dense raw PLY
  N_raw ~ millions
        ↓
voxel 2~5 mm
        ↓
N_net ~ 30k~100k
        ↓
PTv3/Utonia
        ↓
heat + offset + tangent + embedding
```

---

# 25. Seam centerline voting

筛选：

```text
sigmoid(seam_logit) > threshold
```

初值可试：

```text
0.2 ~ 0.4
```

每个候选：

```text
q_i = p_i + offset_i
```

得到 3D seam vote cloud。

---

# 26. Tangent graph

候选 vote `q_i, q_j` 在满足以下条件时连边：

```text
||q_i - q_j|| < r
|dot(t_i, t_j)| > cos(theta)
```

初值：

```text
r = 3 ~ 8 mm
theta = 20° ~ 35°
```

---

# 27. Part consistency

对每条候选 seam 两侧邻域检查 part embedding。

如果：

```text
两侧 embedding 相同
```

降低置信度。

如果：

```text
两侧形成两个稳定不同 embedding cluster
```

提高置信度。

这一步主要用来抑制：

```text
same-part crease
```

不要求先得到整场漂亮的 instance segmentation。

---

# 28. Seam Graph Topology Recovery

不要默认每个 component 都是“两个端点之间的最长路径”。

对每个 tangent-consistent graph 先分析拓扑。

## 28.1 Graph cleanup

```text
remove isolated nodes
remove very short spurs
bridge short high-confidence gaps
merge duplicate nearby votes
```

## 28.2 Open curve

若 graph 存在两个稳定 degree-1 端点：

```text
endpoint ───────── endpoint
```

则：

```text
endpoint-to-endpoint geodesic
```

作为主路径。

可先实现：

```text
MST + longest valid path
```

## 28.3 Closed loop

若：

```text
几乎所有主干节点 degree == 2
且无稳定 endpoint
```

则识别为：

```text
closed
```

按 cycle 顺序遍历，不要人为切掉最长路径。

## 28.4 Junction / branched graph

若存在：

```text
degree >= 3
```

的稳定区域：

1. 聚合为 junction node；
2. 在 junction 间拆成多个 segment；
3. 每个 segment 单独排序和平滑；
4. 保留共享 junction 坐标。

## 28.5 Topology confidence

每条 curve/component 输出：

```text
open_confidence
closed_confidence
junction_confidence
```

V1 可以由 graph heuristic 计算。

V1.1 可融合：

```text
endpoint_logit
junction_logit
```

进行辅助判定。
# 29. 曲线平滑与参数化

不同 topology 使用不同 spline。

## 29.1 Open seam

推荐：

```text
robust median binning
→ cubic B-spline
→ arc-length resampling
```

## 29.2 Closed seam

使用：

```text
periodic cubic B-spline
```

要求首尾：

```text
position continuous
tangent continuous
```

不能简单把闭环随意切开再拟合普通 spline，否则首尾容易产生折点。

## 29.3 Junction segments

每个 branch 单独 spline，但共享 junction endpoint。

避免每条 spline 独立平滑后产生：

```text
junction gap
```

## 29.4 Corner preservation

并非所有 seam 都应该被强行平滑成圆滑曲线。

若 GT/预测存在真实折点：

```text
piecewise spline
```

优于单一强平滑 spline。

最终统一按物理弧长重采样，例如：

```text
2 mm / point
```
# 30. 原始高密度点云精修

网络轨迹不是最终几何精度。

把粗轨迹投回：

```text
N_raw 原始 PLY
```

对每个粗轨迹点建立：

```text
±5 ~ 10 mm local ROI
```

再做局部高分辨率几何精修。

这样：

```text
network voxel size != final seam accuracy
```

---

# 31. 最终输出

推荐：

```csv
trajectory_id,point_index,x,y,z,confidence
0,0,...
0,1,...
...
```

以及可选：

```text
part_a
part_b
trajectory confidence
length
topology = open | closed | segment
graph_id
junction_start_id
junction_end_id
```

---

# 32. 评价指标

不要只看 segmentation IoU。

## Seam field

```text
heat precision / recall
offset error
tangent angle error
```

## Part representation

```text
same/different-part pair accuracy
embedding clustering IoU / ARI
```

## Final trajectory

必须包含：

```text
Precision@3mm
Recall@3mm
Precision@5mm
Recall@5mm
Chamfer Distance
Hausdorff-95
GT length completeness
false seam length
component count error
topology classification accuracy
closed-loop closure error
junction localization error
```

业务上重点看：

```text
Recall@tolerance
False Seam Length
Trajectory Completeness
```

对闭环焊缝额外检查：

```text
Loop Completeness
Closure Gap
```

对 junction 场景额外检查：

```text
Branch Recall
Junction Position Error
```

---

# 33. 推荐项目目录

```text
weld-seam-3d/
├── architecture.md
├── configs/
├── synthetic/
│   ├── blender/
│   │   ├── primitives/
│   │   ├── freeform/
│   │   ├── curves/
│   │   ├── joints/
│   │   └── assemblies/
│   ├── sensor/
│   │   ├── raycast.py
│   │   ├── cameras.py
│   │   ├── fusion.py
│   │   └── noise.py
│   └── labels/
│       ├── seam_labels.py
│       └── build_npz.py
├── datasets/
├── models/
│   ├── backbone/
│   ├── weld_model.py
│   ├── seam_head.py
│   └── embedding_head.py
├── losses/
├── postprocess/
│   ├── voting.py
│   ├── graph.py
│   ├── topology.py
│   ├── part_consistency.py
│   ├── ordering.py
│   ├── spline.py
│   └── dense_refine.py
├── metrics/
├── tools/
│   ├── train.py
│   ├── infer.py
│   └── visualize.py
└── tests/
```

---

# 34. 实施路线

## Milestone 0 — 冻结任务定义

- [ ] 统一坐标单位为米
- [ ] 确认 PLY 字段
- [ ] 确认物理工件 instance 定义
- [ ] 确认 pending/done joint 策略
- [ ] 确认最终 trajectory 文件格式
- [ ] 选初始 voxel size

**完成标准**：同一个 scene 的 GT 不依赖模型，并且定义唯一。

---

## Milestone 1 — Blender procedural assembly

- [ ] standard primitive generators
- [ ] freeform / sweep / bend / rolled-part generators
- [ ] hole / slot / flange / tab feature generators
- [ ] seam curve generator: line / arc / B-spline / periodic loop
- [ ] interface-based joint generator API
- [ ] part IDs
- [ ] exact seam polyline / seam graph
- [ ] open / closed / junction topology metadata
- [ ] joint state
- [ ] hard-negative generators
- [ ] metadata export

**完成标准**：任意生成场景都有精确 part ID 和 seam GT。

---

## Milestone 2 — Open3D virtual scanner

- [ ] RaycastingScene
- [ ] camera sampler
- [ ] geometry_id → instance_id
- [ ] multi-view fusion
- [ ] view dropout
- [ ] pose noise
- [ ] point/patch dropout
- [ ] normal estimation
- [ ] voxelization

**完成标准**：synthetic PLY 在外观和点分布上开始接近真实 PLY。

---

## Milestone 3 — GT builder

- [ ] instance_id
- [ ] nearest seam search
- [ ] seam_distance
- [ ] seam_heat
- [ ] seam_offset
- [ ] seam_tangent
- [ ] NPZ writer
- [ ] GT visualizer

**完成标准**：生成场景可自动转成训练样本，无人工标注。

---

## Milestone 4 — Seam Field baseline

- [ ] PTv3/Utonia integration
- [ ] heat head
- [ ] offset head
- [ ] tangent head
- [ ] focal loss
- [ ] SmoothL1
- [ ] tangent loss
- [ ] trainer
- [ ] validation

**完成标准**：synthetic validation 上 votes 能回到真实 seam centerline。

---

## Milestone 5 — Trajectory reconstruction

- [ ] candidate threshold
- [ ] centerline voting
- [ ] tangent graph
- [ ] component extraction
- [ ] graph cleanup / short-gap bridging
- [ ] open-curve endpoint recovery
- [ ] closed-loop cycle detection
- [ ] junction detection / branch splitting
- [ ] branch pruning
- [ ] curve ordering
- [ ] ordinary + periodic spline fitting
- [ ] corner preservation
- [ ] arc-length resampling
- [ ] topology + trajectory metrics

**完成标准**：输出稳定有序 `K×3`。

---

## Milestone 6 — Part Embedding

- [ ] embedding head `[N,8]`
- [ ] discriminative embedding loss
- [ ] embedding visualization
- [ ] same/different-part metric
- [ ] seam-side consistency
- [ ] same-part false-edge suppression

**完成标准**：hard-negative crease 显著减少，seam recall 不明显下降。

---

## Milestone 7 — Generalization benchmark

- [ ] 3k~5k synthetic scenes
- [ ] shape holdout
- [ ] freeform-family holdout
- [ ] curvature-range holdout
- [ ] open/closed curve-topology holdout
- [ ] dimension holdout
- [ ] joint holdout
- [ ] noise holdout
- [ ] XYZ-only ablation
- [ ] XYZ+Normal ablation
- [ ] normal-sign augmentation ablation

**完成标准**：未见过的 synthetic 组合仍能检测。

---

## Milestone 8 — Real benchmark

先准备：

```text
20 ~ 50 个真实 scene
```

只用于评估，不全部参与训练。

人工标真实 seam polyline。

错误分类：

```text
miss
same-part false seam
outer-boundary false seam
wrong component
partial seam
broken seam
offset error
```

**完成标准**：知道 synthetic→real 的主要误差来源。

---

## Milestone 9 — Real fine-tuning

扩展：

```text
50 ~ 200 verified real scenes
```

- [ ] mixed synthetic + real
- [ ] lower learning rate
- [ ] sensor augmentation calibration
- [ ] threshold tuning
- [ ] held-out real test set

**完成标准**：真实测试集达到项目误差容限。

---

## Milestone 10 — Dense refinement

- [ ] map coarse curve back to raw PLY
- [ ] local dense ROI
- [ ] high-resolution geometry refinement
- [ ] confidence propagation
- [ ] final export

**完成标准**：最终精度主要由原始点云质量决定，而不是网络 voxel 大小。

---

# 35. V1 初始配置

```yaml
input:
  voxel_size: 0.003
  use_normal: true

model:
  backbone: PTv3/Utonia
  part_embedding_dim: 8
  use_endpoint_head: false
  use_junction_head: false

seam:
  sigma: 0.004
  supervision_radius: 0.012

loss:
  heat: 2.0
  offset: 1.0
  tangent: 0.2
  embedding: 0.5

postprocess:
  seam_prob_threshold: 0.30
  graph_radius: 0.006
  tangent_angle_deg: 30
  min_component_length: 0.015
  spline_sample_spacing: 0.002
```

长度单位全部为米。

---

# 36. 必做 Ablation

## 输入

```text
A: XYZ
B: XYZ + Normal
```

## 输出任务

```text
A: heat
B: heat + offset
C: heat + offset + tangent
D: heat + offset + tangent + embedding
```

## Synthetic acquisition

```text
A: uniform mesh sample
B: raycast multi-view
C: raycast + sensor noise
D: raycast + noise + pose/fusion noise
```

## Hard negative

```text
A: none
B: same-part crease
C: all hard negatives
```

## Curve topology

```text
A: open seam only
B: open + closed
C: open + closed + freeform
D: open + closed + freeform + junction
```

## Optional topology heads

```text
A: graph heuristic only
B: + endpoint head
C: + endpoint + junction heads
```

---

# 37. 主要风险

## Synthetic → Real domain gap

这是最大风险。

优先通过：

```text
模拟真实扫描
测量真实传感器噪声
加入 pose/fusion error
少量真实 fine-tune
```

解决。

## Same-part crease

通过：

```text
hard negatives
part embedding
joint multi-task learning
local side consistency
```

解决。

## 焊缝局部缺失

通过：

```text
view dropout
patch dropout
graph gap bridging
confidence-aware spline
```

解决。

## 小结构低于 voxel 尺度

网络只粗定位，最终回原始 dense PLY 精修。

## 已焊/未焊完全同几何

纯点云不可辨识，交给流程层或显式 weld bead 几何。

## 异形件和闭环 seam 在 synthetic 中覆盖不足

如果训练集主要由标准 T-joint 和直线组成，即使模型结构支持自由曲线，也不会自然获得泛化能力。

缓解：

```text
freeform workpiece generators
curvature-domain randomization
periodic closed-loop GT
curve-topology holdout benchmark
```

## 局部几何不可观测

若两个工件在局部完全重合/相切，扫描结果不包含可辨别边界，则 local geometry 可能不足。

缓解：

```text
使用 PTv3 全局上下文
使用 part embedding
增加多尺度上下文
必要时引入工艺状态/CAD/视觉信息
```

---

# 38. 后续升级路径

只有 V1 定量验证后再考虑：

```text
part embedding
→ Mask3D/ISBNet-style full instance decoder

graph topology heuristic
→ endpoint/junction auxiliary heads

single-scale seam field
→ multi-scale seam field

graph postprocess
→ learned graph network

class-agnostic
→ auxiliary workpiece semantic head

point-wise prediction
→ curve query decoder
```

---

# 39. 最终 V1 固定定义

**输入**

```text
N × 6
XYZ + Normal
```

其中 `N = N_net ≈ 30k~100k`。

**Backbone**

```text
PTv3 / Utonia
```

**网络输出**

```text
seam_logit    N × 1
seam_offset   N × 3
seam_tangent  N × 3
part_embed    N × 8
```

概念总输出：

```text
N × 15
```

**后处理**

```text
heat threshold
→ centerline voting
→ tangent graph
→ part consistency
→ topology recovery (open / closed / junction)
→ curve ordering
→ ordinary / periodic B-spline
→ dense PLY refinement
```

**最终业务输出**

```text
trajectory_0: K0 × 3, topology=open
trajectory_1: K1 × 3, topology=closed
trajectory_2: K2 × 3, topology=segment
...
```

该定义天然允许：

```text
直线
圆弧
环焊缝
椭圆焊缝
空间自由曲线
异形曲面接触 seam
多条独立 seam
junction graph 拆分后的多个 segment
```

因此真实工件是否规则，不需要改变网络输出接口。

核心原则保持不变：

> **网络负责“理解正确的焊缝区域和工件关系”；图算法负责“轨迹拓扑与排序”；原始高密度点云负责“最终几何精度”。**


---

# 40. Rev. 2 设计结论

针对真实工业场景中的卷圆件、自由弯曲板、冲压件、圆筒/法兰环焊缝以及复杂附件，本架构的核心模型定义无需改变：

```text
N×6
→ PTv3/Utonia
→ N×15
```

真正需要扩大的是：

```text
Synthetic geometry distribution
Seam curve topology distribution
Sensor visibility distribution
Graph reconstruction capability
```

开发时应避免“为每一种产品单独写规则”，而应让 synthetic generator 在参数空间中随机生成：

```text
基础零件拓扑
+ 自由曲线
+ 自由曲面
+ sweep / bend / roll
+ 孔洞 / 槽 / 翻边
+ interface-based assembly
+ open / closed / junction seam
+ 随机视角与遮挡
```

最终希望模型学习的不是：

> 某种焊缝长得像什么。

而是：

> 哪些点属于不同物理工件、哪些区域对应待焊界面，以及该界面的 3D 中心线和局部走向。

这也是本项目应长期保持的任务定义。
