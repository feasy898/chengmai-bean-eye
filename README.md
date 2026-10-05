# 啡眼 BeanEye · 咖啡生豆 AI 质检台

面向海南罗布斯塔咖啡生豆的全开源桌面质检台：350g 样品铺盘双面拍摄，自动完成瑕疵豆识别计数、粒度色泽计量、按标准一键定级，输出带二维码验真的中 / 英 / 越三语质量护照，并由大模型给出加工环节溯因建议。

**核心能力**

- 双面成像逐粒分割，上下配对（每粒只计最严重缺陷，内部计数规则；标准阈值核对完成前不作为 CQI 符合性声明）
- ArUco 标定到毫米：目数分布、色差、粒数、估算重量
- 标准引擎：YAML 描述 CQI Fine Robusta / 国标 / 地标阈值，换标准不改代码
- 质量护照：每个缺陷附原图证据 + 二维码验真
- 溯因智能体：黑豆 / 酸豆 / 霉豆 / 虫蛀 → 采摘 / 发酵 / 干燥 / 仓储环节定位与改进建议（规则模板离线保底，LLM 后端为可选增强）
- 合成数据引擎：程序化豆素材 + 铺盘合成器，无真实样品阶段全程可演示、可回归

**精度与口径声明（重要）**

- 当前全部精度数字为**合成盘自洽口径**（合成真值 / oracle vs 管线输出），真机标定前
  不得对外引用为现场精度——见 `docs/calibration-error-budget.md`。
- 三套标准 YAML 均为**结构冻结 + 阈值占位**（`verified:false`），生豆到货后按
  `docs/collect_protocol.md` 采集回填（D7）；此前定级结果不作为标准符合性声明。
- 产品演示定位为「**AI 初检 + 人工复核**」辅助，最终质检结论以人工复核为准。

**仓库结构（现状）**

```
beaneye/          全部实现（Python 包）
  acquisition/    采集（Mock / USB / Synth 三源）
  calibration/    ArUco 标定与正射变换
  segment/        分割（ClassicSeg 经典 CV / OracleSeg 合成真值）
  classify/       分类（RulesV0 规则表 / NnOnnxClassifier 神经网络）
  pairing/        上下配对
  severity/       严重度裁决
  metrology/      计量（目数 / 色差 / 估重）
  standards/      标准引擎（YAML 规则）
  report/         质量护照生成与验真
  agent/          溯因智能体
  realtime/       实时标注引擎（SO101 机械臂联动 / classifier 开关）
  kb/             知识库检索（接口留桩）
  app/            FastAPI 应用、API 与单页演示 UI
  synth/          合成数据引擎（素材库 + 铺盘合成器）
configs/          托盘 / 相机 / 标准 / 规则 / 合成等 YAML 配置
data/             数据集目录（本体不入库，见「数据集」节）
docs/             自采协议（collect_protocol）、校准误差预算等
docs/assets/      资产包：manifest / CONTRACTS / REGENERATE / 逐模块 specs
scripts/          doctor 体检、验收门 gate_d1–d4、数据集下载器、e2e 与演示脚本
tests/            逐模块 eval
```

> 逐模块规格、契约与整仓再生步骤见 `docs/assets/`（manifest + specs + REGENERATE）。

---

## 预训练模型权重

本仓**不含模型权重文件**（`.gitignore` 排除 `*.onnx` / `models/`，单权重 43 MB 量级走 LFS 托管更合适）。
已训练的权重发布在 ModelScope：

**`Cloudbird/beaneye-crop-classifier-b13`** — <https://modelscope.cn/models/Cloudbird/beaneye-crop-classifier-b13>

| 文件 | 说明 |
|---|---|
| `crop_cls.onnx` | ONNX 推理权重，opset 17，静态输入 `[1,3,224,224]`，输出 13 维 logits |
| `best.pt` | PyTorch checkpoint，供继续训练 / 复现 |
| `metrics.json` | 完整训练与评测记录（逐类指标、混淆矩阵、延迟基准、门禁判定） |

**类别序固定**（第 0 位 = `normal`）：

```
normal, broken, faded, brocade, immature, peaberry,
shell, elephant, insect, dried, sour, mold, black
```

**实测性能**（口径见 ModelScope model card，如实记录）：

| 指标 | 值 |
|---|---|
| 真实照片 valid macro-F1 | 0.6929 |
| 真实照片 valid accuracy | 0.8932 |
| 合成 test macro-F1（门禁阈值 0.85） | 0.8798 ✅ |
| ONNX↔PyTorch 推理一致性 | max abs diff 2.86e-06 |
| CPU 推理延迟 batch=1 | mean 8.20 ms / p95 8.27 ms |

> ⚠️ 真实 macro-F1（0.69）明显低于合成 test（0.88），合成与真实域间仍有差距；
> `brocade`（花脸）类真实样本为 0；真实数据中 `sour` 大量被判为 `elephant`，是最大单一误差源。
> 这些是本模型当前的真实短板，不是修饰后的数字。

### 用起来

下载 `crop_cls.onnx` 后，通过 `NnOnnxClassifier` 接入：

```python
from beaneye.classify.nn_onnx import NnOnnxClassifier

clf = NnOnnxClassifier("crop_cls.onnx")     # τ 缺省 = 批13 校准值
result = clf.predict(crop_rgb)              # -> SegResult/分类结果
```

或走实时引擎（`beaneye/realtime/`）：

```python
from beaneye.realtime.engine import build_classifier

clf = build_classifier("nn", nn_onnx="path/to/crop_cls.onnx", tau=None)
```

`beaneye/classify/rules.py` 的规则表分类器（`RulesV0`）不需要任何权重，离线可跑，是本仓的零依赖默认路径。

---

## 数据集

**本仓不含任何数据集本体**（`.gitignore` 排除 `data/*`，仅保留 `data/datasets/README.md` 说明书与
`scripts/download_datasets.py` 下载器）。训练与评测所用的真实照片全部来自下列**公开数据集**，
需自行获取（许可均为 **CC BY 4.0**，归原作者所有）：

| 数据集 | 规模 | 平台链接 |
|---|---|---|
| 生咖啡豆缺陷检测集（12 类，西语类名） | 4038 图 | <https://universe.roboflow.com/redtraining/defectoscafeverde> |
| 罗布斯塔豆实例分割集（4 类） | 418 图 | <https://universe.roboflow.com/roasted-coffee-bean-defect/robusta-coffee-bean-defect> |
| 罗布斯塔绿豆检测集（2 类，bbox） | 250 图 | <https://universe.roboflow.com/nur-muhammad-himawan/robusta-green-coffee-bean-defect> |
| SCAA 生豆缺陷检测集（17 类） | 966 图 | <https://universe.roboflow.com/green-coffee-bean-defects/gcb-defect-detection> |

> 全部托管在 **Roboflow Universe**。下载导出版本需要 **Private API Key**
> （Publishable Key 只能托管推理，导出会 401）。

获取步骤、目录规范（`manifest.json` / `SHA256SUMS` / `mapping.yaml`）与校验命令见
[`data/datasets/README.md`](data/datasets/README.md)。合成训练数据（铺盘合成器产出）
无需下载，仓内代码可程序化重建。

**类别映射说明**：上述数据集用西语/英文类名（如 `broca`、`agrio`、`partido`），
本仓通过 `data/datasets/*/mapping.yaml` 映射到内部 13 类 taxonomy
（`configs/defect-taxonomy.yaml`）。注意 `broca` 是虫蛀（→ `insect`），
不是 `brocade`（花脸）——旧映射曾误配，已纠正。

---

**规划中（未开工）**：AR 指选手选辅助、真机采集与阈值回填。
