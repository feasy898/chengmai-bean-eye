"""NN ONNX 分类头 eval（批10 接入）：beaneye/classify/nn_onnx + realtime 开关。

运行（仓库根）::

    pytest tests/test_nn_onnx.py -q

离线纪律：**不依赖真 ONNX 文件、不依赖 onnxruntime、不依赖网络**——推理
会话用 duck-type 替身（MockSession：固定 logits + 输入张量捕获），预
处理数值直接对捕获张量断言（训练口径：alpha 白底合成 → 224² INTER_AREA
→ /255 → ImageNet mean/std → CHW）。τ 判决逻辑对照批10 扫描口径
（``max(12 缺陷类 softmax) >= τ → defect，类别=缺陷类 argmax``）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from beaneye.classify import DEFAULT_TAU, NnOnnxClassifier, NnOnnxError, RulesV0
from beaneye.realtime import (
    ClassifierUsageError,
    RealtimeEngine,
    build_classifier,
)
from beaneye.schemas import BeanMask
from beaneye.taxonomy import load_taxonomy

TAX = load_taxonomy()
CLS = TAX.severity_order  # 训练类别序 = taxonomy severity_order（第 0 位 normal）

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixture_builders import bean_mask_top_017  # noqa: E402

# ImageNet 常数（与 nn_onnx 内部一致，供数值断言）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# 替身：ONNX 会话（固定 logits + 捕获输入张量）
# ---------------------------------------------------------------------------


class MockSession:
    """duck-type onnxruntime.InferenceSession：run() 返回预置 logits。"""

    def __init__(self, logits: np.ndarray, input_name: str = "input") -> None:
        self._logits = np.asarray(logits, dtype=np.float32)
        self._input_name = input_name
        self.last_input: np.ndarray | None = None
        self.n_calls = 0

    def get_inputs(self) -> list:
        return [SimpleNamespace(name=self._input_name)]

    def run(self, _keys, feed: dict) -> list:
        self.n_calls += 1
        self.last_input = feed[self._input_name]
        return [self._logits[None]]  # 真件为批输出 [1,13]，`[0][0]` 取单样本


def _logits_for(p_normal: float, top_defect: str, p_defect: float) -> np.ndarray:
    """按指定概率构造 13 维 logits（softmax 前；相对差即可精确编码概率）。"""
    probs = np.zeros(len(CLS), dtype=np.float64)
    probs[0] = p_normal
    probs[CLS.index(top_defect)] = p_defect
    rest = (1.0 - p_normal - p_defect) / (len(CLS) - 2)
    probs[probs == 0] = rest
    return np.log(probs).astype(np.float32)


@pytest.fixture
def mask() -> BeanMask:
    return bean_mask_top_017()


# ---------------------------------------------------------------------------
# τ 缺省 / 协议契约
# ---------------------------------------------------------------------------


def test_default_tau_is_sweep_recommendation():
    """τ 缺省 = 0.13（批10 扫描推荐工作点：检出 56.36% 过 55% 门 / normal
    66.47% 距 70% 门 3.53pt；两门实测不可同时满足）。"""
    assert DEFAULT_TAU == 0.13
    assert NnOnnxClassifier(session=MockSession(_logits_for(0.9, "broken", 0.05))).tau == 0.13


def test_protocol_output_contract(mask):
    """冻结 ClsModel 契约：(defect:str, conf:float, rank:int)，取值合法。"""
    sess = MockSession(_logits_for(0.10, "black", 0.80))
    cls = NnOnnxClassifier(session=sess)
    out = cls.classify(_rgba(mask), mask)
    assert isinstance(out, tuple) and len(out) == 3
    defect, conf, rank = out
    assert defect == "black"
    assert isinstance(defect, str) and TAX.is_valid_key(defect)
    assert isinstance(conf, float) and 0.0 <= conf <= 1.0
    assert isinstance(rank, int) and rank == TAX.severity_rank("black")
    # 输入张量被消费且形状/类型为训练口径
    assert sess.last_input.shape == (1, 3, 224, 224)
    assert sess.last_input.dtype == np.float32


def _rgba(mask: BeanMask, size: int = 32, color=(200, 100, 50),
          alpha_mode: str = "full") -> np.ndarray:
    """造一个 RGBA crop（RGB 序 + alpha）；默认全 255 alpha（无白底区）。"""
    x0, y0, x1, y1 = mask.bbox_mm
    w = int(x1 - x0) + size
    img = np.zeros((w, w, 4), dtype=np.uint8)
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = color
    img[:, :, 3] = 255 if alpha_mode == "full" else 0
    return img


# ---------------------------------------------------------------------------
# τ 判决逻辑
# ---------------------------------------------------------------------------


def test_tau_decision_defect_and_normal(mask):
    """p_max_defect ≥ τ → defect=缺陷类 argmax；否则 normal。"""
    cls = NnOnnxClassifier(session=MockSession(_logits_for(0.10, "sour", 0.80)))
    assert cls.classify(_rgba(mask), mask)[0] == "sour"
    cls = NnOnnxClassifier(session=MockSession(_logits_for(0.90, "sour", 0.05)))
    assert cls.classify(_rgba(mask), mask)[0] == "normal"


def test_tau_decision_uses_max_defect_not_argmax_overall(mask):
    """argmax=normal 但某缺陷类概率 ≥ τ 时仍判 defect——τ 口径不看全体
    argmax（与批10 扫描 ``p_max_defect >= tau`` 同式）。"""
    # normal=0.45，sour=0.50：全体 argmax=sour 且过门；再造 normal>缺陷argmax 的
    # 情形（normal 0.55 / black 0.40 / τ=0.3 → 判 defect black）
    cls = NnOnnxClassifier(tau=0.3, session=MockSession(_logits_for(0.55, "black", 0.40)))
    out = cls.classify_proba(_rgba(mask), mask)
    assert out["label"] == "black" and out["p_normal"] > out["p_max_defect"]


def test_tau_boundary_is_ge_semantics(mask):
    """判决边界 = ``>=``（与扫描脚本一致）：tau 恰取 p_max_defect → 仍 defect。"""
    probs = NnOnnxClassifier(
        session=MockSession(_logits_for(0.60, "broken", 0.30))
    ).classify_proba(_rgba(mask), mask)
    p = probs["p_max_defect"]
    at = NnOnnxClassifier(tau=p, session=MockSession(_logits_for(0.60, "broken", 0.30)))
    assert at.classify_proba(_rgba(mask), mask)["label"] == "broken"
    above = NnOnnxClassifier(
        tau=np.nextafter(p, 1.0), session=MockSession(_logits_for(0.60, "broken", 0.30))
    )
    assert above.classify_proba(_rgba(mask), mask)["label"] == "normal"


def test_conf_is_decided_class_probability(mask):
    """conf = 判定类别的 softmax 概率（defect=max 缺陷概率；normal=normal 类概率）。"""
    defect = NnOnnxClassifier(
        session=MockSession(_logits_for(0.10, "mold", 0.75))
    ).classify_proba(_rgba(mask), mask)
    assert defect["label"] == "mold"
    assert defect["conf"] == pytest.approx(defect["p_max_defect"])
    normal = NnOnnxClassifier(
        session=MockSession(_logits_for(0.95, "mold", 0.02))
    ).classify_proba(_rgba(mask), mask)
    assert normal["label"] == "normal"
    assert normal["conf"] == pytest.approx(normal["p_normal"])
    # 概率完备：softmax 和为 1
    assert sum(normal["probs"].values()) == pytest.approx(1.0, abs=1e-6)


def test_softmax_numerics_on_large_logits(mask):
    """大 logits 无上溢（数值稳定 softmax）。"""
    logits = np.full(len(CLS), 1000.0, dtype=np.float32)
    logits[0] = 1100.0
    out = NnOnnxClassifier(session=MockSession(logits)).classify(_rgba(mask), mask)
    assert out[0] == "normal" and out[1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 预处理数值（训练口径）
# ---------------------------------------------------------------------------


def test_preprocess_channel_order_and_normalization(mask):
    """RGB 序保持 + /255 + ImageNet 归一化：均匀 crop 的张量逐通道可复算。"""
    color = (200, 100, 50)  # R, G, B
    crop = _rgba(mask, color=color)
    sess = MockSession(_logits_for(0.9, "broken", 0.05))
    NnOnnxClassifier(session=sess).classify(crop, mask)
    t = sess.last_input
    for ch, v in enumerate(color):
        expect = (v / 255.0 - MEAN[ch]) / STD[ch]
        assert t[0, ch, 10, 10] == pytest.approx(expect, abs=1e-5)


def test_preprocess_white_composite_outside_alpha(mask):
    """alpha=0 处白底合成（训练侧「掩码外置白底」先验）→ 像素=(255-mean)/std。"""
    crop = _rgba(mask, alpha_mode="zero")  # alpha 全 0 → 整幅白底
    sess = MockSession(_logits_for(0.9, "broken", 0.05))
    NnOnnxClassifier(session=sess).classify(crop, mask)
    t = sess.last_input
    for ch in range(3):
        assert t[0, ch, 5, 5] == pytest.approx((1.0 - MEAN[ch]) / STD[ch], abs=1e-5)


def test_preprocess_mixed_alpha_binary_matches_hard_mask(mask):
    """二值 alpha（fillPoly 产物 0/255）合成与硬掩码逐位等价。"""
    crop = np.zeros((64, 64, 4), dtype=np.uint8)
    crop[:, :, :3] = (30, 160, 90)
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[16:48, 16:48] = 255  # 豆区
    crop[:, :, 3] = alpha
    sess = MockSession(_logits_for(0.9, "broken", 0.05))
    NnOnnxClassifier(session=sess).classify(crop, mask)
    t = sess.last_input
    # 64→224 上采样（×3.5）：输出坐标 (100,100)∈豆区、(5,5)∈外区
    inside_out = (30 / 255.0 - MEAN[0]) / STD[0]
    outside_out = (1.0 - MEAN[0]) / STD[0]
    assert t[0, 0, 100, 100] == pytest.approx(inside_out, abs=1e-5)  # 豆区=R 通道原值
    assert t[0, 0, 5, 5] == pytest.approx(outside_out, abs=1e-5)  # 外区=白底


def test_preprocess_resizes_any_input_to_224(mask):
    """非 224 输入 → INTER_AREA 缩放到 224²（均匀色下数值不变）。"""
    crop = np.zeros((448, 448, 4), dtype=np.uint8)
    crop[:, :, :3] = (200, 100, 50)
    crop[:, :, 3] = 255
    sess = MockSession(_logits_for(0.9, "broken", 0.05))
    NnOnnxClassifier(session=sess).classify(crop, mask)
    t = sess.last_input
    assert t.shape == (1, 3, 224, 224)
    assert t[0, 1, 112, 112] == pytest.approx((100 / 255.0 - MEAN[1]) / STD[1], abs=1e-5)


# ---------------------------------------------------------------------------
# 退化 / 错误面 / 惰性依赖
# ---------------------------------------------------------------------------


def test_empty_crop_degrades_to_normal_without_raise(mask):
    """空/非法 crop → (normal, 0.0, 0)（对齐 RulesV0 纪律：绝不阻塞管线）。"""
    cls = NnOnnxClassifier(session=MockSession(_logits_for(0.1, "black", 0.8)))
    for bad in (np.zeros((0, 5, 4), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)):
        assert cls.classify(bad, mask) == ("normal", 0.0, 0)
    assert cls.classify_proba(np.zeros((1, 1, 5), dtype=np.uint8), mask)["p_normal"] is None


def test_logit_width_mismatch_raises(mask):
    """logits 宽度 ≠ 类别表宽度 → NnOnnxError（结构性错误要响）。"""
    cls = NnOnnxClassifier(session=MockSession(np.zeros(10, dtype=np.float32)))
    with pytest.raises(NnOnnxError, match="宽度"):
        cls.classify(_rgba(mask), mask)


def test_missing_onnx_file_fails_loudly_on_first_classify(mask):
    """惰性加载：构造不读文件；首次 classify 缺文件 → NnOnnxError。"""
    cls = NnOnnxClassifier(onnx_path="Z:/definitely/absent/b9.onnx")
    with pytest.raises(NnOnnxError, match="不存在"):
        cls.classify(_rgba(mask), mask)


def test_invalid_tau_rejected():
    with pytest.raises(ValueError, match="tau"):
        NnOnnxClassifier(session=MockSession(np.zeros(13)), tau=1.5)


def test_classes_table_must_have_single_normal():
    with pytest.raises(NnOnnxError, match="normal"):
        NnOnnxClassifier(classes=["broken", "black"], session=MockSession(np.zeros(2)))


def test_onnxruntime_stays_lazy(mask):
    """注入会话路径全程不触发 onnxruntime 导入（惰性依赖纪律）。"""
    cls = NnOnnxClassifier(session=MockSession(_logits_for(0.9, "broken", 0.05)))
    cls.classify(_rgba(mask), mask)
    assert "onnxruntime" not in sys.modules


def test_custom_class_order_supported(mask):
    """显式类别表可覆盖缺省 taxonomy 序（logits 位次随注入表解释）。"""
    classes = ["normal", "black", "broken"] + [c for c in CLS if c not in ("normal", "black", "broken")]
    probs = np.full(len(classes), 0.1 / (len(classes) - 2))
    probs[0] = 0.1
    probs[classes.index("black")] = 0.8
    cls = NnOnnxClassifier(classes=classes, session=MockSession(np.log(probs)))
    assert cls.classify(_rgba(mask), mask)[0] == "black"


# ---------------------------------------------------------------------------
# realtime 开关装配 + 引擎
# ---------------------------------------------------------------------------


def test_build_classifier_switch():
    rules = build_classifier("rules")
    assert isinstance(rules, RulesV0)
    nn = build_classifier("nn", nn_onnx="whatever/b9.onnx")
    assert isinstance(nn, NnOnnxClassifier) and nn.tau == 0.13
    nn2 = build_classifier("nn", nn_onnx="whatever/b9.onnx", tau=0.2)
    assert nn2.tau == 0.2
    with pytest.raises(ClassifierUsageError, match="nn-onnx"):
        build_classifier("nn")
    with pytest.raises(ClassifierUsageError, match="未知分类器"):
        build_classifier("torch")


class _StubClassifier:
    """协议替身：恒判 black（验证引擎聚合，不依赖真模型）。"""

    stage = "classify"
    version = "stub:v0"
    tau = 0.13  # 引擎从实现读 τ 上 HUD

    def classify(self, crop_rgba, mask):
        return "black", 0.9, TAX.severity_rank("black")


def _synthetic_frame(dark: bool = True) -> np.ndarray:
    """白底 + 两粒深色椭圆（演示分割可检出）。"""
    frame = np.full((240, 320, 3), 245, dtype=np.uint8)
    if dark:
        cv2.ellipse(frame, (90, 120), (28, 18), 20, 0, 360, (35, 30, 28), -1)
        cv2.ellipse(frame, (230, 110), (26, 17), -35, 0, 360, (40, 36, 30), -1)
    return frame


def test_realtime_engine_aggregates_stub_classifier():
    engine = RealtimeEngine(_StubClassifier())
    result = engine.process_frame(_synthetic_frame())
    assert result.classifier == "stub:v0"
    assert result.tau == 0.13
    assert len(result.beans) >= 1
    assert result.counts == {"black": len(result.beans)}
    for b in result.beans:
        x, y, w, h = b.bbox_px
        assert 0 <= x and 0 <= y and w >= 1 and h >= 1
        assert x + w <= result.width and y + h <= result.height
        assert b.label == "black" and b.severity_rank == TAX.severity_rank("black")
        assert b.mask_id.startswith("top_")
    assert result.frame_ms > 0
    assert "cls=stub:v0" in result.hud_line() and "tau=0.13" in result.hud_line()


def test_realtime_engine_empty_frame_zero_detection():
    engine = RealtimeEngine(_StubClassifier())
    result = engine.process_frame(_synthetic_frame(dark=False))
    assert result.beans == [] and result.counts == {}


def test_realtime_engine_with_real_rules_classifier():
    """rules 模式端到端（真 RulesV0）：标签全部落在 taxonomy 合法键内。"""
    engine = RealtimeEngine(build_classifier("rules"))
    result = engine.process_frame(_synthetic_frame())
    assert result.classifier == RulesV0.version
    assert result.tau is None  # rules 无 τ
    for b in result.beans:
        assert TAX.is_valid_key(b.label)


def test_engine_classifier_versions_differ():
    """开关两态在 HUD 自述上可区分（rules vs nn；nn 用注入会话保持离线）。"""
    rules_engine = RealtimeEngine(build_classifier("rules"))
    nn = NnOnnxClassifier(session=MockSession(_logits_for(0.9, "broken", 0.05)))
    nn_engine = RealtimeEngine(nn)
    assert rules_engine.process_frame(_synthetic_frame()).classifier != \
        nn_engine.process_frame(_synthetic_frame()).classifier
    assert nn_engine.process_frame(_synthetic_frame()).classifier == "nn_onnx:v1"


# ---------------------------------------------------------------------------
# demo CLI 冒烟（图片模式 + 用法错误面；全程无相机/无网络）
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_image(tmp_path: Path) -> Path:
    p = tmp_path / "frame.jpg"
    cv2.imwrite(str(p), _synthetic_frame())
    return p


def test_demo_cli_rules_image_mode(demo_image: Path, capsys):
    from scripts.demo_realtime import EXIT_OK, main

    code = main(["--classifier", "rules", "--source", str(demo_image), "--no-show"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert '"classifier": "rules_v0:v1"' in out
    assert '"tau": null' in out


def test_demo_cli_nn_requires_onnx_path(capsys):
    from scripts.demo_realtime import EXIT_USAGE, main

    assert main(["--classifier", "nn", "--no-show"]) == EXIT_USAGE
    assert "nn-onnx" in capsys.readouterr().err


def test_demo_cli_nn_missing_onnx_file_exits_cleanly(demo_image: Path, capsys):
    from scripts.demo_realtime import EXIT_SOURCE, main

    code = main(["--classifier", "nn", "--nn-onnx", "Z:/nope/b9.onnx",
                 "--source", str(demo_image), "--no-show"])
    assert code == EXIT_SOURCE
    assert "NN" in capsys.readouterr().err


def test_demo_cli_unknown_classifier(capsys):
    """argparse choices 拒绝未知分类器（SystemExit 2；装配层分支另见
    test_build_classifier_switch）。"""
    from scripts.demo_realtime import main

    with pytest.raises(SystemExit) as ei:
        main(["--classifier", "torch"])
    assert ei.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
