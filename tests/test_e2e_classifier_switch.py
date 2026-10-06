"""e2e 静态链分类头开关（``scripts/e2e_synth_run.py --classifier rules|nn``）
装配路径测试（批14）。

覆盖点（只测装配与解析口径，不跑全链——全链硬断言由脚本自验收承担）：

- 缺省 ``rules`` → ``build_components()`` 原样探测：RulesV0 + ClassicSeg、
  非降级（**产品缺省行为逐位不变**）；
- ``nn`` → 注入 :class:`beaneye.classify.NnOnnxClassifier`（实时侧
  ``build_classifier`` 同一装配口径），τ 显式传参与缺省 ``DEFAULT_TAU``
  两条路径、分割腿不受开关影响、装配盒非降级且版本自述 ``nn_onnx:v1``；
- ``--nn-onnx`` 缺省权重解析复用 ``beaneye.realtime.resolve_nn_onnx``：
  ``None`` = 自动探测仓内批13 胜者权重；**显式空串 = 原样透传** →
  ``ClassifierUsageError``（commit 4f3bdb7 契约：None 与空串严格区分，
  不自动探测、不静默回退 rules）。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from beaneye.classify import DEFAULT_TAU, NnOnnxClassifier, RulesV0
from beaneye.realtime import ClassifierUsageError, resolve_nn_onnx
from beaneye.segment import ClassicSeg
from scripts.e2e_synth_run import build_components_for

ROOT = Path(__file__).resolve().parents[1]

_NN_WEIGHT_CANDIDATES = (
    ROOT / "models" / "crop_cls.onnx",
    ROOT / "train" / "runs" / "crop_cls" / "b13_winner.onnx",
)


def _args(**kw) -> Namespace:
    """e2e 脚本开关三参数的最小 Namespace（缺省 = rules 档产品现状）。"""
    base = {"classifier": "rules", "nn_onnx": None, "tau": DEFAULT_TAU}
    base.update(kw)
    return Namespace(**base)


def test_e2e_default_rules_probe_unchanged():
    """缺省档：build_components() 原样探测——RulesV0 + ClassicSeg、非降级。"""
    comps = build_components_for(_args())
    assert comps.degraded_stages() == []
    assert isinstance(comps.classify.impl, RulesV0)
    assert isinstance(comps.segment.impl, ClassicSeg)
    assert comps.classify.version == RulesV0.version


def test_e2e_nn_switch_injects_nn_onnx_head():
    """nn 档：同一冻结协议换入 NnOnnxClassifier；τ 显式传参生效；分割腿
    仍工厂探测 ClassicSeg 不受开关影响；装配盒非降级、版本 nn_onnx:v1。"""
    comps = build_components_for(
        _args(classifier="nn", nn_onnx="whatever/b9.onnx", tau=0.33)
    )
    assert comps.degraded_stages() == []
    assert isinstance(comps.classify.impl, NnOnnxClassifier)
    assert comps.classify.impl.tau == 0.33
    assert comps.classify.version == "nn_onnx:v1"
    assert isinstance(comps.segment.impl, ClassicSeg)


def test_e2e_nn_tau_default_is_deployed_working_point():
    """nn 档 τ 缺省 = DEFAULT_TAU（批13 部署工作点 0.1），与实时侧同口径。"""
    assert DEFAULT_TAU == 0.1
    comps = build_components_for(_args(classifier="nn", nn_onnx="whatever/b9.onnx"))
    assert comps.classify.impl.tau == DEFAULT_TAU


@pytest.mark.skipif(
    not any(p.is_file() for p in _NN_WEIGHT_CANDIDATES),
    reason="仓内无 NN 权重（*.onnx 不入库）——自动探测分支无从覆盖",
)
def test_e2e_nn_default_weight_auto_resolve(monkeypatch):
    """--nn-onnx 缺省（None）→ resolve_nn_onnx 自动探测仓内批13 胜者权重，
    e2e 装配按探测结果构造 NN 头（τ=缺省工作点）。cwd 锚到仓库根，与两个
    CLI「仓库根运行」约定一致。"""
    monkeypatch.chdir(ROOT)
    resolved = resolve_nn_onnx(None)
    assert resolved is not None and Path(resolved).is_file()
    comps = build_components_for(_args(classifier="nn", nn_onnx=resolved))
    assert isinstance(comps.classify.impl, NnOnnxClassifier)
    assert Path(comps.classify.impl.onnx_path) == Path(resolved)
    assert comps.classify.impl.tau == DEFAULT_TAU


def test_e2e_nn_empty_onnx_passthrough_fails_closed():
    """显式空串 ≠ 未提供：透传既有装配层用法错误契约（ClassifierUsageError），
    不自动探测、不静默回退 rules（与 None 分支严格区分，commit 4f3bdb7）。"""
    with pytest.raises(ClassifierUsageError, match="nn_onnx"):
        build_components_for(_args(classifier="nn", nn_onnx=""))
