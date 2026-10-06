"""掩码治理（批15·A路）：纯度测量口径自测 + Lab 色域约束精修验收。

覆盖：

1. ``tools/measure_mask_purity.py`` 自测：栅格化/配对/指标函数对已知形状
   给出解析正确值；「检出=真值」自一致（IoU≈1、泄漏≈0）；合成配置与
   e2e 逐字段同源；
2. ClassicSeg 色域约束精修：合成真彩豆盘上 IoU/泄漏显著优于精修前
   （治理验收门：IoU 提升 ≥15pp 或泄漏 ≤基线 1/3，本仓两者同验）；
3. 中性灰前景守卫：灰度演示盘（无色彩信息）精修不改变行为（退回灰度口径）；
4. 单元级行为：色块+中性阴影贴图上阴影被剔除；
5. 配置：产品缺省=开、仓库 YAML 与缺省一致、非法配置拒绝。

运行（仓库根）::

    pytest tests/test_mask_purity.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _segment_synth import make_scene, place_sparse  # noqa: E402

from beaneye.segment import ClassicSeg  # noqa: E402
from beaneye.segment.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    SegmentConfig,
    SegmentConfigError,
    load_segment_config,
)
from beaneye.synth import ComposeConfig, compose_tray, sample_library  # noqa: E402


def _load_tool():
    """加载 tools/measure_mask_purity.py（验收门工具本体，非复制逻辑）。"""
    spec = importlib.util.spec_from_file_location(
        "measure_mask_purity_under_test", ROOT / "tools" / "measure_mask_purity.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


def _small_synth_tray(seed: int = 410042):
    """小型真彩合成盘（带投影阴影）：1024px 画布、6-8 粒，秒级可测。

    画布小于 warp 网格（2048）时 4× 上采样会把豆边界色度混进背景
    （插值光晕），残留泄漏高于真实 2048 盘——这是小画布物理，不是精修
    缺陷；1024 下治理门（IoU +15pp、泄漏 ≤1/3）仍可复验。
    """
    cfg = ComposeConfig(
        version=1, width_px=1024, height_px=1024, margin_mm=20.0,
        n_beans_min=6, n_beans_max=8, defect_rate=0.15,
        class_weights={"black": 3.0, "mold": 2.0, "sour": 2.0, "dried": 2.0,
                       "broken": 3.0, "faded": 1.0, "immature": 1.0},
        contact_min=0, contact_max=0, overlap_depth=0.8, free_factor=1.06,
        scale_jitter=0.10, angle_jitter=180.0, pair_jitter_mm=0.6,
        mirror_bottom=True, per_class=20, library_seed=20260929,
    )
    lib = sample_library(cfg.library_seed, cfg.per_class)
    return compose_tray(seed=seed, config=cfg, library=lib)


# ---------------------------------------------------------------------------
# 1) 纯度口径自测（tools/measure_mask_purity.py）
# ---------------------------------------------------------------------------


def test_rasterize_poly_known_shapes():
    """栅格化：[5,15]² 方形 @1px/mm → 121px（fillPoly 含边界）；平移 5px 的
    同形重叠 66px、并集 176px——与 fillPoly 语义逐像素对账。"""
    shape = (30, 30)
    a = tool.rasterize_poly([[5, 5], [15, 5], [15, 15], [5, 15]], 1.0, shape)
    b = tool.rasterize_poly([[10, 5], [20, 5], [20, 15], [10, 15]], 1.0, shape)
    assert int(a.sum()) == 121 and int(b.sum()) == 121
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    assert inter == 66 and union == 176


def test_purity_self_consistency_detection_equals_truth():
    """「检出=真值」自一致：IoU=1、泄漏=0（口径退化的锚点用例）。"""
    tray = _small_synth_tray()
    truth_r = tool.truth_rasters(tray.beans, "top", 300.0 / 2048.0, (2048, 2048))
    det_xy = np.asarray([tool.poly_centroid_mm(b.poly_mm_top) for b in tray.beans])
    st = tool.purity_for_side(truth_r, truth_r, det_xy, det_xy, gate_mm=6.0)
    assert st["matched"] == len(tray.beans)
    assert st["iou_mean"] >= 0.99
    assert st["leakage_pooled"] <= 0.01


def test_purity_metrics_known_leakage():
    """泄漏口径：检出=真值同心放大 → 泄漏率/IoU 解析可算（fillPoly 含边界）。"""
    shape = (40, 40)
    truth = [tool.rasterize_poly([[10, 10], [20, 10], [20, 20], [10, 20]], 1.0, shape)]  # 121
    det = [tool.rasterize_poly([[5, 5], [25, 5], [25, 25], [5, 25]], 1.0, shape)]  # 441
    xy = np.asarray([[15.0, 15.0]])
    st = tool.purity_for_side(truth, det, xy, xy, gate_mm=6.0)
    assert st["matched"] == 1
    assert abs(st["leakage_pooled"] - 320.0 / 441.0) < 0.01
    assert abs(st["ious"][0] - 121.0 / 441.0) < 0.01


def test_e2e_compose_config_parity():
    """纯度测量与 e2e 合成配置逐字段同源（同 seed ⇒ 同一张盘）。"""
    cfg = tool.e2e_compose_config()
    assert cfg.width_px == cfg.height_px == 2048
    assert (cfg.n_beans_min, cfg.n_beans_max) == (60, 100)
    assert cfg.contact_min == 0 and cfg.contact_max == 0
    assert cfg.defect_rate == 0.06 and cfg.library_seed == 20260929
    assert cfg.mirror_bottom is True and cfg.per_class == 20


# ---------------------------------------------------------------------------
# 2) 治理验收：色域约束精修在真彩合成盘上显著提纯
# ---------------------------------------------------------------------------


def test_color_refine_improves_purity_on_synth_tray():
    """同一真彩盘（经标定 warp 全口径）：精修开 vs 关。
    门：IoU 提升 ≥15pp 且池化泄漏 ≤基线 1/3（验收门双指标同验）。"""
    tray = _small_synth_tray()
    seg_off = ClassicSeg(cfg=SegmentConfig(color_refine=False, side="top"))
    seg_on = ClassicSeg(cfg=SegmentConfig(color_refine=True, side="top"))
    before = tool.measure_tray(tray, seg_off)
    after = tool.measure_tray(tray, seg_on)
    assert after["iou_mean"] - before["iou_mean"] >= 0.15, (
        f"IoU 提升不足: {before['iou_mean']} → {after['iou_mean']}"
    )
    assert after["leakage_pooled"] <= before["leakage_pooled"] / 3.0, (
        f"泄漏未降至 1/3: {before['leakage_pooled']} → {after['leakage_pooled']}"
    )


def test_color_refine_keeps_bean_count():
    """精修不丢真粒：真彩盘（warp 全口径）检出粒数 ≥ 精修前。"""
    tray = _small_synth_tray()
    seg_off = ClassicSeg(cfg=SegmentConfig(color_refine=False, side="top"))
    seg_on = ClassicSeg(cfg=SegmentConfig(color_refine=True, side="top"))
    for side in ("top", "bottom"):
        n_off = len(seg_off.segment_masks(
            cv2.cvtColor(tray.top_bgr if side == "top" else tray.bottom_bgr,
                         cv2.COLOR_BGR2RGB), side=side))
        n_on = len(seg_on.segment_masks(
            cv2.cvtColor(tray.top_bgr if side == "top" else tray.bottom_bgr,
                         cv2.COLOR_BGR2RGB), side=side))
        assert n_on >= n_off, f"{side}: 精修后检出 {n_on} < 精修前 {n_off}"


# ---------------------------------------------------------------------------
# 3) 适用性守卫：灰度前景不启用色域约束（行为退回纯灰度口径）
# ---------------------------------------------------------------------------


def test_gray_scene_guard_returns_baseline_masks():
    """灰度椭圆夹具盘：精修开/关输出掩码逐位一致（守卫生效）。"""
    scene = make_scene(place_sparse(12, seed=3), seed=11)
    seg_off = ClassicSeg(cfg=SegmentConfig(color_refine=False, side="top"))
    seg_on = ClassicSeg(cfg=SegmentConfig(color_refine=True, side="top"))
    rgb = cv2.cvtColor(scene.canvas, cv2.COLOR_RGB2BGR)
    m_off = seg_off.segment_masks(rgb, side="top")
    m_on = seg_on.segment_masks(rgb, side="top")
    assert len(m_off) == 12, f"灰度盘基线检出 {len(m_off)} != 12"
    assert len(m_on) == len(m_off)
    for a, b in zip(m_off, m_on):
        assert np.allclose(a.polygon, b.polygon)
        assert a.area_mm2 == pytest.approx(b.area_mm2)


# ---------------------------------------------------------------------------
# 4) 单元行为：色块豆 + 中性阴影 → 阴影被剔除
# ---------------------------------------------------------------------------


def _bean_with_shadow_canvas() -> np.ndarray:
    """浅背景 + 一个棕色圆豆 + 其右侧一条中性暗带（模拟投影阴影）。"""
    bg = np.full((160, 160, 3), (208, 203, 200), dtype=np.uint8)  # BGR 浅亚克力
    cv2.circle(bg, (70, 80), 24, (40, 70, 110), -1)  # 棕色豆（有色）
    shadow = np.full((160, 160, 3), (208, 203, 200), dtype=np.uint8)
    cv2.rectangle(shadow, (100, 55), (140, 105), (145, 142, 140), -1)  # 等比暗化=中性
    out = bg.astype(np.float32)
    band = np.zeros((160, 160), dtype=np.float32)
    band[55:105, 100:140] = 1.0
    out = out * (1 - band[..., None]) + shadow.astype(np.float32) * band[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def test_neutral_shadow_excluded_from_mask():
    """精修开：掩码不含中性暗带像素；精修关：暗带被收进前景（现状病理）。
    断言在画布像素系做（栅格化 mm→canvas px，与贴图同一坐标系）。"""
    bgr = _bean_with_shadow_canvas()
    seg_off = ClassicSeg(cfg=SegmentConfig(color_refine=False, side="top"))
    seg_on = ClassicSeg(cfg=SegmentConfig(color_refine=True, side="top"))
    masks_off = seg_off.segment_masks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), side="top")
    masks_on = seg_on.segment_masks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), side="top")
    assert masks_off and masks_on
    mm_per_px = 300.0 / bgr.shape[1]  # segment_masks 口径：图宽=tray_mm

    def shadow_px(masks) -> int:
        total = 0
        band = np.zeros(bgr.shape[:2], np.uint8)
        band[55:105, 100:140] = 1
        for m in masks:
            r = tool.rasterize_poly(m.polygon, mm_per_px, bgr.shape[:2])
            total += int(np.logical_and(r, band).sum())
        return total

    assert shadow_px(masks_off) > 0, "基线（关）应把中性暗带收进掩码（复现污染）"
    assert shadow_px(masks_on) == 0, "精修（开）应剔除全部中性暗带像素"


# ---------------------------------------------------------------------------
# 5) 配置：缺省、YAML 一致性、非法拒绝
# ---------------------------------------------------------------------------


def test_color_refine_defaults_on_and_yaml_consistent():
    cfg = load_segment_config(None)
    assert cfg.color_refine is True  # 产品缺省=治理开（e2e/实时同装配口径）
    assert cfg.color_refine_chroma_thr == 8.0
    assert cfg.color_refine_lum_thr == 90.0
    assert cfg.color_refine_min_chroma_margin == 3.0

    import yaml as _yaml

    raw = _yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    from_repo = load_segment_config(DEFAULT_CONFIG_PATH)
    assert raw["color_refine"] is True
    assert from_repo.color_refine == cfg.color_refine
    assert from_repo.color_refine_chroma_thr == cfg.color_refine_chroma_thr
    assert from_repo.color_refine_lum_thr == cfg.color_refine_lum_thr
    assert from_repo.color_refine_min_chroma_margin == cfg.color_refine_min_chroma_margin


@pytest.mark.parametrize(
    "override",
    [
        "color_refine: bogus\n",
        "color_refine_chroma_thr: -1\n",
        "color_refine_chroma_thr: 101\n",
        "color_refine_lum_thr: -5\n",
        "color_refine_lum_thr: 300\n",
        "color_refine_min_chroma_margin: -0.5\n",
        "color_refine_min_chroma_margin: 99\n",
    ],
)
def test_color_refine_config_rejects_invalid(override, tmp_path):
    base = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    p = tmp_path / "segment.yaml"
    p.write_text(base + override, encoding="utf-8")
    with pytest.raises(SegmentConfigError):
        load_segment_config(p)
