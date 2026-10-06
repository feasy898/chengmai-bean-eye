r"""掩码纯度测量 · ClassicSeg 掩码污染定量口径（候选 A 路 · 验收门工具）。

**问题**：e2e 静态链逐粒标签一致率 final≈0.0125——ClassicSeg（轮廓/阈值系
分割）的掩码含阴影/背景像素，分类头看到的 crop 被污染（批14 已证换分类头
无用，短板在掩码）。合成引擎知道每粒真值位置（它画的豆），因此掩码质量
可对合成盘**定量测量**——本脚本建立该测量口径，同时是掩码治理的验收门。

**口径**（对 --trays N --seed0 S 的每一盘、每一面）：

- 真值粒区域 = 合成 manifest 逐粒多边形（``poly_mm_top/poly_mm_bottom``）
  按正射网格分辨率（``configs/tray.yaml`` grid_px）栅格化——与
  ``beaneye.synth.compose.rasterize_poly_mm`` 同公式，RLE 真值同源；
- 检出掩码 = ClassicSeg 输出 ``BeanMask.polygon``（mm）栅格化到同一网格——
  分类 crop（``extract_mask_crop_rgba`` 的 alpha）正是按该多边形填充，
  栅格口径即分类头实际看到的像素集合；
- 逐粒配对 = 检出质心 ↔ 真值多边形质心匈牙利匹配（6mm 门限，与
  ``scripts/e2e_synth_run.py`` 的 ``MATCH_GATE_MM`` 同一口径）；
- **逐粒 IoU** = 匹配对上 检出栅格 ∩ 真值栅格 / 并集；报告均值/中位；
- **背景泄漏率** = 掩码内非豆像素占比：对每张检出掩码
  ``1 - |det ∩ 真值并集| / |det|``（真值并集 = 全盘所有真值粒，重叠摆放
  下豆间互不构成"背景"）；报告逐掩码均值与按像素池化的总体值。

合成配置与 ``scripts/e2e_synth_run.py`` **逐字段同源**（直接 import 其
``build_config``，缺省参数一致）——同 seed 下 purity 盘与 e2e 盘逐字节
同一张图，掩码纯度与 e2e 一致率可直接对账。

用法（仓库根）::

    python tools/measure_mask_purity.py --trays 1 --seed0 410001
    python tools/measure_mask_purity.py --trays 3 --seed0 410001 --json out/mask_purity/baseline.json
    python tools/measure_mask_purity.py --config configs/segment.yaml   # 显式配置档
    python tools/measure_mask_purity.py --no-refine                     # 治理前基线口径

全程离线 CPU；不写护照/不跑分类，单盘数秒。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beaneye.calibration import calibrate_pair, load_tray_config, warp_to_tray  # noqa: E402
from beaneye.segment import ClassicSeg  # noqa: E402
from beaneye.segment.config import SegmentConfig, load_segment_config  # noqa: E402
from beaneye.synth import ComposeConfig, compose_tray, sample_library  # noqa: E402
from beaneye.synth.compose import BeanPlacement, SynthTray  # noqa: E402

__all__ = [
    "MASK_ID_RE",
    "rasterize_poly",
    "truth_rasters",
    "det_rasters",
    "gate_match",
    "purity_for_side",
    "measure_tray",
    "main",
]

# 真值↔检出质心匹配门限（mm）：与 e2e_synth_run.MATCH_GATE_MM 同口径
# （配对门限 12mm 的一半；弦切豆翻面质心横移实测 ≤4.2mm，6mm 不误吸远邻）
GATE_MM_DEFAULT = 6.0


# ---------------------------------------------------------------------------
# 栅格化（mm 多边形 → 正射网格二值掩码）
# ---------------------------------------------------------------------------


def poly_centroid_mm(poly) -> tuple[float, float]:
    """多边形质心（cv2.moments，与 e2e_synth_run._poly_centroid 同公式）。"""
    m = cv2.moments(np.asarray(poly, dtype=np.float32))
    if m["m00"] > 0:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    pts = np.asarray(poly, dtype=np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def rasterize_poly(
    poly_mm: np.ndarray | list,
    mm_per_px: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """盘面 mm 多边形 → (shape) 网格二值掩码（uint8 {0,1}）。

    与 ``beaneye.synth.compose.rasterize_poly_mm`` 同一取整口径
    （round 到整像素后 ``cv2.fillPoly``），仅网格可为任意 (H, W)。
    """
    pts = np.round(np.asarray(poly_mm, dtype=np.float64) / float(mm_per_px)).astype(np.int32)
    mask = np.zeros((int(shape[0]), int(shape[1])), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def truth_rasters(
    beans: list[BeanPlacement],
    side: str,
    mm_per_px: float,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    """逐粒真值栅格（manifest poly_mm_top/bottom，豆完整轮廓含被压部分）。"""
    out = []
    for b in beans:
        poly = b.poly_mm_top if side == "top" else b.poly_mm_bottom
        out.append(rasterize_poly(poly, mm_per_px, shape))
    return out


def det_rasters(masks, mm_per_px: float, shape: tuple[int, int]) -> list[np.ndarray]:
    """检出 BeanMask.polygon → 栅格（= 分类 crop alpha 的像素集合）。"""
    return [rasterize_poly(m.polygon, mm_per_px, shape) for m in masks]


# ---------------------------------------------------------------------------
# 配对与指标
# ---------------------------------------------------------------------------


def gate_match(
    truth_xy: np.ndarray, obs_xy: np.ndarray, gate_mm: float
) -> list[tuple[int, int, float]]:
    """质心匈牙利匹配（门限外不成对）。返回 [(truth_i, obs_j, dist_mm), ...]。"""
    if len(truth_xy) == 0 or len(obs_xy) == 0:
        return []
    big = float(gate_mm) * 1000.0
    cost = np.linalg.norm(
        np.asarray(truth_xy, dtype=np.float64)[:, None, :]
        - np.asarray(obs_xy, dtype=np.float64)[None, :, :],
        axis=2,
    )
    cost[cost > gate_mm] = big
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c), float(cost[r, c]))
        for r, c in zip(rows, cols)
        if cost[r, c] < big
    ]


def purity_for_side(
    truth_r: list[np.ndarray],
    det_r: list[np.ndarray],
    truth_xy: np.ndarray,
    det_xy: np.ndarray,
    gate_mm: float = GATE_MM_DEFAULT,
) -> dict:
    """单面纯度指标：逐粒 IoU（匹配对）+ 背景泄漏率（全部检出掩码）。

    泄漏分母用**全部**检出掩码（分类头看到的就是全部 crop），匹配失败的
    掩码同样计入——错检/粘连片的污染不因配不上对而豁免。
    """
    pairs = gate_match(truth_xy, det_xy, gate_mm)
    ious: list[float] = []
    for ti, oj, _ in pairs:
        t, d = truth_r[ti], det_r[oj]
        union = int(np.logical_or(t > 0, d > 0).sum())
        inter = int(np.logical_and(t > 0, d > 0).sum())
        ious.append(inter / union if union else 0.0)

    # 真值并集（重叠摆放豆间相交面积不算背景）
    union_truth = np.zeros_like(truth_r[0]) if truth_r else None
    if union_truth is not None:
        for t in truth_r:
            union_truth |= t
    leakages: list[float] = []
    leak_px = kept_px = 0
    for d in det_r:
        area = int(d.sum())
        if area == 0:
            continue
        in_bean = int(np.logical_and(d > 0, union_truth > 0).sum())
        leakages.append(1.0 - in_bean / area)
        leak_px += area - in_bean
        kept_px += area
    return {
        "n_truth": len(truth_r),
        "n_det": len(det_r),
        "matched": len(pairs),
        "iou_mean": float(np.mean(ious)) if ious else None,
        "iou_median": float(np.median(ious)) if ious else None,
        "iou_min": float(np.min(ious)) if ious else None,
        "iou_p25": float(np.percentile(ious, 25)) if ious else None,
        "leakage_mean": float(np.mean(leakages)) if leakages else None,
        "leakage_pooled": (leak_px / kept_px) if kept_px else None,  # 按像素池化
        "mask_px_total": kept_px,
        "nonbean_px_total": leak_px,
        "ious": [round(v, 4) for v in ious],
    }


# ---------------------------------------------------------------------------
# 单盘测量
# ---------------------------------------------------------------------------


def measure_tray(
    tray: SynthTray,
    seg: ClassicSeg,
    *,
    tray_cfg=None,
    sides: tuple[str, ...] = ("top", "bottom"),
    gate_mm: float = GATE_MM_DEFAULT,
) -> dict:
    """对一盘合成盘（双面）跑分割并输出纯度指标（含分割耗时）。

    标定→warp→分割与 ``beaneye.app.pipeline.run_pipeline`` 同装配口径
    （mm_per_px = tray_mm / grid_px，RGB 输入，``segment_masks`` 逐面）。
    """
    tc = tray_cfg if tray_cfg is not None else load_tray_config()
    calib = calibrate_pair(tray.top_bgr, tray.bottom_bgr)
    mm_per_px = tc.tray_mm / tc.grid_px
    shape = (tc.grid_px, tc.grid_px)

    per_side: dict[str, dict] = {}
    seg_runtime_s = 0.0
    for side in sides:
        img = tray.top_bgr if side == "top" else tray.bottom_bgr
        warped = warp_to_tray(img, np.asarray(calib.H_top if side == "top" else calib.H_bottom,
                                              dtype=np.float64), tc)
        t0 = time.perf_counter()
        masks = seg.segment_masks(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), side=side)
        seg_runtime_s += time.perf_counter() - t0

        truth_r = truth_rasters(tray.beans, side, mm_per_px, shape)
        det_r = det_rasters(masks, mm_per_px, shape)
        truth_xy = np.asarray([poly_centroid_mm(b.poly_mm_top if side == "top" else b.poly_mm_bottom)
                               for b in tray.beans], dtype=np.float64)
        det_xy = (np.asarray([m.centroid_mm for m in masks], dtype=np.float64)
                  if masks else np.zeros((0, 2)))
        st = purity_for_side(truth_r, det_r, truth_xy, det_xy, gate_mm=gate_mm)
        st["mask_px_total"] = int(st["mask_px_total"])
        per_side[side] = st

    ious_all = [v for s in per_side.values() for v in s["ious"]]
    return {
        "scan_id": tray.scan_id,
        "seed": tray.seed,
        "n_truth_beans": len(tray.beans),
        "per_side": per_side,
        "iou_mean": float(np.mean(ious_all)) if ious_all else None,
        "iou_median": float(np.median(ious_all)) if ious_all else None,
        "leakage_mean": float(np.mean([s["leakage_mean"] for s in per_side.values()])),
        "leakage_pooled": (
            sum(s["nonbean_px_total"] for s in per_side.values())
            / max(1, sum(s["mask_px_total"] for s in per_side.values()))
        ),
        "seg_runtime_s": round(seg_runtime_s, 4),
    }


# ---------------------------------------------------------------------------
# e2e 同源合成配置（逐字段 import，杜绝口径漂移）
# ---------------------------------------------------------------------------


def e2e_compose_config(
    canvas: int = 2048,
    min_beans: int = 60,
    max_beans: int = 100,
    defect_rate: float = 0.06,
    contacts_min: int = 0,
    contacts_max: int = 0,
) -> ComposeConfig:
    """与 scripts/e2e_synth_run.py::build_config 逐字段同源的合成配置。"""
    spec = importlib.util.spec_from_file_location(
        "e2e_synth_run_for_purity", ROOT / "scripts" / "e2e_synth_run.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import argparse as _ap

    return mod.build_config(_ap.Namespace(
        canvas=canvas, min_beans=min_beans, max_beans=max_beans,
        defect_rate=defect_rate, contacts_min=contacts_min, contacts_max=contacts_max,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trays", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=410001,
                    help="首盘种子（与 e2e --seed0 同口径，缺省 410001）")
    ap.add_argument("--canvas", type=int, default=2048)
    ap.add_argument("--min-beans", type=int, default=60)
    ap.add_argument("--max-beans", type=int, default=100)
    ap.add_argument("--defect-rate", type=float, default=0.06)
    ap.add_argument("--contacts-min", type=int, default=0)
    ap.add_argument("--contacts-max", type=int, default=0)
    ap.add_argument("--gate-mm", type=float, default=GATE_MM_DEFAULT)
    ap.add_argument("--config", default=None,
                    help="分割配置 YAML（缺省 ClassicSeg 内建默认=产品现状）")
    ap.add_argument("--no-refine", action="store_true",
                    help="关闭颜色精修（治理前基线口径，等价 color_refine: false）")
    ap.add_argument("--side", choices=("top", "bottom", "both"), default="both")
    ap.add_argument("--json", default="out/mask_purity/purity.json",
                    help="指标 JSON 输出路径")
    args = ap.parse_args(argv)

    sides = ("top", "bottom") if args.side == "both" else (args.side,)
    cfg_seg: SegmentConfig = load_segment_config(args.config)
    if args.no_refine:
        cfg_seg = SegmentConfig(**{**cfg_seg.__dict__, "color_refine": False,
                                   "source": f"{cfg_seg.source}+no-refine"})
    seg = ClassicSeg(cfg=cfg_seg)

    cfg = e2e_compose_config(canvas=args.canvas, min_beans=args.min_beans,
                             max_beans=args.max_beans, defect_rate=args.defect_rate,
                             contacts_min=args.contacts_min, contacts_max=args.contacts_max)
    library = sample_library(cfg.library_seed, cfg.per_class)

    rows = []
    for k in range(args.trays):
        seed = args.seed0 + k
        tray = compose_tray(seed=seed, config=cfg, library=library)
        row = measure_tray(tray, seg, sides=sides, gate_mm=args.gate_mm)
        row["seg_config"] = cfg_seg.source
        rows.append(row)
        print(
            f"[{k + 1}/{args.trays}] {row['scan_id']} "
            f"IoU(mean/med)={row['iou_mean']:.4f}/{row['iou_median']:.4f} "
            f"leak(mean/pooled)={row['leakage_mean']:.4f}/{row['leakage_pooled']:.4f} "
            f"seg_t={row['seg_runtime_s']}s",
            flush=True,
        )

    ious_all = [v for r in rows for s in r["per_side"].values() for v in s["ious"]]
    summary = {
        "tool": "tools/measure_mask_purity.py",
        "seed0": args.seed0,
        "trays": args.trays,
        "gate_mm": args.gate_mm,
        "seg_config": cfg_seg.source,
        "color_refine": bool(cfg_seg.color_refine),
        "iou_mean": float(np.mean(ious_all)) if ious_all else None,
        "iou_median": float(np.median(ious_all)) if ious_all else None,
        "leakage_mean": float(np.mean([r["leakage_mean"] for r in rows])),
        "leakage_pooled": float(np.mean([r["leakage_pooled"] for r in rows])),
        "seg_runtime_s_mean": round(float(np.mean([r["seg_runtime_s"] for r in rows])), 4),
        "rows": rows,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"MASK PURITY: IoU mean={summary['iou_mean']:.4f} median={summary['iou_median']:.4f} | "
        f"leakage mean={summary['leakage_mean']:.4f} pooled={summary['leakage_pooled']:.4f} | "
        f"seg {summary['seg_runtime_s_mean']}s/盘 → {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
