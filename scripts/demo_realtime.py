#!/usr/bin/env python3
"""实时演示 CLI（批10）：相机/图片源 + ``--classifier rules|nn`` 开关 + HUD。

用法（仓库根）::

    # 缺省 = rules（产品现状，批9 前行为不变）
    python scripts/demo_realtime.py --source 0

    # 批9 ONNX 分类头（τ 校准判决，τ 缺省 0.13 = 批10 扫描推荐工作点）
    python scripts/demo_realtime.py --classifier nn --nn-onnx train/runs/crop_cls/b9.onnx

    # 图片/目录源（无相机/无头环境可跑；--no-show 只打印 JSON 摘要）
    python scripts/demo_realtime.py --classifier nn --nn-onnx <onnx> \
        --source data/samples --no-show

开关语义
    ``rules`` → :class:`beaneye.classify.RulesV0`（缺省，不改变现状）；
    ``nn`` → :class:`beaneye.classify.NnOnnxClassifier`（批9 ONNX 头，
    需 ``--nn-onnx``；``--tau`` 覆盖缺省 0.13）。HUD 首行明注当前分类器
    （``cls=<version>``，nn 模式附 ``tau=…``）。

依赖
    rules 模式仅需主链路依赖（cv2/numpy/yaml/pydantic）；nn 模式额外
    ``pip install onnxruntime``（惰性导入，缺失在首帧显式报错）。
    阈值背景（如实）：批9 τ 扫描两门（检出 ≥55%、normal ≥70%）不可同时
    满足；τ=0.13 = 检出 56.36% 过门 / normal 66.47% 距门 3.53pt 的最接近点。

退出码：0 成功；2 用法错误（未知分类器 / nn 缺 --nn-onnx / 源不存在）；
3 源打开失败（相机占用/图片不可读）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beaneye.classify import NnOnnxError
from beaneye.realtime import CLASSIFIER_CHOICES, ClassifierUsageError, RealtimeEngine, build_classifier  # noqa: E402

EXIT_OK, EXIT_USAGE, EXIT_SOURCE = 0, 2, 3
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
_GREEN, _RED, _WHITE = (60, 180, 75), (50, 50, 230), (240, 240, 240)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="beaneye 实时演示（rules|nn 分类器开关 + HUD）")
    ap.add_argument("--classifier", choices=CLASSIFIER_CHOICES, default="rules",
                    help="分类器开关：rules=规则分类器（缺省）；nn=批9 ONNX 头")
    ap.add_argument("--nn-onnx", default=None,
                    help="nn 模式的 ONNX 路径（批9 头 train/runs/crop_cls/b9.onnx，不入库）")
    ap.add_argument("--tau", type=float, default=None,
                    help="nn 缺陷判决阈值（缺省 0.13 = 批10 扫描推荐工作点）")
    ap.add_argument("--source", default="0",
                    help="相机序号（缺省 '0'）或图片/目录路径（目录逐张处理）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="最多处理帧数（0=不限；相机模式到 ESC/q 或上限为止）")
    ap.add_argument("--every", type=int, default=1,
                    help="每 N 帧处理一帧（相机模式降负载用）")
    ap.add_argument("--no-show", action="store_true",
                    help="不弹窗（无头环境）；逐帧打印 JSON 摘要")
    ap.add_argument("--min-area-px", type=int, default=150,
                    help="演示分割最小豆面积（像素，缺省 150）")
    return ap


# ---------------------------------------------------------------------------
# HUD 叠加（ASCII：cv2.putText 不带中文）
# ---------------------------------------------------------------------------


def draw_hud(frame, result) -> None:
    """把 FrameResult 画到帧上：逐粒框+标签 + 顶部分类器/计数栏。"""
    for b in result.beans:
        x, y, w, h = b.bbox_px
        defect = b.label != "normal"
        color = _RED if defect else _GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{b.label} {b.conf:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    fps = 1000.0 / result.frame_ms if result.frame_ms > 0 else 0.0
    lines = [result.hud_line(), f"beans={len(result.beans)} fps={fps:.1f}"]
    if result.counts:
        lines.append(" ".join(f"{k}={v}" for k, v in sorted(result.counts.items())))
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, _WHITE, 1, cv2.LINE_AA)


def _iter_image_sources(source: Path) -> list[Path] | None:
    """图片文件或含图目录 → 文件列表；非图片源返回 None。"""
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(p for p in source.iterdir()
                       if p.suffix.lower() in _IMAGE_SUFFIXES)
        return files or None
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kwargs = {"tau": args.tau} if args.tau is not None else {}
        classifier = build_classifier(args.classifier, args.nn_onnx, **kwargs)
    except ClassifierUsageError as exc:
        print(f"[用法错误] {exc}", file=sys.stderr)
        return EXIT_USAGE

    def _run(frame):
        """单帧处理；NN 头结构性错误（文件缺失/非法/宽度不符）→ 干净退出。"""
        try:
            return engine.process_frame(frame)
        except NnOnnxError as exc:
            print(f"[错误] NN 分类头前向失败（--nn-onnx 路径/模型正确？）: {exc}",
                  file=sys.stderr)
            raise _Stop(EXIT_SOURCE) from exc

    engine = RealtimeEngine(classifier, min_area_px=args.min_area_px)
    source = Path(args.source)
    images = None if source.is_char_device() or args.source.strip().isdigit() \
        else _iter_image_sources(source)

    # ---- 相机模式 --------------------------------------------------------
    if images is None:
        if not args.source.strip().isdigit():
            print(f"[用法错误] 源不存在或没有可读图片: {source}", file=sys.stderr)
            return EXIT_USAGE
        cap = cv2.VideoCapture(int(args.source.strip()))
        if not cap.isOpened():
            print(f"[错误] 相机打开失败: {args.source}", file=sys.stderr)
            return EXIT_SOURCE
        n = 0
        try:
            while args.max_frames <= 0 or n < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % max(1, args.every) == 0:
                    result = _run(frame)
                    if not args.no_show:
                        draw_hud(frame, result)
                        cv2.imshow("beaneye realtime", frame)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                    else:
                        print(json.dumps(_summary(result), ensure_ascii=False), flush=True)
                n += 1
        finally:
            cap.release()
            if not args.no_show:
                cv2.destroyAllWindows()
        print(f"[完成] 处理 {n} 帧分类器={args.classifier}", flush=True)
        return EXIT_OK

    # ---- 图片/目录模式 ---------------------------------------------------
    if not images:
        print(f"[错误] 源中没有可读图片: {source}", file=sys.stderr)
        return EXIT_SOURCE
    try:
        for p in images:
            frame = cv2.imread(str(p))
            if frame is None:
                print(f"[警告] 读图失败，跳过: {p}", file=sys.stderr)
                continue
            result = _run(frame)
            if not args.no_show:
                draw_hud(frame, result)
                cv2.imshow("beaneye realtime", frame)
                if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                    break
            else:
                print(json.dumps(_summary(result, file=p.name), ensure_ascii=False), flush=True)
    except _Stop as stop:
        return stop.code
    if not args.no_show:
        cv2.destroyAllWindows()
    return EXIT_OK


class _Stop(Exception):
    """内部控制流：携带退出码跳出处理循环。"""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _summary(result, file: str | None = None) -> dict:
    out = {
        "classifier": result.classifier,
        "tau": result.tau,
        "beans": len(result.beans),
        "counts": result.counts,
        "frame_ms": round(result.frame_ms, 2),
        "size": [result.width, result.height],
        "labels": {b.mask_id: [b.label, round(b.conf, 4)] for b in result.beans},
    }
    if file is not None:
        out["file"] = file
    return out


if __name__ == "__main__":
    sys.exit(main())
