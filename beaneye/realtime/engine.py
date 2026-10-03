"""实时演示引擎：单帧 → 演示级分割 → 可开关分类器 → 逐粒判决 + HUD 明细。

分类器开关（批10 任务书步骤 2）
    ``build_classifier("rules")`` → :class:`RulesV0`（**缺省**，产品现状）；
    ``build_classifier("nn", nn_onnx=路径)`` → :class:`NnOnnxClassifier`
    （批9 ONNX 分类头，τ 校准判决，τ 缺省 0.13 = 批10 扫描推荐工作点）。
    两实现满足同一冻结 ``ClsModel`` Protocol，本引擎对二者无差别调用。

分割（演示级，如实定位）
    单帧灰度 Otsu + 开闭形态学 + 连通域外轮廓，坐标用 ``1 px = 1 mm``
    假坐标系装进 :class:`BeanMask`（无标定/无 mm 真值；不冒充 M4
    ClassicSeg 契约口径，仅让分类器吃到与主链路同构的 RGBA crop）。
    crop 复用主链路的 ``beaneye.app.pipeline.extract_mask_crop_rgba``
    （RGB 通道序 + alpha=掩码），保证 rules/nn 两种分类器吃到的输入
    与生产管线逐字节同构。

HUD 明细
    :class:`FrameResult` 携带当前分类器 ``version``、nn 模式的 τ、逐粒
    ``BeanVerdict``（label/conf/rank/框）与缺陷计数——demo 侧叠加到
    画面即「HUD 注明当前分类器」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from beaneye.app.pipeline import extract_mask_crop_rgba
from beaneye.classify import DEFAULT_TAU, NnOnnxClassifier, RulesV0
from beaneye.schemas import BeanMask
from beaneye.taxonomy import Taxonomy, load_taxonomy

__all__ = [
    "CLASSIFIER_CHOICES",
    "ClassifierUsageError",
    "RealtimeEngine",
    "FrameResult",
    "BeanVerdict",
    "build_classifier",
]

CLASSIFIER_CHOICES = ("rules", "nn")


class ClassifierUsageError(ValueError):
    """分类器开关用法错误（未知选项 / nn 未给 --nn-onnx 路径）。"""


def build_classifier(
    choice: str,
    nn_onnx: str | Path | None = None,
    *,
    tau: float = DEFAULT_TAU,
    taxonomy: Taxonomy | None = None,
) -> object:
    """分类器开关装配：``rules`` → RulesV0；``nn`` → NnOnnxClassifier(onnx, τ)。

    ``nn`` 而 ``nn_onnx`` 缺失 → :class:`ClassifierUsageError`（CLI 侧映射
    为用法错误 exit 码）。NN 会话惰性构建：此处不读文件，首次 classify 才
    加载（文件缺失/非法在首帧以 ``NnOnnxError`` 显式报错）。
    """
    tax = taxonomy if taxonomy is not None else load_taxonomy()
    if choice == "rules":
        return RulesV0(taxonomy=tax)
    if choice == "nn":
        if nn_onnx is None:
            raise ClassifierUsageError(
                "--classifier nn 需要 --nn-onnx <ONNX 路径>（批9 头："
                "train/runs/crop_cls/b9.onnx）"
            )
        return NnOnnxClassifier(onnx_path=nn_onnx, tau=tau, taxonomy=tax)
    raise ClassifierUsageError(
        f"未知分类器 {choice!r}；可选 {'|'.join(CLASSIFIER_CHOICES)}"
    )


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------


@dataclass
class BeanVerdict:
    """单粒判决（HUD 用）。"""

    mask_id: str
    label: str
    conf: float
    severity_rank: int
    bbox_px: tuple[int, int, int, int]  # x, y, w, h（帧像素坐标）


@dataclass
class FrameResult:
    """单帧处理结果（HUD 明细 + 计数）。"""

    classifier: str  # 当前分类器自述版本（rules_v0:v1 / nn_onnx:v1）
    tau: float | None  # nn 模式的 τ 判决阈值；rules 为 None
    beans: list[BeanVerdict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    frame_ms: float = 0.0
    width: int = 0
    height: int = 0

    def hud_line(self) -> str:
        """HUD 首行：明注当前分类器（+ nn 模式 τ）。"""
        line = f"cls={self.classifier}"
        if self.tau is not None:
            line += f" tau={self.tau:.2f}"
        return line


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class RealtimeEngine:
    """单帧处理：演示级分割 → 可开关分类器 → FrameResult。

    参数
    ----
    classifier:
        冻结 ``ClsModel`` 实现（RulesV0 / NnOnnxClassifier，或测试替身）。
    taxonomy:
        严重度表（缺省 ``load_taxonomy()``）。
    min_area_px / max_beans:
        演示分割护栏：最小豆面积（像素）与单帧最大检出数。
    """

    def __init__(
        self,
        classifier: object,
        *,
        taxonomy: Taxonomy | None = None,
        min_area_px: int = 150,
        max_beans: int = 500,
    ) -> None:
        self._cls = classifier
        self._tax = taxonomy if taxonomy is not None else load_taxonomy()
        self._min_area_px = int(min_area_px)
        self._max_beans = int(max_beans)
        self.tau = getattr(classifier, "tau", None)

    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """单帧全流程；分割 0 检出时返回空结果（分类器不被调用）。"""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        result = FrameResult(
            classifier=str(getattr(self._cls, "version", type(self._cls).__name__)),
            tau=float(self.tau) if self.tau is not None else None,
            width=int(w),
            height=int(h),
        )
        masks = self._segment(frame_bgr)
        for i, mask in enumerate(masks):
            crop_rgba = extract_mask_crop_rgba(frame_bgr, mask, mm_per_px=1.0)
            defect, conf, rank = self._cls.classify(crop_rgba, mask)
            x0, y0, x1, y1 = mask.bbox_mm
            result.beans.append(
                BeanVerdict(
                    mask_id=mask.mask_id,
                    label=defect,
                    conf=float(conf),
                    severity_rank=int(rank),
                    bbox_px=(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))),
                )
            )
        for b in result.beans:
            result.counts[b.label] = result.counts.get(b.label, 0) + 1
        result.frame_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # 内部：演示级分割（1 px = 1 mm 假坐标系）
    # ------------------------------------------------------------------

    def _segment(self, frame_bgr: np.ndarray) -> list[BeanMask]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks: list[BeanMask] = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(cnt))
            if area < self._min_area_px:
                break  # 面积降序，后面只会更小
            if len(masks) >= self._max_beans:
                break
            x, y, bw, bh = cv2.boundingRect(cnt)
            m = cv2.moments(cnt)
            if m["m00"] <= 0:
                continue
            poly = [[float(p[0][0]), float(p[0][1])] for p in cnt]
            masks.append(
                BeanMask(
                    mask_id=f"top_{len(masks):04d}",
                    side="top",
                    polygon=poly,
                    bbox_mm=(float(x), float(y), float(x + bw), float(y + bh)),
                    area_mm2=area,
                    centroid_mm=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
                    source="classic",
                    conf=0.5,  # 演示级分割置信度（无标定真值，低于 classic 单峰 0.9）
                )
            )
        return masks
