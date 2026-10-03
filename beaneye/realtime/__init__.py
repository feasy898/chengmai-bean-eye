"""实时演示界面（批10 接入批9 ONNX 分类头时新建）。

本包在仓库中**此前不存在**（截至 2026-10-04 的 origin/main 无 realtime
模块；批10 任务书所称「批6 前产出」的 engine 未入库）——本件按批10 任务
新建最小实现：单帧/相机帧 → 演示级分割 → **可开关分类器**
（``--classifier rules|nn``，rules=RulesV0 缺省不改变现状，nn=批9 ONNX
头 ``NnOnnxClassifier``）→ 逐粒判决 + HUD 明细。

- :mod:`beaneye.realtime.engine` —— ``RealtimeEngine``（单帧处理 +
  分类器开关装配 ``build_classifier``）；
- ``scripts/demo_realtime.py`` —— CLI（相机/图片源 + HUD 叠加）。

定位纪律：本包分割为**演示级**（单帧 Otsu + 连通域，无标定/配对/计量），
不冒充 M4 ClassicSeg 契约口径；分类器与主链路同源（冻结 ClsModel
Protocol，两实现可互换），HUD 明注当前分类器与版本。
"""

from beaneye.realtime.engine import (
    CLASSIFIER_CHOICES,
    BeanVerdict,
    ClassifierUsageError,
    FrameResult,
    RealtimeEngine,
    build_classifier,
)

__all__ = [
    "RealtimeEngine",
    "FrameResult",
    "BeanVerdict",
    "ClassifierUsageError",
    "CLASSIFIER_CHOICES",
    "build_classifier",
]
