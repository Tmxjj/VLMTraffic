"""
Author: Codex
Date: 2026-04-27
Description: Golden 数据样本过滤器

设计目标：
1. 原子化：每一种过滤规则都是独立的过滤器对象，方便单独测试与扩展。
2. 可组合：通过统一入口按顺序执行多个过滤器，做到无缝叠加。
3. 低侵入：主流程只负责构造 sample，并在写盘前调用过滤链。
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class FilterDecision:
    """
    单个过滤器的判定结果。

    字段说明：
    - passed: 是否通过当前过滤器
    - reason: 未通过时的原因，便于日志追踪
    - filter_name: 过滤器名称，便于定位是哪条规则拦截了样本
    - was_modified: 当前过滤器是否对样本做了原地清洗
    """

    passed: bool
    reason: Optional[str] = None
    filter_name: Optional[str] = None
    was_modified: bool = False


class FilterStatsTracker:
    """
    过滤统计器。

    统计目标：
    1. 每个过滤器实际过滤掉了多少条样本
    2. 每种过滤原因分别出现了多少次
    3. 有多少条样本是“被清洗后保留”的，而不是直接丢弃

    说明：
    - 这里只统计“过滤原因”的数量，即 dropped 的样本；
    - 清洗保留的样本单独统计在 `cleaned_count_by_filter` 中，避免和真正丢弃混淆。
    """

    def __init__(self):
        self.dropped_count_by_filter: Dict[str, int] = {}
        self.dropped_count_by_reason: Dict[str, int] = {}
        self.cleaned_count_by_filter: Dict[str, int] = {}

    def record(self, decision: Optional[FilterDecision]) -> None:
        """
        记录一次过滤器执行结果。
        """
        if not decision or not decision.filter_name:
            return

        if decision.was_modified:
            self.cleaned_count_by_filter[decision.filter_name] = (
                self.cleaned_count_by_filter.get(decision.filter_name, 0) + 1
            )

        if decision.passed:
            return

        self.dropped_count_by_filter[decision.filter_name] = (
            self.dropped_count_by_filter.get(decision.filter_name, 0) + 1
        )

        reason_key = decision.reason or "未知过滤原因"
        self.dropped_count_by_reason[reason_key] = (
            self.dropped_count_by_reason.get(reason_key, 0) + 1
        )

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """
        输出标准字典，便于日志打印、脚本返回或后续序列化。
        """
        return {
            "dropped_count_by_filter": dict(self.dropped_count_by_filter),
            "dropped_count_by_reason": dict(self.dropped_count_by_reason),
            "cleaned_count_by_filter": dict(self.cleaned_count_by_filter),
        }


class BaseSampleFilter:
    """
    样本过滤器基类。

    约定：
    - 输入为完整 sample 字典
    - 输出为 FilterDecision
    - passed=True 表示允许继续进入后续过滤器
    - passed=False 表示样本应被丢弃
    """

    filter_name = "base_filter"

    def apply(self, sample: Dict[str, Any]) -> FilterDecision:
        raise NotImplementedError


class ErrorResponseFilter(BaseSampleFilter):
    """
    过滤 `vlm_response` 为 ERROR 的样本。

    说明：
    - 这里按大小写不敏感处理，避免出现 `error` / `Error` 等变体漏检。
    - 仅检查标准化后的完整字符串是否等于 ERROR，不扩大到普通文本包含关系，
      以免误伤正常推理内容。
    """

    filter_name = "error_response_filter"

    def apply(self, sample: Dict[str, Any]) -> FilterDecision:
        vlm_response = sample.get("vlm_response")
        normalized_response = str(vlm_response or "").strip().upper()
        if normalized_response == "ERROR":
            return FilterDecision(
                passed=False,
                reason="vlm_response 为 ERROR",
                filter_name=self.filter_name,
            )
        return FilterDecision(passed=True, filter_name=self.filter_name)


class AdaptiveReasoningPathMarkerFilter(BaseSampleFilter):
    """
    处理 `Adaptive Reasoning` 推理结果中残留路径占位标签的样本。

    背景：
    - prompt 中的 `[Path 1]` / `[Path 2]` 只是指导模型选择推理分支的模板标记；
    - 如果模型把这些占位标记原样输出到 `vlm_response` 中，并不一定代表整条数据无效；
    - 如果整体输出结构完整，则只需要把这些占位标记删除即可继续保留。

    实现细节：
    - 同时兼容 `[Path1]`、`[Path 1]`、`[Path2]`、`[Path 2]`
    - 默认仅关注 `Adaptive Reasoning` 段及其后续内容，尽量避免误判 prompt 其他区域
    """

    filter_name = "adaptive_reasoning_path_marker_filter"

    _PATH_MARKER_PATTERN = re.compile(r"\[Path\s*[12]\]", re.IGNORECASE)
    _ADAPTIVE_REASONING_PATTERN = re.compile(r"Adaptive Reasoning\s*:", re.IGNORECASE)

    def apply(self, sample: Dict[str, Any]) -> FilterDecision:
        vlm_response = str(sample.get("vlm_response") or "")
        if not vlm_response:
            return FilterDecision(passed=True, filter_name=self.filter_name)

        adaptive_reasoning_text = self._extract_adaptive_reasoning_text(vlm_response)
        if self._PATH_MARKER_PATTERN.search(adaptive_reasoning_text):
            if self._is_complete_response(sample, vlm_response):
                sample["vlm_response"] = self._remove_path_markers(vlm_response)
                return FilterDecision(
                    passed=True,
                    reason="Adaptive Reasoning 中的路径占位标签已清洗",
                    filter_name=self.filter_name,
                    was_modified=True,
                )
            return FilterDecision(
                passed=False,
                reason="Adaptive Reasoning 中包含路径占位标签，且整体输出不完整",
                filter_name=self.filter_name,
            )
        return FilterDecision(passed=True, filter_name=self.filter_name)

    def _extract_adaptive_reasoning_text(self, vlm_response: str) -> str:
        """
        提取 `Adaptive Reasoning:` 段及其后续内容。

        这么做是为了把检查范围收敛到用户关心的区域。
        如果没有找到该标题，则退化为检查整个 `vlm_response`，
        保证对异常格式也仍然有兜底效果。
        """
        match = self._ADAPTIVE_REASONING_PATTERN.search(vlm_response)
        if not match:
            return vlm_response
        return vlm_response[match.end():]

    def _is_complete_response(self, sample: Dict[str, Any], vlm_response: str) -> bool:
        """
        判断当前 VLM 输出是否“整体完整”。

        这里采用偏保守的结构性判断：
        1. `vlm_response` 中同时包含 `Thought:` 与 `Action:`
        2. 样本里已经存在可用的 `vlm_action.phase_id` 和 `vlm_action.duration`

        这样可以避免仅凭字符串出现 `[Path1]` 就把明显有效的数据误删。
        """
        if "Thought:" not in vlm_response or "Action:" not in vlm_response:
            return False

        vlm_action = sample.get("vlm_action")
        if not isinstance(vlm_action, dict):
            return False

        phase_id = vlm_action.get("phase_id")
        duration = vlm_action.get("duration")
        if phase_id is None or duration is None:
            return False

        return True

    def _remove_path_markers(self, vlm_response: str) -> str:
        """
        删除 `Adaptive Reasoning` 中残留的 `[Path1]/[Path2]` 占位标签。

        处理策略：
        - 仅删除标签本身，不重写其余文本；
        - 顺手清理标签后面紧跟的冒号和多余空格，避免留下别扭的残片；
        - 最后压缩因删除标签导致的多余空行。
        """
        cleaned_response = re.sub(
            r"\[Path\s*[12]\]\s*:?\s*",
            "",
            vlm_response,
            flags=re.IGNORECASE,
        )
        cleaned_response = re.sub(r"\n{3,}", "\n\n", cleaned_response)
        return cleaned_response.strip()


class SampleFilterChain:
    """
    过滤器链。

    用法：
    - 初始化时传入多个过滤器
    - 调用 `apply(sample)` 后按顺序执行
    - 一旦某个过滤器不通过，立即短路返回，避免无意义的后续检查
    """

    def __init__(self, filters: Iterable[BaseSampleFilter]):
        self.filters: List[BaseSampleFilter] = list(filters)
        self.stats_tracker = FilterStatsTracker()

    def apply(self, sample: Dict[str, Any]) -> Tuple[bool, Optional[FilterDecision]]:
        for sample_filter in self.filters:
            decision = sample_filter.apply(sample)
            self.stats_tracker.record(decision)
            if not decision.passed:
                return False, decision
        return True, None

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        获取当前过滤链累计统计。
        """
        return self.stats_tracker.to_dict()


def build_default_sample_filter_chain() -> SampleFilterChain:
    """
    构建默认过滤链。

    当前规则：
    1. 过滤 `vlm_response == ERROR`
    2. 过滤 `Adaptive Reasoning` 中残留 `[Path1]/[Path2]` 的样本

    后续扩展时，只需要继续在这里追加新的过滤器即可。
    """
    return SampleFilterChain(
        filters=[
            ErrorResponseFilter(),
            AdaptiveReasoningPathMarkerFilter(),
        ]
    )
