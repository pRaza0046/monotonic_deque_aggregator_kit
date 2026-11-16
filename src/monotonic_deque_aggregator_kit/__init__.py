"""Public API for monotonic_deque_aggregator_kit.

Exports the sliding-window aggregator and its companion monotonic deque.
"""

from .core import MonotonicDequeAggregator, MonotonicDeque

__all__ = ["MonotonicDequeAggregator", "MonotonicDeque"]
