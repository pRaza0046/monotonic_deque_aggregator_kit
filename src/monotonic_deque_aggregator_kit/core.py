"""Sliding-window aggregator backed by a monotonic deque.

The central guarantee is O(1) amortized cost per pushed element for the
minimum, maximum, and arbitrary fold queries.  A monotonic deque stores
candidate extrema in the window so that the front is always the answer for
that direction.  A separate accumulator tracks the fold value because a
fold is not necessarily invertible and cannot be recovered from the deque
alone.

Design note: the fold is recomputed from scratch on every query rather than
being maintained incrementally.  This keeps the data structure simple and
correct for non-invertible folds, at the cost of O(window) per fold query.
Min and max remain O(1).  If O(1) fold queries are required, pass an
invertible fold and maintain it separately; that is deliberately not done
here to avoid surprising the caller with hidden assumptions.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class MonotonicDeque(Generic[T]):
    """A deque that maintains a monotonic order under a key function.

    Elements are appended to the back and removed from the front.  The
external contract is a FIFO queue; internally, elements that can never be
the minimum (or maximum) for any future window are discarded immediately.
    """

    __slots__ = ("_data", "_key", "_increasing")

    def __init__(self, key: Callable[[T], object], increasing: bool = True) -> None:
        """Create a monotonic deque.

        Args:
            key: Extracts the comparison value from an element.
            increasing: If True, the front is the minimum under ``key``.
                If False, the front is the maximum under ``key``.
        """
        self._data: Deque[T] = deque()
        self._key = key
        self._increasing = increasing

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def push(self, value: T) -> None:
        """Append ``value``, discarding elements it dominates."""
        key = self._key
        data = self._data
        if self._increasing:
            while data and key(data[-1]) >= key(value):
                data.pop()
        else:
            while data and key(data[-1]) <= key(value):
                data.pop()
        data.append(value)

    def popleft(self) -> T:
        """Remove and return the front element.

        The front is the current extremum.  Raises IndexError if empty.
        """
        if not self._data:
            raise IndexError("popleft from an empty MonotonicDeque")
        return self._data.popleft()

    def front(self) -> T:
        """Return the current extremum without removing it.

        Raises IndexError if empty.
        """
        if not self._data:
            raise IndexError("front of an empty MonotonicDeque")
        return self._data[0]

    def clear(self) -> None:
        self._data.clear()


class MonotonicDequeAggregator(Generic[T]):
    """Sliding-window aggregator computing min, max, and a fold.

    The window size is fixed at construction.  ``push`` adds an element and,
    once the window is full, evicts the oldest element.  ``min`` and ``max``
    are O(1); ``fold`` is O(window) because it re-applies the fold over the
    live window.  This trade-off is documented in the module docstring.
    """

    __slots__ = ("_window", "_min_deque", "_max_deque", "_fold", "_identity", "_buffer")

    def __init__(
        self,
        window: int,
        fold: Callable[[U, T], U],
        identity: U,
        key: Optional[Callable[[T], object]] = None,
    ) -> None:
        """Create an aggregator.

        Args:
            window: Number of elements kept.  Must be positive.
            fold: Binary function ``(acc, item) -> acc`` applied left to right.
            identity: Initial accumulator for ``fold``.
            key: Optional key for min/max comparisons.  Defaults to identity.

        Raises:
            ValueError: If ``window`` is not positive.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window!r}")
        self._window = window
        self._fold = fold
        self._identity = identity
        cmp_key: Callable[[T], object] = key if key is not None else (lambda x: x)
        self._min_deque: MonotonicDeque[T] = MonotonicDeque(cmp_key, increasing=True)
        self._max_deque: MonotonicDeque[T] = MonotonicDeque(cmp_key, increasing=False)
        self._buffer: Deque[T] = deque()

    @property
    def window(self) -> int:
        return self._window

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, value: T) -> Optional[T]:
        """Add ``value`` to the window.

        If the window is already full, the oldest element is evicted and
        returned.  Otherwise returns None.
        """
        evicted: Optional[T] = None
        if len(self._buffer) == self._window:
            evicted = self._buffer.popleft()
            if self._min_deque and self._min_deque.front() is evicted:
                self._min_deque.popleft()
            if self._max_deque and self._max_deque.front() is evicted:
                self._max_deque.popleft()
        self._buffer.append(value)
        self._min_deque.push(value)
        self._max_deque.push(value)
        return evicted

    def extend(self, values: Iterable[T]) -> None:
        for value in values:
            self.push(value)

    def min(self) -> T:
        """Return the smallest element in the current window.

        Raises IndexError if the window is empty.
        """
        return self._min_deque.front()

    def max(self) -> T:
        """Return the largest element in the current window.

        Raises IndexError if the window is empty.
        """
        return self._max_deque.front()

    def fold(self) -> U:
        """Return the fold of the current window in insertion order.

        The accumulator starts at the identity supplied at construction.
        Raises IndexError if the window is empty.
        """
        if not self._buffer:
            raise IndexError("fold of an empty window")
        acc = self._identity
        for item in self._buffer:
            acc = self._fold(acc, item)
        return acc

    def __iter__(self) -> Iterator[T]:
        return iter(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._min_deque.clear()
        self._max_deque.clear()
