# monotonic_deque_aggregator_kit

A sliding-window data structure that keeps a monotonic deque to answer running minimum and maximum in O(1), plus a fold over the live window.

```python
from monotonic_deque_aggregator_kit import MonotonicDequeAggregator

agg = MonotonicDequeAggregator(window=3, fold=lambda acc, x: acc + x, identity=0)
agg.push(4)
agg.push(2)
agg.push(7)

assert agg.min() == 2
assert agg.max() == 7
assert agg.fold() == 13

agg.push(1)  # evicts 4, window is [2, 7, 1]
assert agg.min() == 1
assert agg.max() == 7
assert agg.fold() == 10
```

The same module exports `MonotonicDeque`, the underlying structure, if you want to use it directly:

```python
from monotonic_deque_aggregator_kit import MonotonicDeque

dq = MonotonicDeque(lambda x: x, increasing=True)
dq.push(5)
dq.push(1)
dq.push(3)
assert dq.front() == 1
assert dq.popleft() == 1
assert dq.front() == 3
```

## Why this exists

Computing the minimum or maximum of a sliding window by scanning the window is O(window) per query.  A monotonic deque keeps only the elements that could still become the extremum, so the front is always the answer and eviction is amortized O(1).  This library packages that structure with a fixed-size window and a fold so callers do not have to reimplement the eviction bookkeeping.

The trade-off is the fold.  `min` and `max` are O(1), but `fold` re-applies the fold over the live window on every call, which is O(window).  Maintaining an incremental fold would be faster for invertible folds, but would be wrong for non-invertible folds such as string concatenation.  Rather than guess, the library always recomputes the fold so its result is correct for any binary function.  If you need O(1) fold queries, maintain your own accumulator alongside the aggregator.

## The awkward edge

Eviction uses identity comparison (`is`) to decide whether the departing element is the front of a monotonic deque.  For small integers and interned strings this is indistinguishable from equality, but for large integers, tuples, or custom objects, two equal values may be distinct objects.  The deque stores the exact object that was pushed, so the identity check is correct as long as callers push the same object they expect to evict.  If you push freshly constructed equal objects and rely on eviction to remove the right one, the monotonic deque may retain a stale candidate for one extra step.  The window contents themselves are always correct; only the internal candidate set can lag by the evicted element's lifetime.  This is the standard space-for-time trade-off of the structure and is documented here so the behaviour is not a surprise.
