import unittest

from monotonic_deque_aggregator_kit import MonotonicDeque, MonotonicDequeAggregator


class MonotonicDequeTests(unittest.TestCase):
    def test_increasing_front_is_minimum(self) -> None:
        dq: MonotonicDeque[int] = MonotonicDeque(lambda x: x, increasing=True)
        dq.push(5)
        dq.push(1)
        dq.push(3)
        self.assertEqual(dq.front(), 1)
        self.assertEqual(len(dq), 2)

    def test_decreasing_front_is_maximum(self) -> None:
        dq: MonotonicDeque[int] = MonotonicDeque(lambda x: x, increasing=False)
        dq.push(5)
        dq.push(1)
        dq.push(3)
        self.assertEqual(dq.front(), 5)
        self.assertEqual(len(dq), 2)

    def test_popleft_removes_front(self) -> None:
        dq: MonotonicDeque[int] = MonotonicDeque(lambda x: x, increasing=True)
        dq.push(2)
        dq.push(1)
        dq.push(3)
        self.assertEqual(dq.popleft(), 1)
        self.assertEqual(dq.front(), 3)

    def test_popleft_on_empty_raises(self) -> None:
        dq: MonotonicDeque[int] = MonotonicDeque(lambda x: x, increasing=True)
        with self.assertRaises(IndexError):
            dq.popleft()

    def test_front_on_empty_raises(self) -> None:
        dq: MonotonicDeque[int] = MonotonicDeque(lambda x: x, increasing=True)
        with self.assertRaises(IndexError):
            dq.front()

    def test_key_function_used_for_comparison(self) -> None:
        dq: MonotonicDeque[tuple[int, str]] = MonotonicDeque(
            lambda item: item[0], increasing=True
        )
        dq.push((2, "b"))
        dq.push((1, "a"))
        self.assertEqual(dq.front(), (1, "a"))

    def test_clear_empties_deque(self) -> None:
        dq: MonotonicDeque[int] = MonotonicDeque(lambda x: x, increasing=True)
        dq.push(1)
        dq.clear()
        self.assertFalse(dq)
        self.assertEqual(len(dq), 0)


class MonotonicDequeAggregatorTests(unittest.TestCase):
    def test_min_and_max_on_sliding_window(self) -> None:
        agg: MonotonicDequeAggregator[int] = MonotonicDequeAggregator(
            window=3, fold=lambda acc, x: acc + x, identity=0
        )
        for value in (4, 2, 7):
            agg.push(value)
        self.assertEqual(agg.min(), 2)
        self.assertEqual(agg.max(), 7)
        agg.push(1)
        self.assertEqual(agg.min(), 1)
        self.assertEqual(agg.max(), 7)
        agg.push(9)
        self.assertEqual(agg.min(), 1)
        self.assertEqual(agg.max(), 9)

    def test_eviction_returns_oldest(self) -> None:
        agg: MonotonicDequeAggregator[int] = MonotonicDequeAggregator(
            window=2, fold=lambda acc, x: acc + x, identity=0
        )
        self.assertIsNone(agg.push(1))
        self.assertIsNone(agg.push(2))
        self.assertEqual(agg.push(3), 1)
        self.assertEqual(list(agg), [2, 3])

    def test_fold_sum(self) -> None:
        agg: MonotonicDequeAggregator[int] = MonotonicDequeAggregator(
            window=3, fold=lambda acc, x: acc + x, identity=0
        )
        for value in (1, 2, 3):
            agg.push(value)
        self.assertEqual(agg.fold(), 6)
        agg.push(4)
        self.assertEqual(agg.fold(), 9)

    def test_fold_with_non_commutative_operation(self) -> None:
        agg: MonotonicDequeAggregator[str] = MonotonicDequeAggregator(
            window=3, fold=lambda acc, x: acc + x, identity=""
        )
        for value in ("a", "b", "c"):
            agg.push(value)
        self.assertEqual(agg.fold(), "abc")
        agg.push("d")
        self.assertEqual(agg.fold(), "bcd")

    def test_key_function_for_min_max(self) -> None:
        agg: MonotonicDequeAggregator[tuple[int, str]] = MonotonicDequeAggregator(
            window=2,
            fold=lambda acc, x: acc + x[0],
            identity=0,
            key=lambda item: item[0],
        )
        agg.push((3, "c"))
        agg.push((1, "a"))
        self.assertEqual(agg.min(), (1, "a"))
        self.assertEqual(agg.max(), (3, "c"))

    def test_empty_window_queries_raise(self) -> None:
        agg: MonotonicDequeAggregator[int] = MonotonicDequeAggregator(
            window=3, fold=lambda acc, x: acc + x, identity=0
        )
        with self.assertRaises(IndexError):
            agg.min()
        with self.assertRaises(IndexError):
            agg.max()
        with self.assertRaises(IndexError):
            agg.fold()

    def test_window_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            MonotonicDequeAggregator(window=0, fold=lambda acc, x: acc + x, identity=0)

    def test_extend_and_clear(self) -> None:
        agg: MonotonicDequeAggregator[int] = MonotonicDequeAggregator(
            window=2, fold=lambda acc, x: acc + x, identity=0
        )
        agg.extend([1, 2, 3])
        self.assertEqual(list(agg), [2, 3])
        agg.clear()
        self.assertEqual(len(agg), 0)

    def test_duplicate_values_keep_window_length(self) -> None:
        agg: MonotonicDequeAggregator[int] = MonotonicDequeAggregator(
            window=3, fold=lambda acc, x: acc + x, identity=0
        )
        for value in (5, 5, 5, 5):
            agg.push(value)
        self.assertEqual(len(agg), 3)
        self.assertEqual(agg.min(), 5)
        self.assertEqual(agg.max(), 5)
        self.assertEqual(agg.fold(), 15)


if __name__ == "__main__":
    unittest.main()
