"""Check scientific transformations that could misrepresent the saved experiment."""
import unittest
import numpy as np

from experiments.paper_final.module05_figures.data import case_order, padded_trace, reduction, select_oracles


class ScientificTransformTests(unittest.TestCase):
    def test_terminated_trace_is_missing_not_a_zero_or_repeated_action(self):
        values = padded_trace([1., 3., 1.5], 6)
        np.testing.assert_array_equal(values[:3], [1., 3., 1.5])
        self.assertTrue(np.isnan(values[3:]).all())
        with self.assertRaises(ValueError):
            padded_trace([1, 2, 3], 2)

    def test_shared_order_uses_all_reference_difficulties_and_breaks_ties_by_case_id(self):
        cube = np.array([[[20, 10, 10], [10, 20, 10]], [[20, 10, 10], [10, 20, 10]]])
        np.testing.assert_array_equal(case_order(cube), [2, 0, 1])

    def test_cumulative_ratio_is_not_the_mean_of_case_percentages(self):
        reference = np.array([1., 9.]); rl = np.array([.5, 9.])
        self.assertAlmostEqual(float(reduction(reference.sum(), rl.sum())), 5.)
        self.assertAlmostEqual(float(reduction(reference, rl).mean()), 25.)
        self.assertLess(float(reduction(1., 1.2)), 0.)

    def test_global_and_per_case_hindsight_stay_distinct_and_exclude_fast_failures(self):
        cells = {}
        for name, values in {"a": [(1., True), (9., True)], "b": [(5., True), (2., True)],
                             "failed": [(.001, False), (.001, False)]}.items():
            for case, (cost, success) in enumerate(values):
                cells[1, "h", case, name] = {"native": cost, "success": success}
        best, individual = select_oracles(cells, 1, "h", [0, 1], ["a", "b", "failed"])
        self.assertEqual(best, "b")
        self.assertEqual(individual, {0: "a", 1: "b"})


if __name__ == "__main__":
    unittest.main()
