import unittest

from gpu_suite.ordering import (
    IMPLEMENTATION_PERMUTATIONS,
    assignment_counts,
    implementation_order,
    multiwave_assignment_counts,
    permutation_index,
    size_order_index,
)


class OrderingTests(unittest.TestCase):
    def test_all_permutations_and_independent_size_order(self):
        for wave in range(6):
            for node in range(12):
                self.assertEqual(permutation_index(node, wave), (node + wave) % 6)
                self.assertEqual(implementation_order(node, wave),
                                 IMPLEMENTATION_PERMUTATIONS[(node + wave) % 6])
                self.assertEqual(size_order_index(node), node % 2)

    def test_six_nodes_two_waves_cover_each_joint_assignment(self):
        counts = multiwave_assignment_counts(6, [0, 1])
        self.assertEqual(set(counts["permutation_size_order_counts"].values()), {1})
        self.assertEqual(set(counts["permutation_assignment_counts"].values()), {2})
        self.assertEqual(set(counts["size_order_assignment_counts"].values()), {6})

    def test_five_and_eight_node_imbalance_is_explicit(self):
        for nodes in (5, 8):
            counts = assignment_counts(nodes, 0)
            self.assertEqual(sum(counts["permutation_assignment_counts"].values()), nodes)
            self.assertEqual(sum(counts["size_order_assignment_counts"].values()), nodes)
            self.assertGreater(
                max(counts["permutation_assignment_counts"].values()),
                min(counts["permutation_assignment_counts"].values()),
            )

    def test_negative_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            permutation_index(-1, 0)
        with self.assertRaises(ValueError):
            size_order_index(-1)


if __name__ == "__main__":
    unittest.main()
