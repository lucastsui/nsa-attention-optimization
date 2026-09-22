"""Analytic CPU reference checks; no GPU required, and no copied kernel math."""
import unittest
import torch
from check_nsa import reference, make_indices


class ReferenceTests(unittest.TestCase):
    def test_uniform_queries_are_causal_prefix_means(self):
        q = torch.zeros(5, 2, 3)
        k = torch.randn(5, 1, 3)
        v = torch.arange(10).reshape(5, 1, 2).float()
        idx = make_indices([5], 1, 3, 1, block_size=2)
        actual = reference(q, k, v, idx, [5], list(range(5)), block_size=2)
        expected = torch.stack([v[:i+1].double().mean(0).expand(2, 2) for i in range(5)])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_excluded_blocks_and_sequence_isolation(self):
        q = torch.zeros(6, 2, 3)
        k = torch.zeros(6, 1, 3)
        v = torch.tensor([999., 999., 2., 4., 100., 200.]).view(6, 1, 1)
        idx = torch.tensor([[[0], [0], [1], [1], [0], [0]]], dtype=torch.int32)
        actual = reference(q, k, v, idx, [4, 2], [2, 3, 4, 5], block_size=2)
        expected = torch.tensor([2., 3., 100., 150.], dtype=torch.float64).view(4, 1, 1).expand(4, 2, 1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
