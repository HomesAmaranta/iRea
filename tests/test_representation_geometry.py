import json
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from trainers.representation_geometry import (
    SequenceGeometryEpoch, effective_rank, item_geometry, sequence_batch_geometry,
)


class GeometryTests(unittest.TestCase):
    def test_rank_matches_squared_singular_values_not_singular_values(self):
        for shape in ((7, 19), (19, 7)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, generator=torch.Generator().manual_seed(4)).double()
                x[:, 0] *= 10
                singular = torch.linalg.svdvals(x - x.mean(0))
                p = singular.square() / singular.square().sum()
                expected = (-(p * p.log()).sum()).exp()
                self.assertAlmostEqual(effective_rank(x), float(expected), places=10)
                wrong_p = singular / singular.sum()
                wrong = (-(wrong_p * wrong_p.log()).sum()).exp()
                self.assertGreater(abs(effective_rank(x) - float(wrong)), 0.1)
                self.assertAlmostEqual(effective_rank(x * 7 + 31), float(expected), places=10)

    def test_rank_equal_energy_and_one_direction(self):
        self.assertAlmostEqual(effective_rank(torch.tensor([[1., 0.], [-1., 0.],
                                                         [0., 1.], [0., -1.]])), 2.)
        self.assertAlmostEqual(effective_rank(torch.tensor([[1., 0.], [-1., 0.],
                                                         [2., 0.], [-2., 0.]])), 1.)
        self.assertIsNone(effective_rank(torch.ones(6, 3)))
        self.assertIsNone(effective_rank(torch.zeros(6, 3)))
        self.assertIsNone(effective_rank(torch.ones(1, 3)))

    def test_item_cosine_matches_all_off_diagonal_pairs(self):
        x = torch.randn(11, 7, generator=torch.Generator().manual_seed(9)).double()
        unit = F.normalize(x, dim=-1)
        gram = unit @ unit.T
        expected = gram[~torch.eye(len(x), dtype=torch.bool)].mean()
        record = item_geometry(x)
        self.assertAlmostEqual(record["avg_cosine"], float(expected), places=12)
        # Positive per-item scaling must not change cosine; no clipping or centering.
        scaled = x * torch.arange(1, 12)[:, None]
        self.assertAlmostEqual(item_geometry(scaled)["avg_cosine"], float(expected), places=12)
        self.assertAlmostEqual(item_geometry(torch.ones(6, 3))["avg_cosine"], 1.)
        self.assertLess(item_geometry(torch.tensor([[1., 0.], [-1., 0.]]))["avg_cosine"], 0.)

    def test_zeros_and_tiny_norms_are_handled_explicitly(self):
        x = torch.tensor([[1., 0.], [-1., 0.], [0., 0.]], dtype=torch.float64) * 1e-20
        record = item_geometry(x)
        self.assertEqual(record["nonzero_items"], 2)
        self.assertAlmostEqual(record["avg_cosine"], -1.)
        self.assertAlmostEqual(record["effective_rank"], 1.)
        record = item_geometry(torch.zeros(3, 2))
        self.assertIsNone(record["effective_rank"])
        self.assertIsNone(record["avg_cosine"])
        json.dumps(record, allow_nan=False)

    def test_sequence_pairs_cross_sequences_and_ignore_padding(self):
        hidden = torch.tensor([[[1., 0.], [1., 0.], [float("nan"), 0.]],
                               [[-1., 0.], [-1., 0.], [-1., 0.]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        record = sequence_batch_geometry(
            hidden, mask, num_pairs=123, generator=torch.Generator().manual_seed(8)
        )
        self.assertEqual(record["tokens"], 5)
        self.assertEqual(record["sampled_pairs"], 123)
        self.assertAlmostEqual(record["avg_cosine"], -1.)
        self.assertAlmostEqual(record["effective_rank"], effective_rank(hidden[mask]))

    def test_sampling_is_uniform_over_cross_sequence_token_pairs(self):
        hidden = torch.tensor([[[1., 0.]] * 4, [[0., 1.]] * 4, [[-1., 0.]] * 4])
        mask = torch.tensor([[True] * 4, [True, False, False, False],
                             [True, True, False, False]])
        unit = F.normalize(hidden[mask], dim=-1)
        groups = torch.arange(3)[:, None].expand_as(mask)[mask]
        expected = (unit @ unit.T)[groups[:, None] != groups[None, :]].mean().item()
        record = sequence_batch_geometry(
            hidden, mask, num_pairs=100000, generator=torch.Generator().manual_seed(12)
        )
        self.assertAlmostEqual(record["avg_cosine"], expected, delta=0.01)

    def test_sequence_single_row_and_zero_vectors(self):
        hidden = torch.ones(1, 3, 2)
        record = sequence_batch_geometry(
            hidden, torch.ones(1, 3, dtype=torch.bool), num_pairs=64,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertIsNone(record["avg_cosine"])
        self.assertEqual(record["sampled_pairs"], 0)
        hidden = torch.tensor([[[1., 0.], [0., 0.]], [[-1., 0.], [0., 0.]]])
        record = sequence_batch_geometry(
            hidden, torch.ones(2, 2, dtype=torch.bool), num_pairs=64,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(record["tokens"], 4)
        self.assertEqual(record["nonzero_tokens"], 2)
        self.assertAlmostEqual(record["avg_cosine"], -1.)

    def test_epoch_means_are_per_batch_not_token_weighted_or_pooled(self):
        accumulator = SequenceGeometryEpoch("cpu")
        records = [
            dict(effective_rank=2., avg_cosine=0.8, tokens=100, nonzero_tokens=100, sampled_pairs=64),
            dict(effective_rank=4., avg_cosine=-0.2, tokens=20, nonzero_tokens=20, sampled_pairs=64),
            dict(effective_rank=None, avg_cosine=None, tokens=1, nonzero_tokens=1, sampled_pairs=0),
        ]
        with patch("trainers.representation_geometry.sequence_batch_geometry", side_effect=records):
            for _ in records:
                accumulator.update(None, None)
        report = accumulator.result()
        self.assertEqual(report["effective_rank"], 3.)
        self.assertAlmostEqual(report["avg_cosine"], 0.3)
        self.assertEqual(report["batches"], 3)
        self.assertEqual(report["tokens"], 121)
        self.assertEqual(report["valid_batches"], dict(effective_rank=2, avg_cosine=2))
        self.assertEqual(report["skipped_batches"], dict(effective_rank=1, avg_cosine=1))
        json.dumps(report, allow_nan=False)

    def test_metrics_preserve_rng_inputs_and_gradients_cpu_and_cuda(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    hidden = torch.randn(4, 3, 7, device=device, dtype=dtype, requires_grad=True)
                    original = hidden.detach().clone()
                    rng = torch.random.get_rng_state().clone()
                    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
                    a = SequenceGeometryEpoch(device, num_pairs=128, seed=19)
                    b = SequenceGeometryEpoch(device, num_pairs=128, seed=19)
                    mask = torch.ones(4, 3, device=device, dtype=torch.bool)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        a.update(hidden, mask)
                        b.update(hidden, mask)
                        item_geometry(hidden[:, -1])
                    self.assertEqual(a.result(), b.result())
                    torch.testing.assert_close(original, hidden)
                    self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                    if cuda_rng is not None:
                        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
                    self.assertIsNone(hidden.grad)
                    hidden.float().square().sum().backward()
                    torch.testing.assert_close(hidden.grad, 2 * original)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            effective_rank(torch.ones(3, 2, 4))
        with self.assertRaises(ValueError):
            item_geometry(torch.full((3, 2), float("nan")))
        hidden = torch.ones(2, 3, 4)
        for mask in (torch.ones(2, 3), torch.ones(2, 4, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                sequence_batch_geometry(
                    hidden, mask, num_pairs=64, generator=torch.Generator()
                )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
