import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.evaluate_checkpoint_retrieval import cosine_mse, fit_probe
from scripts.evaluate_embedding_retrieval import retrieval_metrics
from trainers.affine_probe import RECALL_KS


class CheckpointRetrievalTests(unittest.TestCase):
    def test_cosine_only_gradient_has_no_mse(self):
        pred = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        target = torch.tensor([[2., 1.], [1., 3.]])
        loss, mse, cosine_loss = cosine_mse(pred, target, "cosine")
        torch.testing.assert_close(loss, cosine_loss)
        actual = torch.autograd.grad(loss, pred, retain_graph=True)[0]
        expected = torch.autograd.grad(
            1 - F.cosine_similarity(pred, target).mean(), pred)[0]
        torch.testing.assert_close(actual, expected)
        self.assertGreater(mse.item(), 0)

    def test_cosine_only_selection_ignores_mse_improvement(self):
        values = iter([3., 2., 1.])

        def step(optimizer):
            with torch.no_grad():
                weight, bias = optimizer.param_groups[0]["params"]
                weight.zero_()
                bias.copy_(torch.tensor([next(values), 0.]))

        with patch.object(torch.optim.Adam, "step", step):
            state, report = fit_probe(
                torch.zeros(6, 2), torch.tensor([[1., 0.]] * 6),
                {"train": [0, 1], "valid": [2, 3], "test": [4, 5]},
                loss_type="cosine", max_epochs=10, patience=2, log_every=0)
        self.assertEqual(report["best_epoch"], 1)
        self.assertEqual(report["epochs_trained"], 3)
        self.assertEqual(report["stop_reason"], "patience")
        self.assertEqual(report["loss_weights"], {"cosine": 1, "mse": 0})
        self.assertEqual(report["history"][-1]["valid_mse"], 0)
        torch.testing.assert_close(state["bias"], torch.tensor([3., 0.]))

    def test_loss_is_equal_weight_raw_mse_plus_cosine_loss(self):
        pred = torch.tensor([[1., 2.], [3., 4.]])
        target = torch.tensor([[2., 1.], [1., 3.]])
        loss, mse, cosine_loss = cosine_mse(pred, target)
        torch.testing.assert_close(mse, (pred - target).square().mean())
        torch.testing.assert_close(cosine_loss, 1 - F.cosine_similarity(pred, target).mean())
        torch.testing.assert_close(loss, mse + cosine_loss)

    def test_best_valid_restore_patience_and_test_never_selects(self):
        hidden = torch.zeros(6, 2, requires_grad=True)
        target = torch.tensor([[1., 0.]] * 4 + [[3., 0.]] * 2, requires_grad=True)
        splits = {"train": [0, 1], "valid": [2, 3], "test": [4, 5]}

        def run(y):
            values = iter([2., 1., 2., 3.])

            def step(optimizer):
                with torch.no_grad():
                    weight, bias = optimizer.param_groups[0]["params"]
                    weight.zero_()
                    bias.copy_(torch.tensor([next(values), 0.]))
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0)

            with patch.object(torch.optim.Adam, "step", step):
                return fit_probe(hidden, y, splits, max_epochs=10, patience=2, log_every=0)

        rng = torch.random.get_rng_state().clone()
        state, report = run(target)
        changed = target.detach().clone()
        changed[4:] = -100
        other, other_report = run(changed)
        self.assertEqual(report["best_epoch"], 2)
        self.assertEqual(report["epochs_trained"], 4)
        self.assertEqual(report["stop_reason"], "patience")
        self.assertEqual(report["best_valid_loss"], 0)
        self.assertEqual(report["history"], other_report["history"])
        torch.testing.assert_close(state["bias"], torch.tensor([1., 0.]))
        for key in state:
            torch.testing.assert_close(state[key], other[key])
        self.assertIsNone(hidden.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_adam_fit_max_budget_and_detached_weights(self):
        generator = torch.Generator().manual_seed(9)
        hidden = torch.randn(30, 4, generator=generator, requires_grad=True)
        target = hidden.detach() @ torch.randn(4, 6, generator=generator)
        splits = {"train": list(range(24)), "valid": [24, 25, 26], "test": [27, 28, 29]}
        state, report = fit_probe(hidden, target, splits, max_epochs=4, patience=10, log_every=0)
        self.assertEqual(report["stop_reason"], "max_epochs")
        self.assertEqual(report["epochs_trained"], 4)
        self.assertTrue(all(not value.requires_grad for value in state.values()))
        self.assertIsNone(hidden.grad)
        with torch.no_grad():
            restored_loss = cosine_mse(
                F.linear(hidden[splits["valid"]], **state), target[splits["valid"]])[0].item()
        self.assertAlmostEqual(restored_loss, report["best_valid_loss"])

    def test_retrieval_uses_full_catalogue_and_exact_tie_order(self):
        catalogue = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]])
        queries = catalogue[[1, 2]]
        recall, ranks = retrieval_metrics(queries, catalogue, torch.tensor([1, 2]), batch_size=1)
        self.assertEqual(ranks.tolist(), [2, 1])
        self.assertEqual(recall["Recall@1"]["value"], .5)
        self.assertEqual(recall["Recall@5"]["value"], 1.)

    def test_requested_recall_ks_and_item_id_ties_with_unsorted_catalogue(self):
        catalogue = torch.tensor([[1., 0.]] * 130)
        ids = torch.arange(130, 0, -1)
        rows = torch.tensor([0, 110, 120, 129])
        recall, ranks = retrieval_metrics(
            catalogue[rows], catalogue, rows, top_ks=RECALL_KS, item_ids=ids, batch_size=2,
        )
        self.assertEqual(ranks.tolist(), [130, 20, 10, 1])
        self.assertEqual(recall, {
            "Recall@10": {"hits": 2, "value": .5},
            "Recall@20": {"hits": 3, "value": .75},
            "Recall@50": {"hits": 3, "value": .75},
            "Recall@100": {"hits": 3, "value": .75},
        })
        permutation = torch.arange(129, -1, -1)
        other, other_ranks = retrieval_metrics(
            catalogue[rows], catalogue[permutation], 129 - rows,
            top_ks=RECALL_KS, item_ids=ids[permutation],
        )
        self.assertEqual(other, recall)
        self.assertEqual(other_ranks.tolist(), ranks.tolist())

    def test_retrieval_rejects_invalid_input(self):
        values = torch.ones(5, 2)
        for query_rows in ([5], [-1], []):
            with self.assertRaises(ValueError):
                retrieval_metrics(values[:1], values, query_rows)
        with self.assertRaises(ValueError):
            retrieval_metrics(values[:1], values, [0], item_ids=[1] * 5)
        with self.assertRaises(ValueError):
            retrieval_metrics(values[:1], values, [0], top_ks=(0,))

    def test_fit_disables_outer_autocast(self):
        x = torch.randn(10, 3, generator=torch.Generator().manual_seed(1))
        y = torch.randn(10, 4, generator=torch.Generator().manual_seed(2))
        splits = {"train": list(range(8)), "valid": [8], "test": [9]}
        state, report = fit_probe(x, y, splits, max_epochs=3)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            other, other_report = fit_probe(x, y, splits, max_epochs=3)
        self.assertEqual(report, other_report)
        for key in state:
            torch.testing.assert_close(state[key], other[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_fit_and_retrieval_preserve_rng(self):
        generator = torch.Generator().manual_seed(9)
        hidden = torch.randn(30, 4, generator=generator).cuda().requires_grad_()
        target = hidden.detach() @ torch.randn(4, 6, generator=generator).cuda()
        splits = {"train": list(range(24)), "valid": [24, 25, 26], "test": [27, 28, 29]}
        rng = torch.cuda.get_rng_state().clone()
        state, _ = fit_probe(hidden, target, splits, max_epochs=4, patience=10, log_every=0)
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        self.assertTrue(state["weight"].is_cuda)
        self.assertIsNone(hidden.grad)
        query = F.linear(hidden.detach()[splits["test"]], **state)
        gpu_metrics, gpu_ranks = retrieval_metrics(query, target, splits["test"])
        cpu_metrics, cpu_ranks = retrieval_metrics(query.cpu(), target.cpu(), splits["test"])
        self.assertEqual(gpu_metrics, cpu_metrics)
        self.assertEqual(gpu_ranks.tolist(), cpu_ranks.tolist())


if __name__ == "__main__":
    unittest.main()
