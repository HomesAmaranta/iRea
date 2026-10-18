import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.factory import build_sid_model
from trainers.general_trainer import compute_alignment_loss, configure_alignment, train
from trainers.item_probe import (
    CKA_DESCRIPTION, GENREC_CKA_DESCRIPTION, genrec_linear_cka,
    ItemLinearProbe,
    _center_cosine_features,
    cosine_gram_cka,
    extract_item_hidden,
    fit_linear_probe,
    load_reference,
    split_items,
)
from trainers.representation_geometry import item_geometry
from trainers.affine_probe import PROTOCOL, RECALL_KS, retrieval_metrics


CONFIGS = {
    "hstu": dict(embed_dim=8, num_heads=2, num_blocks=2, dropout=0.1),
    "sasrec": dict(embed_dim=8, num_heads=2, num_blocks=2, ffn_dim=16, dropout=0.1),
    "tiger": dict(d_model=8, d_ff=16, num_heads=2, d_kv=4,
                  num_layers=2, num_decoder_layers=2, dropout_rate=0.1),
}


class TinyDataset:
    def __init__(self, **kwargs):
        self.item_ids = np.arange(1, 21)
        self.sem_ids_list = [[i % 4, i // 4 % 4, i // 16 % 4] for i in range(20)]
        self.item_embeddings = F.normalize(
            torch.randn(20, 5, generator=torch.Generator().manual_seed(99)), dim=-1
        )
        self.samples = [
            SimpleNamespace(
                item_ids=torch.tensor(self.sem_ids_list[i]),
                target_ids=torch.tensor(self.sem_ids_list[i + 1]),
                align_target_emb=self.item_embeddings[i],
            ) for i in range(4)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def write_reference(directory, dataset):
    path = Path(directory) / "reference.parquet"
    pd.DataFrame({
        "ItemID": dataset.item_ids,
        "embedding": dataset.item_embeddings.tolist(),
    }).to_parquet(path, index=False)
    return path


def category_labels():
    return pd.DataFrame({
        "ItemID": np.arange(20, 0, -1),
        "category": ["even" if i % 2 == 0 else "odd" for i in range(20, 0, -1)],
        "status": ["included"] * 20,
    }), {"status": "ready"}


def paper_cka_reference(x, y, dtype=torch.float64):
    """Explicit hollow-Gram HSIC from sophicle/tokens; independent of production helpers."""
    def kernel(values):
        values = values.float()
        threshold = torch.quantile(values.abs().flatten(start_dim=1), 0.95, dim=1).mean()
        values = F.normalize(values.clamp(-threshold, threshold), p=2, dim=-1).to(dtype)
        return values @ values.T

    def hsic(k, l):
        n = len(k)
        k = k.clone().fill_diagonal_(0)
        l = l.clone().fill_diagonal_(0)
        return (
            (k * l.T).sum()
            + k.sum() * l.sum() / ((n - 1) * (n - 2))
            - 2 * (k @ l).sum() / (n - 2)
        ) / (n * (n - 3))

    k, l = kernel(x), kernel(y)
    return (hsic(k, l) / (torch.sqrt(hsic(k, k) * hsic(l, l)) + 1e-6)).item()


class ProbeMathTests(unittest.TestCase):
    def test_genrec_cka_matches_centered_raw_grams(self):
        generator = torch.Generator().manual_seed(123)
        for n, dx, dy in ((2, 3, 5), (17, 4, 7), (19, 128, 257)):
            x = torch.randn(n, dx, generator=generator)
            y = torch.randn(n, dy, generator=generator)
            centering = torch.eye(n) - torch.ones(n, n) / n
            k, l = centering @ (x @ x.T) @ centering, centering @ (y @ y.T) @ centering
            expected = ((k * l).sum() / (k.norm() * l.norm())).item()
            self.assertAlmostEqual(genrec_linear_cka(x, y), expected, delta=2e-6)
            self.assertAlmostEqual(genrec_linear_cka(x, x), 1., delta=2e-6)
            self.assertAlmostEqual(genrec_linear_cka(3 * x + 4, y), expected, delta=2e-6)

    def test_genrec_cka_uses_raw_norms_without_clipping_and_preserves_state(self):
        generator = torch.Generator().manual_seed(8)
        x = torch.randn(23, 7, generator=generator, requires_grad=True)
        y = torch.randn(23, 11, generator=generator, requires_grad=True)
        scaled = x.detach().clone()
        scaled[0] *= 10000
        original = scaled.clone()
        rng = torch.random.get_rng_state().clone()
        result = genrec_linear_cka(scaled, y)
        normalized = genrec_linear_cka(F.normalize(scaled, dim=-1), F.normalize(y, dim=-1))
        self.assertGreater(abs(result - normalized), .01)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            self.assertEqual(genrec_linear_cka(scaled.double(), y.double()), result)
        genrec_linear_cka(x, y)
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        torch.testing.assert_close(scaled, original, rtol=0, atol=0)

    def test_genrec_cka_undefined_and_invalid_shapes(self):
        x = torch.arange(12.).reshape(4, 3)
        for y in (torch.ones_like(x), torch.zeros_like(x),
                  torch.full_like(x, float("nan")), torch.full_like(x, float("inf"))):
            self.assertIsNone(genrec_linear_cka(x, y))
        for y in (torch.ones(3, 2), torch.ones(4, 0), torch.ones(4)):
            with self.assertRaises(ValueError):
                genrec_linear_cka(x, y)
        with self.assertRaises(ValueError):
            genrec_linear_cka(x[:1], x[:1])

    def test_split_stable_disjoint_exhaustive_and_rng_independent(self):
        ids = np.arange(100, 201)
        rng = torch.random.get_rng_state().clone()
        splits = split_items(ids, 42)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual([len(v) for v in splits.values()], [80, 10, 11])
        joined = torch.cat(list(splits.values()))
        self.assertEqual(len(joined.unique()), 101)
        reversed_ids = ids[::-1].copy()
        reversed_split = split_items(reversed_ids, 42)
        for key, indices in splits.items():
            np.testing.assert_array_equal(ids[indices], reversed_ids[reversed_split[key]])
        with self.assertRaises(ValueError):
            split_items([1] * 20, 42)

    def test_reference_join_and_missing_duplicate_ids(self):
        ds = TinyDataset()
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference(directory, ds)
            y = load_reference(path, ds.item_ids[::-1].copy())
            torch.testing.assert_close(y, ds.item_embeddings.flip(0).double())
            with self.assertRaises(ValueError):
                load_reference(path, ds.item_ids[:-1])
            with self.assertRaises(ValueError):
                load_reference(path, np.ones(20, dtype=int))

    def test_cka_matches_author_hollow_gram_hsic(self):
        generator = torch.Generator().manual_seed(4)
        for n, dx, dy in ((4, 3, 5), (31, 7, 11), (19, 128, 257)):
            with self.subTest(n=n, dx=dx, dy=dy):
                x = torch.randn(n, dx, generator=generator, dtype=torch.float64)
                y = torch.randn(n, dy, generator=generator, dtype=torch.float64)
                actual = cosine_gram_cka(x, y)
                self.assertAlmostEqual(actual, paper_cka_reference(x, y), places=12)
                self.assertAlmostEqual(actual, paper_cka_reference(x, y, torch.float32), delta=2e-6)
                self.assertAlmostEqual(cosine_gram_cka(x, x), paper_cka_reference(x, x), places=12)
                self.assertEqual(cosine_gram_cka(torch.ones_like(x), y), 0.)
                self.assertEqual(cosine_gram_cka(torch.zeros_like(x), y), 0.)

    def test_cka_clips_using_mean_per_item_quantile_without_mutating_input(self):
        x = torch.arange(1, 121, dtype=torch.float32).reshape(6, 20)
        x[0, -1] = 10000
        original = x.clone()
        threshold = torch.quantile(x.abs(), 0.95, dim=1).mean()
        self.assertNotEqual(threshold.item(), torch.quantile(x.abs(), 0.95).item())
        normalized = F.normalize(x.clamp(-threshold, threshold), dim=1).double()
        expected = normalized - normalized.mean(dim=0, keepdim=True)
        torch.testing.assert_close(_center_cosine_features(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_cka_matches_appendix_u_centering_and_preserves_negative_values(self):
        x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        y = x[[0, 2, 1, 3]]

        def u_center(k):
            n = len(k)
            k = k.clone().fill_diagonal_(0)
            return (
                k - k.sum(0)[None, :] / (n - 2) - k.sum(1)[:, None] / (n - 2)
                + k.sum() / ((n - 1) * (n - 2))
            ).fill_diagonal_(0)

        k, l = u_center(x.double() @ x.double().T), u_center(y.double() @ y.double().T)
        scale = len(x) * (len(x) - 3)
        expected = (k * l).sum() / (k.norm() * l.norm() + scale * 1e-6)
        actual = cosine_gram_cka(x, y)
        self.assertLess(actual, 0)
        self.assertAlmostEqual(actual, expected.item(), places=12)

    def test_cka_validates_sample_count_shape_and_finiteness(self):
        x = torch.ones(4, 2)
        for y in (torch.ones(3, 2), torch.ones(4, 0), torch.ones(4),
                  torch.full((4, 2), float("nan")), torch.full((4, 2), float("inf"))):
            with self.subTest(shape=y.shape), self.assertRaises(ValueError):
                cosine_gram_cka(x, y)
        with self.assertRaises(ValueError):
            cosine_gram_cka(x[:3], x[:3])

    def test_affine_recovery_bias_and_test_targets_never_used_for_fitting(self):
        gen = torch.Generator().manual_seed(17)
        x = torch.randn(100, 6, generator=gen, dtype=torch.float64)
        weight = torch.randn(5, 6, generator=gen, dtype=torch.float64)
        bias = torch.arange(1., 6., dtype=torch.float64)
        y = F.linear(x, weight, bias)
        splits = split_items(np.arange(100), 42)
        state, report = fit_linear_probe(x, y, splits)
        self.assertEqual(report["alpha"], 0.)
        torch.testing.assert_close(state["weight"], weight)
        torch.testing.assert_close(state["bias"], bias)
        self.assertLess(report["splits"]["test"]["mse"], 1e-20)
        self.assertAlmostEqual(report["splits"]["test"]["cosine"], 1.)
        altered = y.clone()
        altered[splits["test"]] += 1000
        other, other_report = fit_linear_probe(x, altered, splits)
        self.assertEqual(report["alpha"], other_report["alpha"])
        for key in state:
            torch.testing.assert_close(state[key], other[key], rtol=0, atol=0)
        self.assertGreater(other_report["splits"]["test"]["mse"], 1e5)

    def test_constant_features_use_bias_and_no_nonfinite_values(self):
        x = torch.ones(20, 8)
        y = TinyDataset().item_embeddings
        splits = split_items(np.arange(20), 4)
        state, report = fit_linear_probe(x, y, splits)
        torch.testing.assert_close(state["weight"], torch.zeros(5, 8, dtype=torch.float64))
        torch.testing.assert_close(state["bias"], y[splits["train"]].double().mean(0))
        json.dumps(report, allow_nan=False)


class ProbeIntegrationTests(unittest.TestCase):
    def test_encoder_mean_matches_three_tokens_and_preserves_state(self):
        ds = TinyDataset()
        for precision in ("no", "bf16"):
            model = build_sid_model("tiger", 4, 3, 2, CONFIGS["tiger"]).train()
            model.model.encoder.block[0].eval()
            modes = [module.training for module in model.modules()]
            before = copy.deepcopy(model.state_dict())
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            rng = torch.random.get_rng_state().clone()
            with patch.object(model.model.decoder, "forward", side_effect=AssertionError):
                hidden = extract_item_hidden(
                    model, ds.sem_ids_list, 7, precision, hidden_source="encoder_mean"
                )
            self.assertEqual(modes, [module.training for module in model.modules()])
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertFalse(hidden.requires_grad)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            for parameter in model.parameters():
                torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
            model.eval()
            tokens = torch.tensor(ds.sem_ids_list) + torch.tensor([1, 5, 9])
            with torch.no_grad(), torch.autocast(
                "cpu", dtype=torch.bfloat16, enabled=precision == "bf16"
            ):
                expected = model.model.encoder(
                    input_ids=tokens, attention_mask=torch.ones_like(tokens)
                )[0].float()
            torch.testing.assert_close(hidden, expected.mean(dim=1))
            self.assertFalse(torch.allclose(hidden, expected[:, -1]))
        with self.assertRaises(ValueError):
            extract_item_hidden(
                build_sid_model("hstu", 4, 3, 2, CONFIGS["hstu"]),
                ds.sem_ids_list, hidden_source="encoder_mean",
            )

    def test_encoder_probe_has_independent_artifacts_and_namespaced_metrics(self):
        ds = TinyDataset()
        model = build_sid_model("tiger", 4, 3, 2, CONFIGS["tiger"])
        with tempfile.TemporaryDirectory() as directory:
            reference = write_reference(directory, ds)
            with patch("trainers.item_probe.load_dataset_labels", return_value=category_labels()):
                last, encoder = [
                    ItemLinearProbe(
                        ds, reference, directory, Mock(), hidden_source=source, max_epochs=3
                    ) for source in ("last", "encoder_mean")
                ]
            original = last.evaluate(model, 0)
            saved = (last.directory / "latest_linear.pt").read_bytes()
            metrics = encoder.evaluate(model, 0)
            self.assertTrue(set(original).isdisjoint(metrics))
            self.assertTrue(all(key.startswith("encoder_mean/") for key in metrics))
            self.assertEqual(saved, (last.directory / "latest_linear.pt").read_bytes())
            for split in last.splits:
                torch.testing.assert_close(last.splits[split], encoder.splits[split])
            self.assertEqual(encoder.directory.parent.name, "item_probe_encoder_mean")
            self.assertTrue((encoder.directory / "latest_category_linear.pt").exists())
            self.assertEqual(encoder.latest["hidden_source"], "encoder_mean")
            hidden = extract_item_hidden(model, ds.sem_ids_list, hidden_source="encoder_mean")
            self.assertEqual(encoder.latest["geometry"], item_geometry(hidden))
            self.assertEqual(encoder.latest["cka_genrec"], genrec_linear_cka(hidden, encoder.target))
            self.assertAlmostEqual(encoder.latest["cka"],
                                   paper_cka_reference(hidden, encoder.target), places=10)
            for k in RECALL_KS:
                self.assertIn(f"encoder_mean/linear_probe/test_Recall@{k}", metrics)

    def _check_hidden(self, name, precision="no", device="cpu"):
        torch.manual_seed(4)
        model = build_sid_model(name, 4, 3, 2, CONFIGS[name]).to(device).train()
        ds = TinyDataset()
        before = copy.deepcopy(model.state_dict())
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        cpu_rng = torch.random.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
        seen = []
        handle = model.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        hidden = extract_item_hidden(model, ds.sem_ids_list, 7, precision)
        handle.remove()
        self.assertEqual(hidden.shape, (20, 8))
        self.assertEqual(hidden.device.type, torch.device(device).type)
        self.assertFalse(hidden.requires_grad)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        if gpu_rng is not None:
            self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state(device)))
        for inp in seen:
            self.assertEqual(set(inp), {"input_ids", "attention_mask"})
            self.assertEqual(inp["input_ids"].shape[1], 3)
            self.assertTrue(inp["attention_mask"].eq(1).all())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for p in model.parameters():
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        if precision == "no":
            model.eval()
            ids = torch.tensor(ds.sem_ids_list[:7], device=device) + torch.tensor([1, 5, 9], device=device)
            with torch.no_grad():
                output = model({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
            if name == "tiger":
                self.assertEqual(output.last_hidden_state.shape[1], 1)
            torch.testing.assert_close(hidden[:7], output.last_hidden_state[:, -1])

    def test_normal_last_hidden_cpu_all_models_fp32_bf16(self):
        for name in CONFIGS:
            for precision in ("no", "bf16"):
                with self.subTest(model=name, precision=precision):
                    self._check_hidden(name, precision)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_normal_last_hidden_cuda_all_models(self):
        for name in CONFIGS:
            for precision in ("no", "bf16"):
                if precision == "bf16" and not torch.cuda.is_bf16_supported():
                    continue
                with self.subTest(model=name, precision=precision):
                    self._check_hidden(name, precision, "cuda")

    def test_probe_files_and_persistent_split_across_epochs(self):
        ds = TinyDataset()
        model = build_sid_model("tiger", 4, 3, 2, CONFIGS["tiger"])
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference(directory, ds)
            with patch("trainers.item_probe.load_dataset_labels", return_value=category_labels()):
                probe = ItemLinearProbe(ds, path, directory, Mock(), batch_size=6, max_epochs=5)
            before_rng = torch.random.get_rng_state().clone()
            first = probe.evaluate(model, 0)
            second = probe.evaluate(model, 1)
            self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
            self.assertEqual(first, second)
            records = [json.loads(line) for line in probe.path.read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [0, 1])
            config = json.loads((probe.directory / "config.json").read_text())
            self.assertEqual(config["cka"], CKA_DESCRIPTION)
            self.assertEqual(config["cka_genrec"], GENREC_CKA_DESCRIPTION)
            self.assertEqual(records[0]["cka_method"], CKA_DESCRIPTION)
            hidden = extract_item_hidden(model, ds.sem_ids_list, batch_size=6)
            self.assertAlmostEqual(first["linear_probe/cka"],
                                   paper_cka_reference(hidden, probe.target), places=10)
            self.assertEqual(first["linear_probe/cka_genrec"], genrec_linear_cka(hidden, probe.target))
            self.assertEqual(records[0]["cka_genrec"], first["linear_probe/cka_genrec"])
            self.assertEqual(records[0]["cka_genrec_method"], GENREC_CKA_DESCRIPTION)
            self.assertEqual(records[0]["cka_genrec_status"], "valid")
            geometry = item_geometry(hidden)
            self.assertEqual(records[0]["geometry"], geometry)
            for key in ("effective_rank", "avg_cosine"):
                self.assertAlmostEqual(first[f"geometry/item/{key}"], geometry[key], places=10)
            self.assertEqual([len(v) for v in config["split_item_ids"].values()], [16, 2, 2])
            checkpoint = torch.load(probe.directory / "latest_linear.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["protocol"], PROTOCOL)
            self.assertNotIn("alpha", checkpoint)
            self.assertEqual(checkpoint["state_dict"]["bias"].shape, (5,))
            self.assertFalse(any("probe" in k for k in model.state_dict()))
            self.assertIn("category_probe/test_acc", first)
            self.assertEqual(records[0]["category"]["status"], "valid")
            classifier = torch.load(probe.directory / "latest_category_linear.pt", weights_only=True)
            self.assertEqual(classifier["epoch"], 1)
            self.assertEqual(classifier["state_dict"]["weight"].shape, (2, 8))
            self.assertEqual(config["category_probe"]["split_item_ids"], config["split_item_ids"])
            test = probe.splits["test"]
            prediction = F.linear(hidden[test].float(), **checkpoint["state_dict"])
            expected_cos = F.cosine_similarity(prediction, probe.target[test].float()).mean().item()
            expected_mse = F.mse_loss(prediction, probe.target[test].float()).item()
            self.assertAlmostEqual(first["linear_probe/test_cosine"], expected_cos)
            self.assertAlmostEqual(first["linear_probe/test_mse"], expected_mse)
            recall, ranks = retrieval_metrics(
                prediction, probe.target, test, top_ks=RECALL_KS, item_ids=probe.item_ids,
            )
            self.assertEqual(records[0]["test"]["acc"], records[0]["category"]["splits"]["test"]["accuracy"])
            self.assertEqual(records[0]["loss_weights"], {"mse": 1, "cosine": 1})
            for k in RECALL_KS:
                self.assertEqual(first[f"linear_probe/test_Recall@{k}"], recall[f"Recall@{k}"]["value"])
                self.assertEqual(records[0]["test"][f"Recall@{k}"], recall[f"Recall@{k}"]["value"])
            saved_ranks = pd.read_csv(probe.directory / "latest_test_ranks.csv")
            self.assertEqual(saved_ranks["ItemID"].tolist(), ds.item_ids[test.numpy()].tolist())
            self.assertEqual(saved_ranks["rank"].tolist(), ranks.tolist())

    def test_category_disabled_does_not_read_metadata(self):
        ds = TinyDataset()
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference(directory, ds)
            with patch("trainers.item_probe.load_dataset_labels", side_effect=AssertionError):
                probe = ItemLinearProbe(ds, path, directory, Mock(), category_probe=False, max_epochs=5)
            model = build_sid_model("hstu", 4, 3, 2, CONFIGS["hstu"])
            report = probe.evaluate(model, 0)
            self.assertNotIn("category_probe/test_acc", report)
            self.assertEqual(probe.latest["category"]["status"], "disabled")
            self.assertIsNone(probe.latest["test"]["acc"])
            self.assertIn("linear_probe/test_Recall@100", report)

    def test_regression_training_does_not_affect_classifier(self):
        ds = TinyDataset()
        model = build_sid_model("hstu", 4, 3, 2, CONFIGS["hstu"])
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference(directory, ds)
            classifiers, regressors = [], []
            for lr in (1e-3, 0.5):
                with patch("trainers.item_probe.load_dataset_labels", return_value=category_labels()):
                    probe = ItemLinearProbe(
                        ds, path, directory, Mock(), learning_rate=lr, max_epochs=3,
                    )
                probe.evaluate(model, 0)
                classifiers.append(torch.load(
                    probe.directory / "latest_category_linear.pt", weights_only=True
                )["state_dict"])
                regressors.append(torch.load(
                    probe.directory / "latest_linear.pt", weights_only=True
                )["state_dict"])
            self.assertFalse(torch.equal(regressors[0]["weight"], regressors[1]["weight"]))
            for key in classifiers[0]:
                torch.testing.assert_close(classifiers[0][key], classifiers[1][key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_full_probe_stays_on_gpu_and_matches_cpu(self):
        ds = TinyDataset()
        model = build_sid_model("hstu", 4, 3, 2, CONFIGS["hstu"]).cuda().train()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference(directory, ds)
            with patch("trainers.item_probe.load_dataset_labels", return_value=category_labels()):
                probe = ItemLinearProbe(ds, path, directory, Mock(), max_epochs=5)
            rng = torch.cuda.get_rng_state().clone()
            probe.evaluate(model, 0, "bf16")
            self.assertTrue(probe.target.is_cuda)
            self.assertTrue(probe.centered_target.is_cuda)
            self.assertTrue(probe.target_cov_norm.is_cuda)
            self.assertEqual(probe.latest["device"], "cuda:0")
            self.assertEqual(probe.latest["category"]["device"], "cuda:0")
            self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
            self.assertTrue(model.training)
            for parameter in model.parameters():
                self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))
            hidden = extract_item_hidden(model, ds.sem_ids_list, mixed_precision="bf16")
            gpu_state, gpu = fit_linear_probe(hidden, probe.target, probe.splits)
            cpu_state, cpu = fit_linear_probe(hidden.cpu(), probe.target.cpu(), probe.splits)
            for key in gpu_state:
                self.assertTrue(gpu_state[key].is_cuda)
                torch.testing.assert_close(gpu_state[key].cpu(), cpu_state[key])
            self.assertEqual(gpu["alpha"], cpu["alpha"])
            self.assertAlmostEqual(gpu["splits"]["test"]["mse"],
                                   cpu["splits"]["test"]["mse"], places=10)
            self.assertAlmostEqual(cosine_gram_cka(hidden, probe.target),
                                   cosine_gram_cka(hidden.cpu(), probe.target.cpu()), delta=2e-6)

    def test_every_epoch_probe_does_not_change_training_or_early_stopping(self):
        for name in CONFIGS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                path = write_reference(directory, TinyDataset())
                runs = []
                for enabled in (False, True):
                    save_dir = Path(directory) / str(enabled)
                    save_dir.mkdir()
                    with patch("trainers.general_trainer.setup_logger", return_value=Mock()), \
                            patch("trainers.item_probe.load_dataset_labels", return_value=category_labels()):
                        result = train(
                            model_name=name, model_config=CONFIGS[name], dataset=TinyDataset,
                            save_dir_root=str(save_dir), linear_probe_reference_path=str(path),
                            epochs=2, batch_size=2, infer_batch_size=2, num_workers=0, device="cpu",
                            codebook_size=4, sem_id_dim=3, max_seq_len=2, beam_size=1, top_ks=(1,),
                            mse_loss_weight=10., align_loss_type="cosgram", linear_probe=enabled,
                            linear_probe_max_epochs=5, linear_probe_patience=2,
                            sequence_geometry=enabled, sequence_geometry_cosine_pairs=64,
                        )
                    checkpoint = next(save_dir.glob("best_model*.pt"))
                    runs.append((result, torch.load(checkpoint, weights_only=True)))
                    if enabled:
                        metrics = list((save_dir / "item_probe").glob("*/metrics.jsonl"))[0]
                        self.assertEqual(len(metrics.read_text().splitlines()), 2)
                        self.assertEqual(result["linear_probe_latest"]["epoch"], 1)
                        self.assertEqual(result["linear_probe_latest"]["category"]["status"], "valid")
                        if name == "tiger":
                            encoder_dir = Path(result["encoder_probe_dir"])
                            self.assertEqual(
                                len((encoder_dir / "metrics.jsonl").read_text().splitlines()), 2
                            )
                            self.assertEqual(result["encoder_probe_latest"]["epoch"], 1)
                            self.assertEqual(result["encoder_probe_latest"]["hidden_source"],
                                             "encoder_mean")
                        else:
                            self.assertIsNone(result["encoder_probe_dir"])
                        sequence_rows = [
                            json.loads(row) for row in
                            Path(result["sequence_geometry_path"]).read_text().splitlines()
                        ]
                        self.assertEqual([row["epoch"] for row in sequence_rows], [0, 1])
                        self.assertEqual(result["sequence_geometry_latest"], sequence_rows[-1])
                        for row in sequence_rows:
                            self.assertEqual(row["batches"], 2)
                            self.assertEqual(row["tokens"], 12)
                            self.assertEqual(row["sampled_pairs"], 128)
                            self.assertEqual(row["valid_batches"],
                                             dict(effective_rank=2, avg_cosine=2))
                    else:
                        self.assertFalse((save_dir / "item_probe").exists())
                        self.assertFalse((save_dir / "item_probe_encoder_mean").exists())
                        self.assertIsNone(result["encoder_probe_latest"])
                        self.assertIsNone(result["sequence_geometry_path"])
                        self.assertIsNone(result["sequence_geometry_latest"])
                self.assertEqual(runs[0][0]["best_valid_value"], runs[1][0]["best_valid_value"])
                self.assertEqual(runs[0][0]["test"], runs[1][0]["test"])
                for key in runs[0][1]:
                    torch.testing.assert_close(runs[0][1][key], runs[1][1][key], rtol=0, atol=0)

    def test_alignment_losses_unchanged(self):
        model = build_sid_model("hstu", 4, 3, 2, CONFIGS["hstu"])
        configure_alignment(model, 5, "cosgram")
        output = SimpleNamespace(last_hidden_state=torch.randn(3, 1, 8))
        inp = {"align_target_emb": torch.randn(3, 5)}
        cos = compute_alignment_loss(model, output, inp, "cos")
        gram = compute_alignment_loss(model, output, inp, "gram")
        joint = compute_alignment_loss(model, output, inp, "cosgram")
        torch.testing.assert_close(joint, 0.5 * cos + 0.5 * gram, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
