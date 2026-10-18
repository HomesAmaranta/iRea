import unittest
import gzip
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_category_probe import beauty_categories, metrics
from trainers.category_probe import (
    MISSING_SECOND_LEVEL, ItemCategoryProbe, fit_category_probe, load_dataset_labels,
    CategoryProbeNumericalError, PROBE_PROTOCOL,
)
from trainers.item_probe import split_items


class CategoryProbeTests(unittest.TestCase):
    def test_valid_ce_patience_resets_and_restores_best_not_last(self):
        values = iter([1., 2., 0., 1., 2.])

        def step(optimizer):
            with torch.no_grad():
                optimizer.param_groups[0]["params"][1][1] = next(values)

        hidden = torch.zeros(6, 2, dtype=torch.float64)
        labels = ["a", "b", "a", "a", "b", "b"]
        splits = {"train": [0, 1], "valid": [2, 3], "test": [4, 5]}
        with patch.object(torch.optim.Adam, "step", step):
            state, report = fit_category_probe(hidden, labels, splits, patience=2, max_epochs=10)
        self.assertEqual(report["best_epoch"], 3)
        self.assertEqual(report["epochs_trained"], 5)
        self.assertEqual(report["stop_reason"], "patience")
        self.assertAlmostEqual(report["best_valid_ce"], np.log(2))
        self.assertEqual(report["best_valid_ce"], min(h["valid_ce"] for h in report["history"]))
        torch.testing.assert_close(state["bias"], torch.zeros(2, dtype=torch.float64))
        # Test labels favor the last state, but they cannot influence selection.
        self.assertEqual(report["splits"]["test"]["accuracy"], 0)

    def test_equal_valid_ce_counts_as_no_improvement(self):
        with patch.object(torch.optim.Adam, "step", return_value=None):
            _, report = fit_category_probe(
                torch.zeros(6, 2), ["a", "b"] * 3,
                {"train": [0, 1], "valid": [2, 3], "test": [4, 5]},
                patience=3, max_epochs=10)
        self.assertEqual(report["best_epoch"], 1)
        self.assertEqual(report["epochs_trained"], 4)
        self.assertEqual(report["stop_reason"], "patience")

    def test_nonfinite_ce_reports_numerical_error(self):
        def step(optimizer):
            with torch.no_grad():
                optimizer.param_groups[0]["params"][1].fill_(float("nan"))
        with patch.object(torch.optim.Adam, "step", step):
            with self.assertRaises(CategoryProbeNumericalError) as caught:
                fit_category_probe(torch.zeros(6, 2), ["a", "b"] * 3,
                                   {"train": [0, 1], "valid": [2, 3], "test": [4, 5]})
        self.assertEqual(caught.exception.epoch, 1)

    def test_same_parent_from_multiple_paths(self):
        self.assertEqual(beauty_categories({"categories": [
            ["Beauty", "Skin Care", "Face", "Creams"], ["Beauty", "Skin Care", "Face"],
        ]}), ["Skin Care > Face"])

    def test_multiple_parents_remain_ambiguous(self):
        self.assertEqual(beauty_categories({"categories": [
            ["Beauty", "Skin Care", "Face"], ["Beauty", "Makeup", "Face"],
        ]}), ["Makeup > Face", "Skin Care > Face"])

    def test_missing_and_other_roots(self):
        for metadata in ({}, {"categories": None}, {"categories": [
                ["Beauty"], ["Other", "Skin Care"]]}):
            self.assertEqual(beauty_categories(metadata), [])

    def test_metrics_include_unpredicted_classes(self):
        result = metrics(["a", "a", "b"], ["a", "a", "a"], ["a", "b"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 0.4)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 0]])
        self.assertEqual(result["per_class"]["b"]["recall"], 0)

    def test_fit_is_affine_detached_and_ignores_test_data_for_selection(self):
        rng = np.random.default_rng(42)
        x = torch.tensor(rng.normal(size=(90, 5)), requires_grad=True)
        y = np.where(x.detach().numpy()[:, 0] > 0, "positive", "negative")
        splits = split_items(np.arange(90), 42)
        before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        state, report = fit_category_probe(x, y, splits)
        altered_x, altered_y = x.detach().clone(), y.copy()
        altered_x[splits["test"]] += 10000
        altered_y[splits["test"]] = "unseen_test_label"
        other, other_report = fit_category_probe(altered_x, altered_y, splits)
        self.assertEqual(report["best_epoch"], other_report["best_epoch"])
        self.assertEqual(report["history"], other_report["history"])
        for key in state:
            torch.testing.assert_close(state[key], other[key], rtol=0, atol=0)
        predicted = np.asarray(report["class_order"])[
            (x.detach() @ state["weight"].T + state["bias"]).argmax(1).numpy()
        ]
        test = splits["test"].numpy()
        self.assertAlmostEqual((predicted[test] == y[test]).mean(),
                               report["splits"]["test"]["accuracy"])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        after = np.random.get_state()
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertIsNone(x.grad)

    def test_raw_mapping_cache_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw" / "beauty"
            raw.mkdir(parents=True)
            reviews = raw / "reviews_Beauty_5.json.gz"
            meta = raw / "meta_Beauty.json.gz"
            with gzip.open(reviews, "wt") as handle:
                for asin in ("b", "a", "b", "c"):
                    handle.write(json.dumps({"asin": asin}) + "\n")
            def write_meta(category):
                with gzip.open(meta, "wt") as handle:
                    for asin, paths in (
                        ("a", [["Beauty", "Hair Care"]]),
                        ("b", [["Beauty", category, "Face"]]),
                        ("c", [["Beauty", "Hair Care", "Shampoos"], ["Beauty", "Makeup", "Face"]]),
                    ):
                        handle.write(repr({"asin": asin, "categories": paths}) + "\n")
            write_meta("Skin Care")
            ds = SimpleNamespace(root=temp, split="beauty")
            frame, metadata = load_dataset_labels(ds, [2, 1, 3])
            self.assertEqual(frame.set_index("ItemID").loc[1, "category"], "Skin Care > Face")
            self.assertEqual(frame.set_index("ItemID").loc[2, "category"], MISSING_SECOND_LEVEL)
            self.assertEqual(frame.set_index("ItemID").loc[2, "status"], "included")
            self.assertEqual(frame.set_index("ItemID").loc[3, "status"], "multiple_labels")
            with patch("trainers.category_probe.read_labels", side_effect=AssertionError):
                cached, _ = load_dataset_labels(ds, [3, 2, 1])
            self.assertEqual(frame.to_dict("records"), cached.to_dict("records"))
            write_meta("Makeup")
            changed, changed_meta = load_dataset_labels(ds, [1, 2, 3])
            self.assertEqual(changed.set_index("ItemID").loc[1, "category"], "Makeup > Face")
            self.assertNotEqual(metadata["fingerprint"], changed_meta["fingerprint"])

    def test_reject_invalid_training_settings(self):
        for kwargs in ({"learning_rate": 0}, {"learning_rate": float("nan")},
                       {"max_epochs": 0}, {"patience": -1}, {"patience": 1.5}):
            with self.assertRaises(ValueError):
                fit_category_probe(torch.ones(10, 2), ["a"] * 10, {}, **kwargs)

    def test_rare_and_missing_classes_remain_in_evaluation(self):
        ids = np.arange(100)
        splits = split_items(ids, 42)
        labels = np.array(["common_a" if i % 2 else "common_b" for i in ids], dtype=object)
        labels[splits["test"][:2]] = "rare_test_only"
        labels[splits["train"][:2]] = MISSING_SECOND_LEVEL
        frame = pd.DataFrame({"ItemID": ids, "category": labels, "status": "included"})
        with tempfile.TemporaryDirectory() as temp:
            probe = ItemCategoryProbe(frame, ids, splits, temp)
            for split in splits:
                np.testing.assert_array_equal(probe.splits[split], splits[split])
            result = probe.evaluate(
                torch.randn(100, 3, generator=torch.Generator().manual_seed(1)), 0)
            self.assertIn("rare_test_only", result["unseen_train_classes"])
            self.assertIn(MISSING_SECOND_LEVEL, result["class_order"])
            self.assertEqual(result["splits"]["test"]["per_class"]["rare_test_only"]["support"], 2)
            self.assertEqual(result["splits"]["test"]["per_class"]["rare_test_only"]["recall"], 0)

    def test_unweighted_ce_no_regularization_and_standardization_folds_into_affine(self):
        x = torch.randn(100, 4, generator=torch.Generator().manual_seed(7)).double()
        labels = np.array(["majority"] * 90 + ["minority"] * 10)
        splits = split_items(np.arange(100), 42)
        state, report = fit_category_probe(x, labels, splits, max_epochs=1)
        self.assertEqual(report["backend"], "torch_adam")
        self.assertEqual(report["protocol"], PROBE_PROTOCOL)
        self.assertIsNone(report["class_weight"])
        self.assertEqual(report["weight_decay"], 0)
        self.assertEqual(report["epochs_trained"], 1)
        self.assertEqual(report["stop_reason"], "max_epochs")
        train = splits["train"]
        mean, scale = x[train].mean(0), x[train].std(0, correction=0)
        weight = torch.zeros(2, x.shape[1], dtype=x.dtype, requires_grad=True)
        bias = torch.zeros(2, dtype=x.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=.01, weight_decay=0)
        truth = torch.tensor([report["class_order"].index(label) for label in labels[train]])
        loss = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear((x[train] - mean) / scale, weight, bias), truth)
        loss.backward()
        optimizer.step()
        expected_weight = weight.detach() / scale
        expected_bias = bias.detach() - expected_weight @ mean
        torch.testing.assert_close(state["weight"], expected_weight)
        torch.testing.assert_close(state["bias"], expected_bias)
        valid_truth = torch.tensor([report["class_order"].index(label)
                                    for label in labels[splits["valid"]]])
        restored_ce = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(x[splits["valid"]], **state), valid_truth).item()
        self.assertAlmostEqual(restored_ce, report["best_valid_ce"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_multiclass_matches_cpu_and_does_not_touch_rng_or_hidden_grad(self):
        x = torch.randn(120, 6, generator=torch.Generator().manual_seed(72)).double()
        labels = np.array(["a"] * 60 + ["b"] * 40 + ["c"] * 20, dtype=object)
        splits = split_items(np.arange(120), 42)
        labels[splits["test"][:2]] = "test_only"
        cpu_state, cpu = fit_category_probe(x, labels, splits)
        gpu_x = x.cuda().requires_grad_()
        rng = torch.cuda.get_rng_state().clone()
        state, report = fit_category_probe(gpu_x, labels, splits)
        self.assertEqual(report["device"], "cuda:0")
        self.assertEqual(report["best_epoch"], cpu["best_epoch"])
        self.assertIsNone(gpu_x.grad)
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        for key in state:
            self.assertTrue(state[key].is_cuda)
            self.assertFalse(state[key].requires_grad)
            torch.testing.assert_close(state[key].cpu(), cpu_state[key], atol=1e-5, rtol=1e-5)
        self.assertEqual(report["splits"]["test"]["confusion_matrix"],
                         cpu["splits"]["test"]["confusion_matrix"])
        changed_x = gpu_x.detach().clone()
        changed_x[splits["test"]] += 1000
        other, other_report = fit_category_probe(changed_x, labels, splits)
        self.assertEqual(report["best_epoch"], other_report["best_epoch"])
        for key in state:
            torch.testing.assert_close(state[key], other[key], rtol=0, atol=0)

    def test_validation_only_class_has_finite_ce_without_test_vocabulary_leak(self):
        state, report = fit_category_probe(
            torch.zeros(6, 2), ["a", "b", "c", "a", "test_only", "b"],
            {"train": [0, 1], "valid": [2, 3], "test": [4, 5]}, max_epochs=5)
        self.assertEqual(report["class_order"], ["a", "b", "c"])
        self.assertEqual(state["weight"].shape, (3, 2))
        self.assertTrue(np.isfinite(report["best_valid_ce"]))
        self.assertEqual(report["splits"]["test"]["per_class"]["test_only"]["recall"], 0)

    def test_filter_preserves_split_and_unavailable_probe_is_explicit(self):
        ids = np.arange(1, 41)
        splits = split_items(ids, 42)
        frame = pd.DataFrame({
            "ItemID": ids, "category": ["a" if i % 2 else "b" for i in ids],
            "status": ["included"] * len(ids),
        })
        frame.loc[splits["test"][0].item(), "status"] = "multiple_labels"
        with tempfile.TemporaryDirectory() as temp:
            probe = ItemCategoryProbe(frame.iloc[::-1], ids, splits, temp)
            self.assertEqual(probe.metadata["excluded_counts"], {"multiple_labels": 1})
            for split in ("train", "valid"):
                np.testing.assert_array_equal(probe.splits[split], splits[split])
            np.testing.assert_array_equal(probe.splits["test"], splits["test"][1:])
            frame["status"] = "missing_metadata"
            unavailable = ItemCategoryProbe(frame, ids, splits, temp)
            self.assertEqual(unavailable.evaluate(torch.ones(40, 3), 0)["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
