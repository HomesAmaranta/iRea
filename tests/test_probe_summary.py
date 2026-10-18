import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "debug/fig1_acc_cka_new/summarize.py"
spec = importlib.util.spec_from_file_location("probe_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ProbeSummaryTests(unittest.TestCase):
    def test_collect_keeps_encoder_items_separate_from_decoder_sequences(self):
        probe = {
            "epoch": 0, "cka": 0.4, "cka_genrec": 0.5,
            "cka_status": "valid", "cka_genrec_status": "valid",
            "protocol": "raw_hidden_affine_cosine_plus_mse_v1",
            "geometry": {"effective_rank": 3., "avg_cosine": 0.2,
                         "effective_rank_status": "valid", "cosine_status": "valid"},
            "test": {"acc": 0.7}, "splits": {"test": {"mse": 0.1}},
            "best_epoch": 1, "epochs_trained": 2, "best_valid_loss": 0.2,
            "category": {"status": "valid", "splits": {"test": {"accuracy": 0.7}},
                         "best_epoch": 1, "best_valid_ce": 0.3},
        }
        sequence = {
            "epoch": 0, "avg_cosine": 0.8, "effective_rank": 5.,
            "valid_batches": {"avg_cosine": 1, "effective_rank": 1},
            "skipped_batches": {"avg_cosine": 0, "effective_rank": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            for model in ("hstu", "tiger"):
                for item, weight in summary.VARIANTS:
                    folder = (root / f"out_linear2gram/{model}/amazon/beauty"
                              / f"align_item_{item}_cos{weight}_last_lr5e-4")
                    folder.mkdir(parents=True)
                    result = {
                        "model_name": model, "align_item": item, "mse_loss_weight": weight,
                        "align_loss_type": "cos", "best_epoch": 0, "best_valid_ce": 0.3,
                        "test": {"Recall@10": 0.1},
                        "sequence_geometry_path": str(folder / "sequence.jsonl"),
                        "sequence_geometry_latest": sequence,
                    }
                    (folder / "sequence.jsonl").write_text(json.dumps(sequence) + "\n")
                    for source in (("last", "encoder_mean") if model == "tiger" else ("last",)):
                        row = copy.deepcopy(probe)
                        row.update(run_id=source, hidden_source=source)
                        if source == "encoder_mean":
                            row["geometry"]["avg_cosine"] = 0.6
                        location = folder / f"item_probe_{source}" / source
                        location.mkdir(parents=True)
                        config = {
                            "protocol": probe["protocol"], "hidden_source": source,
                            "reference_sha256": "ref", "sid_sha256": "sid",
                            "split_item_ids": {"train": [1], "valid": [2], "test": [3]},
                            "fit_config": {"seed": 42},
                        }
                        summary.json_write(location / "config.json", config)
                        (location / "metrics.jsonl").write_text(json.dumps(row) + "\n")
                        prefix = "encoder_probe" if source == "encoder_mean" else "linear_probe"
                        result[prefix + "_dir"] = str(location)
                        result[prefix + "_latest"] = row
                    summary.json_write(folder / "results.json", result)
            with patch.object(summary, "ROOT", root), patch.object(summary, "OUT", output):
                runs, manifest = summary.collect()
            self.assertEqual(len(runs), 12)
            self.assertEqual(len(manifest), 12)
            for run in runs:
                record = run["records"][0]
                if run["identity"]["model"] == "tiger-encoder":
                    self.assertEqual(record["item_avg_cosine"], 0.6)
                    self.assertNotIn("sequence_avg_cosine", record)
                    self.assertEqual(run["identity"]["source_model"], "tiger")
                else:
                    self.assertEqual(record["sequence_avg_cosine"], 0.8)
                    self.assertEqual(record["item_avg_cosine"], 0.2)

    def test_three_rows_pdf_only_and_no_encoder_sequence_substitution(self):
        runs = []
        for model in summary.MODELS:
            for item, weight in summary.VARIANTS:
                records = [
                    {"epoch": epoch, "item_avg_cosine": 0.3 + 0.1 * epoch}
                    for epoch in range(2)
                ]
                if model != "tiger-encoder":
                    for record in records:
                        record["sequence_avg_cosine"] = 0.2
                runs.append({
                    "identity": {"model": model, "align_item": item, "align_weight": weight},
                    "records": records, "best": records[0],
                })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "geometry"
            with patch.object(summary.plt, "close"):
                summary.plot(
                    runs, [("item_avg_cosine", "Item cosine", False),
                           ("sequence_avg_cosine", "Sequence cosine", False)],
                    "Test", output,
                )
                figure = summary.plt.gcf()
                self.assertEqual(len(figure.axes), 6)
                self.assertEqual(
                    [figure.axes[i].get_title(loc="left") for i in (0, 2, 4)],
                    ["Beauty / HSTU", "Beauty / TIGER", "Beauty / TIGER-ENCODER"],
                )
                self.assertEqual(len(figure.axes[4].lines), 8)
                self.assertEqual(len(figure.axes[5].lines), 0)
                self.assertIn("Not computed", figure.axes[5].texts[0].get_text())
            summary.plt.close(figure)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["geometry.pdf"])
            self.assertTrue(output.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))

    def test_old_runs_rejected_before_any_output(self):
        with patch.object(summary, "load", return_value={"encoder_probe_latest": None}), \
                patch.object(summary.shutil, "copy2") as copy:
            with self.assertRaisesRegex(ValueError, "rerun training"):
                summary.collect()
            copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
