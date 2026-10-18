import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from models.lcrec import LCRec
from data.lcrec import LCSIDCollator, DEFAULT_PROMPT
from trainers.category_probe import CategoryProbeNumericalError
from trainers.affine_probe import PROTOCOL, RECALL_KS, fit_probe, retrieval_metrics
from trainers.representation_geometry import SequenceGeometryEpoch
from trainers.item_probe import cosine_gram_cka, genrec_linear_cka
from trainers.qwen3_item_probe import Qwen3ItemProbe, extract_qwen3_item_hidden, on_main_process
from trainers.qwen3_trainer import train_lcrec, aggregate_sequence_geometry


def tiny_model():
    lm = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        attention_dropout=0.1, pad_token_id=0, eos_token_id=1,
    ))
    lm = get_peft_model(lm, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], modules_to_save=["embed_tokens", "lm_head"],
    ))
    # Nonzero adapters catch accidental extraction from an unadapted backbone.
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if "lora_B" in name:
                p.fill_(0.02)
    return LCRec(lm, torch.arange(2, 14).reshape(3, 4), 1, 0, [[0, 1, 2]])


def prompt_collator():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens and text.endswith("?|start_of_answer|")
            codes = re.findall(r"<([abc])_(\d+)>", text)
            # Variable prefix length also exercises left-padding and position IDs.
            prefix = [15, 16] if codes[0][1] == "0" else [15]
            return prefix + [2 + "abc".index(layer) * 4 + int(code)
                             for layer, code in codes] + [17, 18]

    return LCSIDCollator(Tokenizer(), torch.arange(2, 14).reshape(3, 4))


class Dataset:
    def __init__(self):
        self.item_ids = np.arange(1, 41)
        self.sem_ids_list = [[i % 4, i // 4 % 4, i // 16 % 4] for i in range(40)]

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return SimpleNamespace(align_target_emb=None)


def reference(directory, ds):
    path = Path(directory) / "reference.parquet"
    rng = np.random.default_rng(12)
    pd.DataFrame({"ItemID": ds.item_ids[::-1],
                  "embedding": list(rng.normal(size=(40, 6)))}).to_parquet(path)
    return path


def labels(ds):
    return pd.DataFrame({
        "ItemID": ds.item_ids, "category": [f"parent > child{i % 3}" for i in range(40)],
        "status": ["included"] * 40,
    }), {"status": "ready"}


class QwenProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_extraction(self, device, precision):
        model = tiny_model().to(device).train()
        model.backbone.get_base_model().model.layers[0].eval()
        modes = [m.training for m in model.modules()]
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        weights = copy.deepcopy(model.state_dict())
        cpu_rng = torch.random.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        codes = Dataset().sem_ids_list
        with patch.object(model.backbone.get_base_model().lm_head, "forward",
                          side_effect=AssertionError("Probe must not compute logits")):
            hidden = extract_qwen3_item_hidden(model, codes, 7, precision, collator=prompt_collator())
        self.assertEqual(hidden.shape, (40, 16))
        self.assertEqual(hidden.device.type, device)
        self.assertFalse(hidden.requires_grad)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        if gpu_rng is not None:
            self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        for name, p in model.state_dict().items():
            torch.testing.assert_close(p, weights[name], rtol=0, atol=0)
        for p in model.parameters():
            if p.requires_grad:
                self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        model.eval()
        batch = prompt_collator()([SimpleNamespace(item_ids=c, target_ids=c) for c in codes])
        ids = batch["input_ids"].to(device)
        # Match training forward, including target tokens; causal masking must
        # leave the preceding question-mark representation unchanged.
        ids = torch.cat((ids, batch["labels"].to(device)), dim=1)
        mask = torch.cat((batch["attention_mask"].to(device),
                          torch.ones(len(codes), 3, device=device, dtype=torch.long)), dim=1)
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
            expected = model.backbone(
                input_ids=ids, attention_mask=mask, position_ids=positions, output_hidden_states=True,
                use_cache=False, return_dict=True,
            ).hidden_states[-1][:, -5].float()
        torch.testing.assert_close(hidden, expected, rtol=0.02 if precision == "bf16" else 1e-5,
                                   atol=0.02 if precision == "bf16" else 1e-5)

    def test_extraction_cpu(self):
        self.check_extraction("cpu", "no")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_extraction_gpu(self):
        for precision in ("no", "bf16"):
            self.check_extraction("cuda", precision)

    def test_invalid_codes(self):
        model = tiny_model()
        for codes in ([[0, 1]], [[0, 1, 4]], [[0., 1., 2.]]):
            with self.assertRaises(ValueError):
                extract_qwen3_item_hidden(model, codes, collator=prompt_collator())

    def test_raw_sid_exact_tokens_last_position_and_preserved_state(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for precision in ("no", "bf16"):
                model = tiny_model().to(device).train()
                decoder = model.backbone.get_base_model().model
                decoder.layers[0].eval()
                modes = [m.training for m in model.modules()]
                weights = copy.deepcopy(model.state_dict())
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = torch.ones_like(p)
                cpu_rng = torch.random.get_rng_state().clone()
                gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
                codes = Dataset().sem_ids_list
                collator = Mock(side_effect=AssertionError("Raw SID must not use prompt collator"))
                with patch.object(model.backbone.get_base_model().lm_head, "forward",
                                  side_effect=AssertionError("No logits")), \
                        patch.object(decoder, "forward", wraps=decoder.forward) as forward:
                    hidden = extract_qwen3_item_hidden(
                        model, codes, 7, precision, raw_sid=True, collator=collator)
                collator.assert_not_called()
                inputs = torch.cat([call.kwargs["input_ids"] for call in forward.call_args_list])
                ids = model.sid_token_ids[
                    torch.arange(3, device=device), torch.tensor(codes, device=device)]
                torch.testing.assert_close(inputs, ids)
                self.assertEqual(inputs.shape, (40, 3))
                for call in forward.call_args_list:
                    self.assertTrue(call.kwargs["attention_mask"].eq(1).all())
                    torch.testing.assert_close(
                        call.kwargs["position_ids"],
                        torch.arange(3, device=device).expand_as(call.kwargs["input_ids"]))
                self.assertEqual(modes, [m.training for m in model.modules()])
                self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
                if gpu_rng is not None:
                    self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
                for p in model.parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, torch.ones_like(p))
                self.assertFalse(hidden.requires_grad)
                model.eval()
                with torch.no_grad(), torch.autocast(
                    device, dtype=torch.bfloat16, enabled=precision == "bf16"
                ):
                    expected = model.backbone(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        position_ids=torch.arange(3, device=device).expand_as(ids),
                        output_hidden_states=True, use_cache=False, return_dict=True,
                    ).hidden_states[-1][:, -1].float()
                torch.testing.assert_close(hidden, expected, atol=0.02, rtol=0.02)

    def test_metrics_only_and_device(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            ds = Dataset()
            with tempfile.TemporaryDirectory() as directory:
                path = reference(directory, ds)
                with patch("trainers.qwen3_item_probe.load_dataset_labels", return_value=labels(ds)):
                    probe = Qwen3ItemProbe(ds, path, directory, Mock(), collator=prompt_collator(),
                                          max_epochs=5)
                model = tiny_model().to(device).train()
                before = torch.random.get_rng_state().clone()
                report = probe.evaluate(model, 0, "no")
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                self.assertTrue(model.training)
                self.assertEqual(probe.target.device.type, device)
                self.assertEqual(probe.latest["category"]["device"].split(":")[0], device)
                hidden = extract_qwen3_item_hidden(model, ds.sem_ids_list, mixed_precision="no",
                                                  collator=prompt_collator())
                self.assertAlmostEqual(report["linear_probe/cka"],
                                       cosine_gram_cka(hidden, probe.target), places=10)
                self.assertEqual(report["linear_probe/cka_genrec"], genrec_linear_cka(hidden, probe.target))
                self.assertEqual(probe.latest["cka_genrec"], report["linear_probe/cka_genrec"])
                raw = extract_qwen3_item_hidden(
                    model, ds.sem_ids_list, mixed_precision="no", raw_sid=True)
                self.assertAlmostEqual(report["raw_sid/cka"],
                                       cosine_gram_cka(raw, probe.target), places=10)
                self.assertEqual(report["raw_sid/cka_genrec"], genrec_linear_cka(raw, probe.target))
                raw_record = probe.latest["raw_sid"]
                self.assertEqual(raw_record["items"], 40)
                self.assertEqual(raw_record["cka_status"], "valid")
                self.assertEqual(raw_record["cka"], report["raw_sid/cka"])
                self.assertNotIn("category", raw_record)
                self.assertNotIn("splits", raw_record)
                serialized = json.loads((probe.directory / "metrics.jsonl").read_text())
                self.assertEqual(serialized["raw_sid"], raw_record)
                config = json.loads((probe.directory / "config.json").read_text())
                self.assertTrue(config["raw_sid"]["enabled"])
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertFalse(list(Path(directory).rglob("*.pt")))
                self.assertIn("category_probe/test_acc", report)
                self.assertEqual(probe.latest["category"]["category_level"], 2)
                state, expected = fit_probe(hidden, probe.target, probe.splits, **probe.fit_config)
                self.assertEqual(probe.latest["protocol"], PROTOCOL)
                self.assertEqual(probe.latest["best_epoch"], expected["best_epoch"])
                for split, values in expected["splits"].items():
                    for key, value in values.items():
                        self.assertEqual(report[f"linear_probe/{split}_{key}"], value)
                test_rows = probe.splits["test"].to(hidden.device)
                recall, _ = retrieval_metrics(
                    torch.nn.functional.linear(hidden[test_rows], **state),
                    probe.target, test_rows, top_ks=RECALL_KS, item_ids=ds.item_ids,
                )
                for key, value in recall.items():
                    self.assertEqual(report[f"linear_probe/test_{key}"], value["value"])
                    self.assertEqual(probe.latest["test"][key], value["value"])
                self.assertEqual(probe.latest["test"]["acc"],
                                 probe.latest["category"]["splits"]["test"]["accuracy"])

    def test_rank_zero_only_and_errors(self):
        action = Mock(return_value=7)
        worker = SimpleNamespace(is_main_process=False, num_processes=2, wait_for_everyone=Mock())
        with patch("trainers.qwen3_item_probe.broadcast_object_list") as broadcast:
            self.assertIsNone(on_main_process(worker, action))
            action.assert_not_called()
            broadcast.assert_called_once()
        main = SimpleNamespace(is_main_process=True, num_processes=1, wait_for_everyone=Mock())
        self.assertEqual(on_main_process(main, action), 7)
        with self.assertRaisesRegex(RuntimeError, "rank zero"):
            on_main_process(main, Mock(side_effect=ValueError("bad catalogue")))
        with patch("trainers.qwen3_item_probe.broadcast_object_list",
                   side_effect=lambda values, **kwargs: values.__setitem__(0, "remote failure")):
            with self.assertRaisesRegex(RuntimeError, "remote failure"):
                on_main_process(worker, action)

    def test_regression_learning_rate_does_not_change_category(self):
        ds = Dataset()
        model = tiny_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = reference(directory, ds)
            reports = []
            for lr in (1e-3, 0.5):
                with patch("trainers.qwen3_item_probe.load_dataset_labels", return_value=labels(ds)):
                    probe = Qwen3ItemProbe(
                        ds, path, directory, Mock(), collator=prompt_collator(),
                        learning_rate=lr, max_epochs=3,
                    )
                probe.evaluate(model, 0, "no")
                reports.append(probe.latest)
            self.assertNotEqual(reports[0]["test"]["mse"], reports[1]["test"]["mse"])
            self.assertEqual(reports[0]["category"]["splits"], reports[1]["category"]["splits"])
            self.assertEqual(reports[0]["category"]["best_epoch"],
                             reports[1]["category"]["best_epoch"])

    def test_nonfinite_ce_keeps_cka_without_fake_accuracy(self):
        ds = Dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = reference(directory, ds)
            with patch("trainers.qwen3_item_probe.load_dataset_labels", return_value=labels(ds)):
                probe = Qwen3ItemProbe(ds, path, directory, Mock(), collator=prompt_collator(),
                                      max_epochs=5)
            model = tiny_model().train()
            error = CategoryProbeNumericalError("cpu", 2)
            with patch.object(probe.category, "evaluate", side_effect=error):
                report = probe.evaluate(model, 0, "no", global_step=50)
            self.assertIn("linear_probe/cka", report)
            self.assertIn("linear_probe/cka_genrec", report)
            self.assertNotIn("category_probe/test_acc", report)
            record = json.loads((probe.directory / "metrics.jsonl").read_text())
            self.assertEqual(record["category"]["status"], "nonfinite")
            self.assertEqual(record["category"]["failed_epoch"], 2)
            self.assertEqual(record["global_step"], 50)
            self.assertNotIn("splits", record["category"])
            self.assertIsNone(record["test"]["acc"])
            self.assertIn("Recall@100", record["test"])
            self.assertTrue(model.training)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            # Unexpected failures must still propagate, not silently become missing acc.
            with patch.object(probe.category, "evaluate", side_effect=RuntimeError("OOM")):
                with self.assertRaisesRegex(RuntimeError, "OOM"):
                    probe.evaluate(model, 0, "no", global_step=100)

    def test_trainer_every_epoch_and_no_training_change(self):
        self.check_training_schedule(2, 1, [2, 4], 0)

    def test_step_schedule_no_final_partial_probe(self):
        self.check_training_schedule(3, 1, [3], 0)

    def test_short_training_has_no_probe(self):
        self.check_training_schedule(50, 1, [], 0)

    def test_step_schedule_counts_accumulated_updates(self):
        self.check_training_schedule(1, 2, [1, 2], 0)

    def test_final_exact_interval_not_duplicated(self):
        self.check_training_schedule(2, 1, [2, 4], 0)

    def check_training_schedule(self, interval, accumulation, expected_steps, best_epoch):
        ds = Dataset()
        states = []
        with tempfile.TemporaryDirectory() as directory:
            path = reference(directory, ds)
            for enabled in (False, True):
                saved = []
                evaluations = []

                def build(*args, **kwargs):
                    model = tiny_model()
                    saved.append(model)
                    return model

                def collate(samples):
                    n = len(samples)
                    return {"input_ids": torch.tensor([[2, 6, 10]] * n),
                            "attention_mask": torch.ones(n, 3, dtype=torch.long),
                            "labels": torch.tensor([[3, 7, 11]] * n)}
                collate.prompt_template = DEFAULT_PROMPT

                def evaluate(*args, metric, **kwargs):
                    evaluations.append(metric)
                    return {"ce": 1.} if metric == "ce" else {"Recall@1": 0.5, "NDCG@1": 0.5}

                output = Path(directory) / str(enabled)
                output.mkdir()
                with patch("trainers.qwen3_trainer.load_lm_datasets",
                           return_value=dict(train=ds, valid=ds, test=ds)), \
                        patch("trainers.qwen3_trainer.AutoTokenizer.from_pretrained", return_value=Mock()), \
                        patch("trainers.qwen3_trainer.register_sid_tokens", return_value=torch.arange(12).view(3, 4)), \
                        patch("trainers.qwen3_trainer.LCSIDCollator", return_value=collate), \
                        patch("trainers.qwen3_trainer.build_lcrec", side_effect=build), \
                        patch("trainers.qwen3_trainer.setup_training_logger", return_value=Mock()), \
                        patch("trainers.qwen3_trainer.setup_accelerator",
                              return_value=Accelerator(cpu=True, mixed_precision="no",
                                                       gradient_accumulation_steps=accumulation)), \
                        patch("trainers.qwen3_trainer.evaluate_lm", side_effect=evaluate), \
                        patch("trainers.qwen3_item_probe.load_dataset_labels", return_value=labels(ds)):
                    result = train_lcrec(
                        save_dir_root=str(output), epochs=2, batch_size=2, infer_batch_size=2,
                        mixed_precision="no", codebook_size=4, sem_id_dim=3, beam_size=1, top_ks=(1,),
                        num_warmup_steps=0, linear_probe=enabled,
                        linear_probe_max_epochs=5, linear_probe_patience=2,
                        sequence_geometry=enabled, sequence_geometry_cosine_pairs=64,
                        linear_probe_every_steps=interval, gradient_accumulate_every=accumulation,
                        linear_probe_reference_path=str(path),
                    )
                states.append(copy.deepcopy(saved[0].state_dict()))
                self.assertFalse(list(output.rglob("*.pt")))
                self.assertFalse(list(output.rglob("*.safetensors")))
                self.assertEqual(evaluations, ["ce", "hr", "ce"])
                self.assertEqual(result["best_epoch"], best_epoch)
                self.assertEqual(result["best_step"], 2 // accumulation)
                if enabled:
                    sequence_rows = [
                        json.loads(line) for line in
                        Path(result["sequence_geometry_path"]).read_text().splitlines()
                    ]
                    self.assertEqual([row["epoch"] for row in sequence_rows], [0, 1])
                    self.assertEqual(result["sequence_geometry_latest"], sequence_rows[-1])
                    for row in sequence_rows:
                        self.assertEqual(row["batches"], 2)
                        self.assertEqual(row["tokens"], 12)
                        self.assertEqual(row["sampled_pairs"], 128)
                        self.assertEqual(row["valid_batches"], dict(effective_rank=2, avg_cosine=2))
                else:
                    self.assertIsNone(result["sequence_geometry_latest"])
                    self.assertIsNone(result["sequence_geometry_path"])
                if enabled:
                    paths = list(output.rglob("metrics.jsonl"))
                    records = [json.loads(line) for line in paths[0].read_text().splitlines()] if paths else []
                    self.assertEqual([r["global_step"] for r in records], expected_steps)
                    if result["best_step"] in expected_steps:
                        self.assertEqual(result["linear_probe_best"]["epoch"], best_epoch)
                    else:
                        self.assertIsNone(result["linear_probe_best"])
                    if not expected_steps:
                        self.assertIsNone(result["linear_probe_latest"])
                    self.assertTrue((output / "results.json").exists())
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)

    def test_sequence_capture_matches_supervised_hidden_with_checkpointing(self):
        model = tiny_model().train()
        model.backbone.enable_input_require_grads()
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        batch = prompt_collator()([
            SimpleNamespace(item_ids=code, target_ids=code)
            for code in Dataset().sem_ids_list[:3]
        ])
        decoder = model.backbone.get_base_model().model
        hooks_before = len(decoder.norm._forward_hooks)
        rng = torch.random.get_rng_state().clone()
        with patch.object(decoder, "forward", wraps=decoder.forward) as forward:
            output = model(batch, return_last_hidden_state=True)
            self.assertFalse(forward.call_args.kwargs["output_hidden_states"])
        self.assertEqual(len(decoder.norm._forward_hooks), hooks_before)
        self.assertFalse(output.last_hidden_state.requires_grad)
        self.assertEqual(output.last_hidden_state.shape, (3, 3, 16))
        ids = torch.cat((batch["input_ids"], batch["labels"]), dim=1)
        mask = torch.cat((batch["attention_mask"], torch.ones_like(batch["labels"])), dim=1)
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        torch.random.set_rng_state(rng)
        expected = model.backbone(
            input_ids=ids, attention_mask=mask, position_ids=positions,
            use_cache=False, output_hidden_states=True, return_dict=True,
        ).hidden_states[-1][:, -4:-1]
        torch.testing.assert_close(output.last_hidden_state, expected)
        output.loss.backward()
        self.assertEqual(len(decoder.norm._forward_hooks), hooks_before)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        with patch.object(model.backbone, "forward", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                model(batch, return_last_hidden_state=True)
        self.assertEqual(len(decoder.norm._forward_hooks), hooks_before)

    def test_distributed_sequence_reduces_valid_batch_counts(self):
        stats = SequenceGeometryEpoch("cpu")
        stats.sums = dict(effective_rank=8., avg_cosine=0.6)
        stats.counts = dict(effective_rank=2, avg_cosine=1)
        stats.batches, stats.tokens, stats.nonzero_tokens, stats.sampled_pairs = 2, 12, 12, 10
        remote = torch.tensor([3., 0.8, 1, 2, 2, 9, 8, 20], dtype=torch.float64)
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            reduce=Mock(side_effect=lambda values, reduction: values + remote),
        )
        result = aggregate_sequence_geometry(accelerator, stats)
        self.assertAlmostEqual(result["effective_rank"], 11 / 3)
        self.assertAlmostEqual(result["avg_cosine"], 1.4 / 3)
        self.assertEqual(result["batches"], 4)
        self.assertEqual(result["tokens"], 21)
        self.assertEqual(result["sampled_pairs"], 30)
        self.assertEqual(result["skipped_batches"], dict(effective_rank=1, avg_cosine=1))
        empty = SequenceGeometryEpoch("cpu")
        accelerator.reduce.side_effect = lambda values, reduction: values
        result = aggregate_sequence_geometry(accelerator, empty)
        self.assertIsNone(result["effective_rank"])
        self.assertIsNone(result["avg_cosine"])


if __name__ == "__main__":
    unittest.main()
