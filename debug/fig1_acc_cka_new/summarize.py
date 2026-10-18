"""Export the eight Beauty runs selected by train_grid.sh, without mixing runs."""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODELS = ("hstu", "tiger", "tiger-encoder")
VARIANTS = (("current", 0), ("next", 0), ("current", 5), ("next", 5))
STYLES = {
    ("current", 0): ("#7b8490", "--", "current / weight 0"),
    ("next", 0): ("#d18b28", ":", "next / weight 0"),
    ("current", 5): ("#1683de", "-", "current / weight 5"),
    ("next", 5): ("#8155ce", "-", "next / weight 5"),
}


def load(path):
    return json.loads(path.read_text())


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rows_read(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def scalar_fields(data, prefix):
    return {
        prefix + key: value
        for key, value in data.items()
        if isinstance(value, (int, float, str)) or value is None
    }


def flatten(probe, sequence):
    result = {
        "epoch": probe["epoch"],
        "cka": probe["cka"],
        "cka_genrec": probe["cka_genrec"],
        "cka_status": probe["cka_status"],
        "cka_genrec_status": probe["cka_genrec_status"],
    }
    result.update(scalar_fields(probe["geometry"], "item_"))
    result.update(scalar_fields(probe["test"], "probe_test_"))
    for split, metrics in probe["splits"].items():
        result.update(scalar_fields(metrics, f"probe_{split}_"))
    for split, metrics in probe["category"]["splits"].items():
        result.update(scalar_fields(metrics, f"category_{split}_"))
    result.update({
        "probe_fit_best_epoch": probe["best_epoch"],
        "probe_fit_epochs_trained": probe["epochs_trained"],
        "probe_fit_best_valid_loss": probe["best_valid_loss"],
        "category_fit_best_epoch": probe["category"]["best_epoch"],
        "category_fit_best_valid_ce": probe["category"]["best_valid_ce"],
    })
    if sequence is not None:
        result.update(scalar_fields(sequence, "sequence_"))
        for key in ("valid_batches", "skipped_batches"):
            result.update(scalar_fields(sequence[key], "sequence_" + key + "_"))
    return result


def csv_write(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def collect():
    runs = []
    manifest = []
    # Reject old TIGER runs before writing any snapshots or reports.
    for item, weight in VARIANTS:
        path = (ROOT / "out_linear2gram/tiger/amazon/beauty"
                / f"align_item_{item}_cos{weight}_last_lr5e-4/results.json")
        if not load(path).get("encoder_probe_latest"):
            raise ValueError(f"{path}: rerun training with tiger_encoder_probe=True first")
    for model in MODELS:
        encoder = model == "tiger-encoder"
        source_model = "tiger" if encoder else model
        for item, weight in VARIANTS:
            run_dir = ROOT / "out_linear2gram" / source_model / "amazon/beauty"
            run_dir /= f"align_item_{item}_cos{weight}_last_lr5e-4"
            result_path = run_dir / "results.json"
            result = load(result_path)
            assert (result["model_name"], result["align_item"], result["mse_loss_weight"]) == (
                source_model, item, weight
            )
            assert result["align_loss_type"] == "cos"
            if source_model == "tiger" and not result.get("encoder_probe_latest"):
                raise ValueError(
                    f"{result_path}: missing encoder_mean run; finish the new training before "
                    "generating the HSTU / TIGER / TIGER-encoder comparison"
                )
            prefix = "encoder_probe" if encoder else "linear_probe"
            latest = result[f"{prefix}_latest"]
            probe_dir = ROOT / result[f"{prefix}_dir"]
            assert probe_dir.parent.parent == run_dir
            assert probe_dir.name == latest["run_id"]
            config_path = probe_dir / "config.json"
            config = load(config_path)
            assert config["protocol"] == "raw_hidden_affine_cosine_plus_mse_v1"
            hidden_source = "encoder_mean" if encoder else "last"
            assert config.get("hidden_source", "last") == hidden_source
            if encoder:
                original_config = load(ROOT / result["linear_probe_dir"] / "config.json")
                for key in ("reference_sha256", "sid_sha256", "split_item_ids", "fit_config"):
                    assert config[key] == original_config[key], f"Probe comparison mismatch: {key}"
                assert latest["epoch"] == result["linear_probe_latest"]["epoch"]
            probe_path = probe_dir / "metrics.jsonl"
            sequence_path = ROOT / result["sequence_geometry_path"]
            assert sequence_path.parent == run_dir
            sequence_rows = list(rows_read(sequence_path))
            sequence = {row["epoch"]: row for row in sequence_rows}
            assert len(sequence) == len(sequence_rows)
            records = []
            for probe in rows_read(probe_path):
                assert probe["run_id"] == latest["run_id"]
                assert probe.get("hidden_source", "last") == hidden_source
                assert probe["protocol"] == config["protocol"]
                assert probe["cka_status"] == probe["cka_genrec_status"] == "valid"
                assert probe["category"]["status"] == "valid"
                assert probe["geometry"]["effective_rank_status"] == "valid"
                assert probe["geometry"]["cosine_status"] == "valid"
                seq = sequence[probe["epoch"]]
                assert all(value == 0 for value in seq["skipped_batches"].values())
                records.append(flatten(probe, None if encoder else seq))
            assert [row["epoch"] for row in records] == list(range(latest["epoch"] + 1))
            assert set(sequence) == {row["epoch"] for row in records}
            assert records[-1] == flatten(
                latest, None if encoder else result["sequence_geometry_latest"]
            )
            best = next(row for row in records if row["epoch"] == result["best_epoch"])
            name = f"{model}_{item}_cos{weight}"
            identity = {
                "experiment": name, "model": model, "align_item": item,
                "align_weight": weight, "run_id": latest["run_id"],
                "source_model": source_model, "hidden_source": hidden_source,
            }
            recommendation = {
                "recommendation_best_epoch": result["best_epoch"],
                "recommendation_best_valid_ce": result["best_valid_ce"],
                **{f"recommendation_test_{k}": v for k, v in result["test"].items()},
            }
            runs.append({
                "identity": identity, "result": result, "records": records,
                "best": {**identity, **recommendation, **best},
                "last": {**identity, **recommendation, **records[-1]},
            })
            snapshot = OUT / "sources" / name
            snapshot.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_path, snapshot / "results.json")
            shutil.copy2(config_path, snapshot / "probe_config.json")
            manifest.append({
                **identity,
                "epochs": len(records),
                "first_epoch": records[0]["epoch"],
                "last_epoch": records[-1]["epoch"],
                "best_recommendation_epoch": result["best_epoch"],
                "sources": {
                    key: {"path": str(path), "sha256": digest(path)}
                    for key, path in {
                        "result": result_path, "probe_config": config_path,
                        "probe_metrics": probe_path, "sequence_metrics": sequence_path,
                    }.items()
                },
            })
    return runs, manifest


def collect_qwen(item, weight):
    dirname = "sft_lr0.0005" if weight == 0 else f"sft_lr0.0005_item_{item}_last_inp_cos{weight}"
    result_path = ROOT / "out/lcrec/amazon/beauty" / dirname / "results.json"
    result = load(result_path)
    config, extra = result["config"], result["extra"]
    assert result["split"] == "beauty" and result["model"] == "lcrec"
    assert config["mse_loss_weight"] == weight and config["learning_rate"] == 0.0005
    assert config["align_item"] == item and config["align_target"] == "item"
    assert config["align_loss_type"] == "cos"
    assert Path(config["base_model"]).name == "Qwen3-1.7B"
    probe_dir = ROOT / extra["linear_probe_directory"]
    probe_path = probe_dir / "metrics.jsonl"
    probe_config = load(probe_dir / "config.json")
    assert probe_config["probe_input_format"] == "single_item_prompt_last_inp_v1"
    # Check common targets and item splits, but keep the model-specific hidden protocol.
    reference = load(OUT / "sources/hstu_current_cos0/probe_config.json")
    for key in ("reference_sha256", "sid_sha256", "split_item_ids", "fit_config"):
        assert probe_config[key] == reference[key], f"Qwen3 mismatch: {key}"
    sequence_path = ROOT / extra["sequence_geometry_path"]
    sequence = list(rows_read(sequence_path))
    assert [r["epoch"] for r in sequence] == list(range(len(sequence)))
    assert sequence[-1] == extra["sequence_geometry_latest"]
    assert sequence[-1]["global_step"] == extra["global_step"]
    probes = list(rows_read(probe_path))
    # The run-result serializer drops __missing_second_level__ per-class keys.
    # Check all exported scalars and the exact run/step identity against the raw JSONL.
    assert flatten(probes[-1], None) == flatten(extra["linear_probe_latest"], None)
    for key in ("run_id", "global_step"):
        assert probes[-1][key] == extra["linear_probe_latest"][key]
    interval = config["linear_probe_every_steps"]
    assert [r["global_step"] for r in probes] == list(
        range(interval, extra["global_step"] + 1, interval)
    )
    records = []
    for probe in probes:
        assert probe["run_id"] == probe_config["run_id"] == probe_dir.name
        assert probe["protocol"] == probe_config["protocol"]
        assert probe["probe_input_format"] == probe_config["probe_input_format"]
        assert probe["cka_status"] == probe["cka_genrec_status"] == "valid"
        assert probe["category"]["status"] == "valid"
        assert probe["geometry"]["effective_rank_status"] == "valid"
        assert probe["geometry"]["cosine_status"] == "valid"
        record = flatten(probe, None)
        record["global_step"] = probe["global_step"]
        raw = probe.get("raw_sid")
        if raw is not None:
            assert raw["probe_input_format"] == "single_item_raw_sid_last_v1"
            assert raw["cka_status"] == raw["cka_genrec_status"] == "valid"
            record.update(scalar_fields(raw, "raw_sid_"))
        records.append(record)
    for row in sequence:
        assert all(value == 0 for value in row["skipped_batches"].values())
    identity = {
        "experiment": f"qwen3_1.7b_{item}_cos{weight}", "model": "qwen3",
        "source_model": "lcrec", "base_model": config["base_model"],
        "align_item": item, "align_weight": weight, "run_id": probe_dir.name,
        "hidden_source": "single_item_prompt_last_inp_v1",
    }
    recommendation = {
        "recommendation_best_epoch": extra["best_epoch"],
        "recommendation_best_step": extra["best_step"],
        "recommendation_best_valid_ce": extra["best_valid_value"],
        **{f"recommendation_test_{key}": value for key, value in extra["test"].items()},
    }
    best = next((r for r in records if r["global_step"] == extra["best_step"]), None)
    assert (best is None) == (extra["linear_probe_best"] is None)
    best_summary = {**identity, **recommendation}
    best_summary.update(best if best is not None else {
        "epoch": extra["best_epoch"], "global_step": extra["best_step"],
        "probe_test_acc": None, "cka": None, "cka_genrec": None,
    })
    best_summary["probe_at_best_step_available"] = best is not None
    run = {
        "identity": identity, "result": result, "records": records, "sequence_records": sequence,
        "best": best_summary, "last": {**identity, **recommendation, **records[-1]},
    }
    sources = {
        "result": result_path, "probe_config": probe_dir / "config.json",
        "probe_metrics": probe_path, "sequence_metrics": sequence_path,
    }
    snapshot = OUT / "sources" / identity["experiment"]
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, snapshot / "results.json")
    shutil.copy2(probe_dir / "config.json", snapshot / "probe_config.json")
    manifest = {
        **identity, "probe_points": len(records), "sequence_epochs": len(sequence),
        "first_step": records[0]["global_step"], "last_step": records[-1]["global_step"],
        "training_final_step": extra["global_step"],
        "best_recommendation_epoch": extra["best_epoch"],
        "best_recommendation_step": extra["best_step"],
        "probe_at_best_step_available": best is not None,
        "raw_sid_probe_points": sum("raw_sid_cka" in row for row in records),
        "sources": {key: {"path": str(path), "sha256": digest(path)}
                    for key, path in sources.items()},
    }
    return run, manifest


def plot(runs, metrics, title, path):
    models = MODELS + (("qwen3",) if any(r["identity"]["model"] == "qwen3" for r in runs) else ())
    fig, axes = plt.subplots(
        len(models), len(metrics), figsize=(6 * len(metrics), 3.4 * len(models)), squeeze=False
    )
    for model_index, model in enumerate(models):
        for col, (metric, label, percentage) in enumerate(metrics):
            ax = axes[model_index, col]
            if model == "tiger-encoder" and metric.startswith("sequence_"):
                ax.set_title("Beauty / TIGER-ENCODER", loc="left", weight="bold")
                ax.text(0.5, 0.5, "Not computed\nEncoder mean is an item-level representation",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue
            for run in runs:
                ident = run["identity"]
                if ident["model"] != model:
                    continue
                variant = (ident["align_item"], ident["align_weight"])
                color, style, _ = STYLES[variant]
                if model == "qwen3":
                    is_sequence = metric.startswith("sequence_")
                    records = run["sequence_records"] if is_sequence else run["records"]
                    key = metric.removeprefix("sequence_") if is_sequence else metric
                    xkey = "epoch" if is_sequence else "global_step"
                    values = [row[key] for row in records]
                    assert all(value is not None and math.isfinite(value) for value in values)
                    ax.plot([row[xkey] for row in records], values, color=color, ls=style,
                            lw=1.6, marker=".", ms=4)
                    best_key = "recommendation_best_epoch" if is_sequence else "recommendation_best_step"
                    best_x = run["best"][best_key]
                    match = next((row for row in records if row[xkey] == best_x), None)
                    if match is not None:
                        ax.plot(best_x, match[key], "o", color=color, mfc="white", ms=5, mew=1.4)
                    elif not is_sequence:
                        ax.axvline(best_x, color=color, ls=":", lw=1, alpha=0.5)
                    continue
                x = [row["epoch"] for row in run["records"]]
                y = [row[metric] for row in run["records"]]
                assert all(value is not None and math.isfinite(value) for value in y)
                ax.plot(x, y, color=color, ls=style, lw=1.6)
                ax.plot(run["best"]["epoch"], run["best"][metric], "o",
                        color=color, mfc="white", ms=5, mew=1.4)
            label_model = "Qwen3-1.7B (LCRec)" if model == "qwen3" else model.upper()
            ax.set_title(f"Beauty / {label_model}", loc="left", weight="bold")
            ax.set_xlabel("Optimizer step" if model == "qwen3" and not metric.startswith("sequence_")
                          else "Epoch (0-based)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if percentage:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles = [
        Line2D([0], [0], color=color, ls=style, label=label, lw=2)
        for color, style, label in STYLES.values()
    ]
    handles.append(Line2D([0], [0], color="#444444", ls="", marker="o",
                          mfc="white", label="Best recommender epoch (valid CE)"))
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=3, fontsize=10)
    fig.text(0.5, 0.008,
             "No smoothing. Weight-0 curves may overlap. Each curve ends at its actual final epoch.\n"
             "Circle: diagnostics at the best recommender checkpoint, not the best probe test score.\n"
             "Qwen3: item x-axis = steps, sequence x-axis = epochs; dotted vertical lines = best steps without probes.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def documentation(runs, baselines):
    def display(value, scale=1, digits=5, suffix=""):
        return "NA" if value is None else f"{value * scale:.{digits}f}{suffix}"

    lines = [
        "# Beauty 实验结果",
        "",
        "范围：HSTU/TIGER × current/next × cos 权重 0/5；学习率 5e-4，seed=42。",
        f"图按 HSTU、TIGER、TIGER-encoder、Qwen3-1.7B 四行排列；{len(runs) - 4} 组训练，{len(runs)} 组表示诊断。",
        "Qwen3 包含 baseline（current、权重 0）、current ×5、next ×5；目标均为 item，损失为 cos。",
        "TIGER-encoder 来自同一个 TIGER checkpoint，不是额外训练的推荐模型。",
        "只读取各 results.json 指向的 run ID，未混入目录中的历史运行。",
        "",
        "## 推荐测试与最佳 epoch 的诊断",
        "",
        "推荐指标来自按验证 CE 选出的最佳 checkpoint；以下探针和几何指标取同一 epoch 的记录。",
        "epoch 从 0 开始。acc 和 Recall 在表中显示为百分比；CSV 保留原始 0–1 数值。",
        "",
        "| 模型 | align item | 权重 | 最佳 epoch | 最后 epoch | 推荐 R@10 | 推荐 N@10 | 分类 acc | 论文 CKA | genrec CKA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        row = run["best"]
        lines.append(
            f"| {row['model'].upper()} | {row['align_item']} | {row['align_weight']} | "
            f"{row['recommendation_best_epoch']} | {run['last']['epoch']} | "
            f"{100 * row['recommendation_test_Recall@10']:.3f}% | "
            f"{100 * row['recommendation_test_NDCG@10']:.3f}% | "
            f"{display(row['probe_test_acc'], 100, 2, '%')} | "
            f"{display(row['cka'])} | {display(row['cka_genrec'])} |"
        )
    lines += [
        "",
        "## 文件",
        "",
        "- `acc/beauty_category_acc.*`：测试分类 acc 曲线。",
        "- `cka/beauty_cka_comparison.*`：论文 debiased CKA 与 genrec linear CKA，分开显示。",
        "- `overview/beauty_acc_cka.*`：acc 和两种 CKA 的总览。",
        "- `geometry/beauty_effective_rank.*`：物品级和序列级有效秩。",
        "- `geometry/beauty_avg_cosine.*`：物品级和序列级平均 cos。",
        "- `probe/beauty_regression.*`：测试集回归 cos、MSE。",
        "- `probe/beauty_retrieval.*`：测试物品检索全量 Sentence-T5 库的 Recall@10/20/50/100。",
        "- `summary_best_epoch.csv`：推荐最佳 epoch 的全部诊断及最终推荐测试结果。",
        "- `summary_last_epoch.csv`：最后一轮诊断；其中推荐测试列仍是最佳 checkpoint 的结果。",
        "- `epoch_metrics.csv`：全部表示的标量，包括 train/valid/test 探针和几何统计；Qwen3 以 global_step 区分同 epoch 的多个点。",
        "- `qwen3_sequence_epoch_metrics.csv`：Qwen3 各实验序列统计，含 experiment/run_id；不复制到每个物品探针点。",
        "- `cka/qwen3_raw_sid_cka.pdf`：Qwen3 仅三个 SID、末 token 最后层 hidden 的两种 CKA，与主图 prompt 分支分开。",
        "- `summary.json`：最佳和最后一轮汇总，以及两组零权重基线的一致性检查。",
        "- `sources/`：各组 results.json 和探针配置快照。",
        "- `manifest.json`：精确来源、run ID、epoch 范围和源文件 SHA256。",
        "- `summarize.py`：从原始实验目录重新生成上述文件。",
        "",
        "按用户偏好，今后图表只输出 PDF，不再生成 PNG/SVG；无平滑、无插值、不截断不同长度的曲线。",
        "",
        "## 指标口径",
        "",
        "- 论文 CKA：原始物品 hidden 对 Sentence-T5，95 分位数裁剪、L2 归一化、去偏。",
        "- genrec CKA：原始物品 hidden 对 Sentence-T5，列中心化的 linear CKA。",
        "- 分类 acc：独立分类探针，固定 8:1:1 物品划分，测试集统计。",
        "- 回归 cos/MSE：独立仿射探针用 MSE+(1-cos) 训练，验证集选模；测试集评估。",
        "- 探针 Recall：测试物品作为查询，全量 Sentence-T5 物品作为候选；不是推荐 Recall。",
        "- item 几何：全量单物品 last hidden；有效秩基于中心化原始特征协方差，cos 为不同非零物品对的精确平均。",
        "- sequence 几何：训练 batch 的统一 last_hidden_state；epoch 指标是有效 batch 的等权平均。",
        "- TIGER-encoder：encoder 最后一层三个原始 SID token hidden 先取 FP32 mean，再计算物品指标；探针独立训练。",
        "- TIGER-encoder 没有额外的序列级统计，对应格子标记未计算，不复用 decoder 序列结果。",
        "- Qwen3：单物品训练 prompt 的答案分隔符前 hidden；每 50 optimizer step 测物品指标，横轴使用实际 step。",
        "- Qwen3 序列指标：三个 SID 预测位置的训练 batch 几何均值，多卡按有效 batch 数汇总，横轴为 epoch。",
        "- Qwen3 推荐最佳 step 若没有物品探针，最佳汇总填 NA，不插值也不选邻近点；图中用竖虚线标注。",
        "- 裸 SID CKA 不替代 prompt CKA，CSV 以 raw_sid_ 前缀区分；无记录的 baseline 不补值、不引用历史旧公式。",
        "- 不用测试 acc 或测试 Recall 挑选主模型 epoch；圆点统一标记验证 CE 最佳的推荐 epoch。",
        "- 原始 epoch JSONL 含大量探针内部优化历史，未复制；保留来源及 SHA256，标量已完整导出。",
        "",
        "## 完整性检查",
        "",
        "- HSTU/TIGER 十二组表示诊断的 epoch 从 0 连续覆盖到最后一轮；Qwen3 定步数记录完整覆盖所有到达的 50-step 间隔。",
        "- 探针 run ID、协议、最后一轮数值与 results.json 一致；序列 epoch 与探针对齐。",
        "- 所有 CKA 和物品几何状态 valid；序列统计没有跳过的 batch。",
    ]
    for model, status in baselines.items():
        lines.append(
            f"- {model.upper()} 的 current/next 权重 0 基线："
            f"推荐结果一致={status['recommendation_equal']}，"
            f"逐 epoch 导出指标一致={status['epoch_metrics_equal']}。两条曲线重合，仍分别保存。"
        )
    for run in runs:
        if run["identity"]["model"] == "qwen3":
            lines.append(
                f"- {run['identity']['experiment']}：{len(run['records'])} 个物品探针点，"
                f"{len(run['sequence_records'])} 个序列 epoch；"
                f"最佳 step={run['best']['recommendation_best_step']}，"
                f"该点有探针={run['best']['probe_at_best_step_available']}；"
                f"裸 SID 点数={sum('raw_sid_cka' in r for r in run['records'])}。"
            )
    lines += [
        "",
        "## 重新生成",
        "",
        "```bash",
        f"python {OUT / 'summarize.py'}",
        "```",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def plot_qwen_raw(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in zip(
        axes, ("raw_sid_cka", "raw_sid_cka_genrec"),
        ("Debiased cosine-Gram CKA", "genrec linear CKA"),
    ):
        for run in runs:
            if run["identity"]["model"] != "qwen3":
                continue
            rows = [r for r in run["records"] if key in r]
            if not rows:
                continue
            ident = run["identity"]
            color, style, legend = STYLES[(ident["align_item"], ident["align_weight"])]
            assert all(math.isfinite(r[key]) for r in rows)
            ax.plot([r["global_step"] for r in rows], [r[key] for r in rows],
                    color=color, ls=style, label=legend, marker=".", ms=4)
        ax.set(xlabel="Optimizer step", ylabel=label, title=label)
        ax.grid(alpha=0.2)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Beauty / Qwen3-1.7B: raw three-SID item CKA")
    fig.text(0.5, 0.025,
             "Last SID token, final decoder layer; no prompt/BOS/EOS; no projection.\n"
             "Baseline has no raw-SID records. No smoothing or interpolation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(OUT / "cka/qwen3_raw_sid_cka.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    runs, manifest = collect()
    baselines = {}
    for model in MODELS:
        current, nxt = [
            run for run in runs
            if run["identity"]["model"] == model and run["identity"]["align_weight"] == 0
        ]
        baselines[model] = {
            "recommendation_equal": current["result"]["test"] == nxt["result"]["test"],
            "epoch_metrics_equal": current["records"] == nxt["records"],
        }
    for item, weight in (("current", 0), ("current", 5), ("next", 5)):
        qwen, qwen_manifest = collect_qwen(item, weight)
        runs.append(qwen)
        manifest.append(qwen_manifest)
    csv_write(OUT / "qwen3_sequence_epoch_metrics.csv", [
        {**run["identity"], **row}
        for run in runs if run["identity"]["model"] == "qwen3"
        for row in run["sequence_records"]
    ])
    csv_write(OUT / "summary_best_epoch.csv", [run["best"] for run in runs])
    csv_write(OUT / "summary_last_epoch.csv", [run["last"] for run in runs])
    csv_write(OUT / "epoch_metrics.csv", [
        {**run["identity"], **record} for run in runs for record in run["records"]
    ])
    json_write(OUT / "summary.json", {
        "best_recommender_epoch": [run["best"] for run in runs],
        "last_epoch": [run["last"] for run in runs],
        "baseline_checks": baselines,
    })
    json_write(OUT / "manifest.json", manifest)
    documentation(runs, baselines)
    acc = ("probe_test_acc", "Category test accuracy", True)
    cka = ("cka", "Debiased cosine-Gram CKA", False)
    genrec = ("cka_genrec", "genrec linear CKA", False)
    plot(runs, [acc], "Beauty: category linear probe", OUT / "acc/beauty_category_acc")
    plot(runs, [cka, genrec], "Beauty: two item CKA definitions",
         OUT / "cka/beauty_cka_comparison")
    plot(runs, [acc, cka, genrec], "Beauty: item accuracy and CKA",
         OUT / "overview/beauty_acc_cka")
    plot(runs, [("item_effective_rank", "Item effective rank", False),
                ("sequence_effective_rank", "Sequence effective rank (batch mean)", False)],
         "Beauty: effective rank", OUT / "geometry/beauty_effective_rank")
    plot(runs, [("item_avg_cosine", "Item mean pairwise cosine", False),
                ("sequence_avg_cosine", "Sequence pairwise cosine (batch mean)", False)],
         "Beauty: hidden geometry cosine", OUT / "geometry/beauty_avg_cosine")
    plot(runs, [("probe_test_cosine", "Regression test cosine", False),
                ("probe_test_mse", "Regression test MSE", False)],
         "Beauty: hidden-to-Sentence-T5 affine probe", OUT / "probe/beauty_regression")
    plot(runs, [(f"probe_test_Recall@{k}", f"Probe retrieval Recall@{k}", True)
                for k in (10, 20, 50, 100)],
         "Beauty: probe retrieval against the full item catalogue",
         OUT / "probe/beauty_retrieval")
    plot_qwen_raw(runs)
    print(json.dumps({
        "output": str(OUT), "training_runs": len(runs) - len(VARIANTS),
        "representation_runs": len(runs),
        "epoch_rows": sum(len(run["records"]) for run in runs),
        "baseline_checks": baselines,
    }, indent=2))


if __name__ == "__main__":
    main()
