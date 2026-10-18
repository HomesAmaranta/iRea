"""Plot the three historical Beauty GRHSTU CKA runs; PDF output only."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE = Path("/opt/tiger/project/genrec/out/grhstu/amazon/beauty")
OUTPUT = Path(__file__).resolve().parent / "cka/genrec_grhstu_beauty_cka.pdf"
RUNS = (
    ("grhstu_current_latent_0_lr0.0001_seed42_len20_20261014_234509.log",
     "Baseline (weight 0)", "#7b8490", "--"),
    ("grhstu_current_latent_5_lr0.0001_seed42_len20_20261014_234514.log",
     "Current latent alignment (weight 5)", "#1683de", "-"),
    ("grhstu_next_latent_5_lr0.0001_seed42_len20_20261015_001510.log",
     "Next latent alignment (weight 5)", "#8155ce", "-"),
)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for filename, label, color, style in RUNS:
        text = (SOURCE / filename).read_text()
        matches = re.findall(r"Epoch (\d+) - item_cka: ([\d.eE+-]+)", text)
        epochs = [int(epoch) for epoch, _ in matches]
        scores = [float(value) for _, value in matches]
        if not epochs or epochs != list(range(epochs[-1] + 1)):
            raise ValueError(f"Incomplete or duplicate epochs in {filename}")
        ax.plot(epochs, scores, label=label, color=color, ls=style, lw=1.8)
        ax.scatter(epochs[-1], scores[-1], color=color, s=25, zorder=3)
        ax.annotate(f"{scores[-1]:.4f}", (epochs[-1], scores[-1]),
                    xytext=(6, 9 if style == "--" else -13),
                    textcoords="offset points", fontsize=9, color=color)
        print(f"{label}: {len(epochs)} epochs, {scores[0]:.6f} -> {scores[-1]:.6f}")
    ax.set_title("Beauty / GRHSTU: item CKA during training", loc="left", fontsize=15)
    ax.set_xlabel("Epoch (0-based)")
    ax.set_ylabel("Centered linear CKA")
    ax.set_ylim(0, 1)
    ax.margins(x=0.08)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.text(
        0.5, 0.025,
        "genrec historical runs | learning rate 1e-4 | seed 42 | full catalogue\n"
        "Isolated 3-SID item, last token after final_norm vs original item embeddings.\n"
        "No smoothing; original epoch coverage. Dots and labels indicate final recorded values.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
