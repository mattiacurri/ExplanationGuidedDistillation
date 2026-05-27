from __future__ import annotations

import json
import itertools
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
SCORED = ROOT / "runs" / "exps" / "scored_reports_test_5pc"
FIGURES = REPORT / "figures"


VARIANT_LABELS = {
    "base_vitbase_mse": "Base MSE",
    "mean_vitbase_expl_only": "Mean\nexpl.",
    "attn_vitbase_expl_only": "Attention\nexpl.",
    "cross_vitbase_expl_only": "Cross-attn\nexpl.",
    "mean_vitbase_mse_expl": "Mean\nMSE+expl.",
    "attn_vitbase_mse_expl": "Attention\nMSE+expl.",
    "cross_vitbase_mse_expl": "Cross-attn\nMSE+expl.",
}

DISPLAY_ORDER = [
    "base_vitbase_mse",
    "mean_vitbase_expl_only",
    "mean_vitbase_mse_expl",
    "attn_vitbase_expl_only",
    "attn_vitbase_mse_expl",
    "cross_vitbase_expl_only",
    "cross_vitbase_mse_expl",
]


@dataclass
class VariantMetrics:
    key: str
    label: str
    vision_valid: int
    vision_student: int
    vision_teacher: int
    vision_no_majority: int
    vision_agreement: float
    vision_pairwise: float
    vision_kappa: float
    text_valid: int
    text_student: int
    text_teacher: int
    text_no_majority: int
    text_agreement: float


@dataclass
class ProbeMetrics:
    key: str
    label: str
    best_epoch: int
    train_acc: float
    val_acc: float
    best_val_acc: float
    test_acc: float
    val_loss: float
    test_loss: float


def _variant_key(path: Path) -> str:
    name = path.stem
    prefix = "qwen_report_"
    suffix = "_judged"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _read_metrics() -> list[VariantMetrics]:
    rows: list[VariantMetrics] = []
    for path in sorted(SCORED.glob("qwen_report_*_judged.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        key = _variant_key(path)
        vision = data["multi_judge_vision_blind_valid3"]
        text = data["multi_judge_blind_valid3"]
        vision_counts = vision["majority_counts"]
        text_counts = text["majority_counts"]
        rows.append(
            VariantMetrics(
                key=key,
                label=VARIANT_LABELS.get(key, key.replace("_", " ")),
                vision_valid=vision["num_items"],
                vision_student=vision_counts.get("student_win", 0),
                vision_teacher=vision_counts.get("teacher_win", 0),
                vision_no_majority=vision_counts.get("no_majority", 0),
                vision_agreement=vision["all_judges_agreement"],
                vision_pairwise=vision["mean_pairwise_agreement"],
                vision_kappa=vision["fleiss"]["kappa"],
                text_valid=text["num_items"],
                text_student=text_counts.get("student_win", 0),
                text_teacher=text_counts.get("teacher_win", 0),
                text_no_majority=text_counts.get("no_majority", 0),
                text_agreement=text["all_judges_agreement"],
            )
        )
    return rows


def _image_key(item: dict[str, object]) -> tuple[object, ...]:
    return (
        item.get("split"),
        item.get("dataset_index"),
        item.get("source_index"),
        item.get("label"),
    )


def _majority_vote(votes: list[str]) -> str:
    counts = Counter(votes)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "no_majority"
    return ranked[0][0]


def _agreement_metrics(vote_rows: list[dict[str, str]]) -> tuple[float, float, float]:
    judges = list(vote_rows[0])
    all_agree = sum(len(set(votes.values())) == 1 for votes in vote_rows)
    pairs = list(itertools.combinations(judges, 2))
    pairwise = sum(
        sum(votes[left] == votes[right] for votes in vote_rows) / len(vote_rows)
        for left, right in pairs
    ) / len(pairs)

    categories = ("student_win", "teacher_win", "tie")
    totals: Counter[str] = Counter()
    agreement_per_image: list[float] = []
    for votes in vote_rows:
        counts = Counter(votes.values())
        totals.update(counts)
        agreement_per_image.append(
            (sum(count * count for count in counts.values()) - len(judges))
            / (len(judges) * (len(judges) - 1))
        )
    p_bar = sum(agreement_per_image) / len(agreement_per_image)
    total_ratings = len(vote_rows) * len(judges)
    p_e = sum((totals[category] / total_ratings) ** 2 for category in categories)
    kappa = 0.0 if p_e >= 1.0 else (p_bar - p_e) / (1.0 - p_e)
    return all_agree / len(vote_rows), pairwise, kappa


def _read_shared_valid_primary_metrics(
    rows: list[VariantMetrics],
) -> list[VariantMetrics]:
    payloads: dict[str, dict[str, object]] = {}
    valid_keys: list[set[tuple[object, ...]]] = []
    for path in sorted(SCORED.glob("qwen_report_*_judged.json")):
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        key = _variant_key(path)
        payloads[key] = data
        vision = data["multi_judge_vision_blind_valid3"]
        valid_keys.append(
            {_image_key(data["results"][idx]) for idx in vision["complete_row_indices"]}
        )

    shared_keys = set.intersection(*valid_keys)
    updated: list[VariantMetrics] = []
    for row in rows:
        data = payloads[row.key]
        judges = data["multi_judge_vision_blind_valid3"]["judges"]
        results_by_key = {_image_key(item): item for item in data["results"]}
        counts: Counter[str] = Counter()
        vote_rows: list[dict[str, str]] = []
        for image_key in shared_keys:
            item = results_by_key[image_key]
            votes = {
                judge: item["vlm_judges"][judge]["vision_blind"]["winner"]
                for judge in judges
            }
            vote_rows.append(votes)
            counts[_majority_vote(list(votes.values()))] += 1
        agreement, pairwise, kappa = _agreement_metrics(vote_rows)
        updated.append(
            replace(
                row,
                vision_valid=len(shared_keys),
                vision_student=counts.get("student_win", 0),
                vision_teacher=counts.get("teacher_win", 0),
                vision_no_majority=(
                    counts.get("tie", 0) + counts.get("no_majority", 0)
                ),
                vision_agreement=agreement,
                vision_pairwise=pairwise,
                vision_kappa=kappa,
            )
        )
    return updated


def _read_probe_metrics() -> list[ProbeMetrics]:
    wanted = {
        "runs\\vlm_qwen25vl3b_probe_mean_full": ("mean", "Mean"),
        "runs\\vlm_qwen25vl3b_probe_attention_full": ("attention", "Attention"),
        "runs\\vlm_qwen25vl3b_probe_cross_attention_full": ("cross", "Cross-attn"),
    }
    by_output: dict[str, ProbeMetrics] = {}
    for path in sorted((ROOT / "wandb").glob("run-*/files/wandb-summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        output = data.get("output_dir")
        if output not in wanted:
            continue
        key, label = wanted[output]
        metric = ProbeMetrics(
            key=key,
            label=label,
            best_epoch=int(data["best_epoch"]),
            train_acc=float(data["train/accuracy"]),
            val_acc=float(data["validation/accuracy"]),
            best_val_acc=float(data["best_validation_accuracy"]),
            test_acc=float(data["test_accuracy"]),
            val_loss=float(data["validation/loss"]),
            test_loss=float(data["test/loss"]),
        )
        # Duplicate W&B folders can exist after resumed inspection; the values are identical.
        by_output[output] = metric
    return [by_output[key] for key in wanted if key in by_output]


def _read_distillation_losses() -> list[dict[str, float | str]]:
    csv_path = ROOT / "runs" / "exps" / "retro_val_test_losses.csv"
    rows: list[dict[str, float | str]] = []
    if not csv_path.exists():
        return rows
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    for line in lines[1:]:
        values = line.split(",")
        item: dict[str, float | str] = dict(zip(header, values))
        if item.get("checkpoint") != "best":
            continue
        for key in header:
            if key not in {"student", "checkpoint"}:
                item[key] = float(item[key])
        rows.append(item)
    order = [
        "student_base_vitbase_mse",
        "student_mean_vitbase_expl_only",
        "student_mean_vitbase_mse_expl",
        "student_attn_vitbase_expl_only",
        "student_attn_vitbase_mse_expl",
        "student_cross_vitbase_expl_only",
        "student_cross_vitbase_mse_expl",
    ]
    rows_by_key = {str(row["student"]): row for row in rows}
    return [rows_by_key[key] for key in order if key in rows_by_key]


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_primary_preference(rows: list[VariantMetrics]) -> None:
    by_key = {row.key: row for row in rows}
    rows = [by_key[key] for key in DISPLAY_ORDER if key in by_key]
    labels = [row.label for row in rows]
    rates = [100.0 * row.vision_student / row.vision_valid for row in rows]
    counts = [row.vision_student for row in rows]
    colors = ["#777777"] + [
        "#0072B2",
        "#56B4E9",
        "#0072B2",
        "#56B4E9",
        "#0072B2",
        "#56B4E9",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = range(len(rows))
    bars = ax.bar(x, rates, color=colors, edgecolor="#333333", linewidth=0.6)
    baseline = rates[0]
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.0)
    for bar, count, rate in zip(bars, counts, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.45,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Student majority preference (%)")
    ax.set_ylim(0, 20)
    ax.set_title("Student wins under image-grounded judging", fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "primary_preference_valid3")


def plot_validity_and_agreement(rows: list[VariantMetrics]) -> None:
    image_valid = [100.0 * row.vision_valid / 500 for row in rows]
    text_valid = [100.0 * row.text_valid / 500 for row in rows]
    image_agreement = [row.vision_pairwise for row in rows]
    text_agreement = [row.text_agreement for row in rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    labels = ["image-\ngrounded", "text-\nonly"]
    colors = ["#009E73", "#CC79A7"]

    valid_means = [
        sum(image_valid) / len(image_valid),
        sum(text_valid) / len(text_valid),
    ]
    valid_err = [
        [valid_means[0] - min(image_valid), valid_means[1] - min(text_valid)],
        [max(image_valid) - valid_means[0], max(text_valid) - valid_means[1]],
    ]
    ax0.bar(
        labels,
        valid_means,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
        yerr=valid_err,
        capsize=4,
    )
    ax0.set_ylim(0, 105)
    ax0.set_ylabel("Usable judgments (%)")
    ax0.set_title("Structured outputs", fontsize=10)
    ax0.grid(axis="y", color="#dddddd", linewidth=0.7)

    agree_means = [
        sum(image_agreement) / len(image_agreement),
        sum(text_agreement) / len(text_agreement),
    ]
    agree_err = [
        [agree_means[0] - min(image_agreement), agree_means[1] - min(text_agreement)],
        [max(image_agreement) - agree_means[0], max(text_agreement) - agree_means[1]],
    ]
    ax1.bar(
        labels,
        agree_means,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
        yerr=agree_err,
        capsize=4,
    )
    ax1.set_ylim(0.45, 1.0)
    ax1.set_ylabel("Judge agreement")
    ax1.set_title("Preference consistency", fontsize=10)
    ax1.grid(axis="y", color="#dddddd", linewidth=0.7)

    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Image-grounded judging is the cleaner evaluation signal", fontsize=11)
    fig.tight_layout()
    _save(fig, "validity_agreement")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 2.55))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_width = 0.145
    box_height = 0.26
    box_y = 0.55 - box_height / 2

    left = 0.03
    gap = (1.0 - left * 2 - 5 * box_width) / 4
    x_positions = [left + i * (box_width + gap) for i in range(5)]
    boxes = [
        ("Teacher VLM\nvisual tokens", x_positions[0], "#F4F7FB"),
        ("Grad-CAM\nsaliency", x_positions[1], "#FBF6EC"),
        ("Explanation-guided\nmatching", x_positions[2], "#EEF7F1"),
        ("Student ViT\nvisual module", x_positions[3], "#F7F1FA"),
        ("Plug-in VLM\nevaluation", x_positions[4], "#FDF0F0"),
    ]
    for text, x, color in boxes:
        patch = FancyBboxPatch(
            (x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.016,rounding_size=0.012",
            linewidth=0.9,
            edgecolor="#4A4A4A",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + box_width / 2,
            box_y + box_height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9.15,
        )

    arrows = [
        ((x_positions[0] + box_width, 0.55), (x_positions[1], 0.55)),
        ((x_positions[1] + box_width, 0.55), (x_positions[2], 0.55)),
        ((x_positions[2] + box_width, 0.55), (x_positions[3], 0.55)),
        ((x_positions[3] + box_width, 0.55), (x_positions[4], 0.55)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10.5,
                linewidth=0.9,
                color="#4A4A4A",
            )
        )
    fig.tight_layout(pad=0.05)
    _save(fig, "pipeline_overview")


def plot_probe_results(rows: list[ProbeMetrics]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    labels = [row.label for row in rows]
    x = range(len(rows))
    width = 0.32
    val = [100 * row.best_val_acc for row in rows]
    test = [100 * row.test_acc for row in rows]
    ax.bar(
        [i - width / 2 for i in x],
        val,
        width,
        label="validation",
        color="#0077BB",
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.bar(
        [i + width / 2 for i in x],
        test,
        width,
        label="test",
        color="#33BBEE",
        edgecolor="#333333",
        linewidth=0.6,
    )
    for i, row in enumerate(rows):
        ax.text(
            i,
            max(val[i], test[i]) + 0.18,
            f"epoch {row.best_epoch}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(93.5, 95.6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Teacher probes provide high-accuracy saliency sources", fontsize=11)
    ax.set_xticks(list(x), labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "probe_results")


def plot_distillation_losses(rows: list[dict[str, float | str]]) -> None:
    labels = [
        "Base\nMSE",
        "Mean\nexpl.",
        "Mean\nMSE+expl.",
        "Attention\nexpl.",
        "Attention\nMSE+expl.",
        "Cross-attn\nexpl.",
        "Cross-attn\nMSE+expl.",
    ]
    x = range(len(rows))
    global_loss = [float(row["test_loss_global"]) for row in rows]
    expl_loss = [float(row["test_loss_expl"]) for row in rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax0.bar(x, global_loss, color="#0077BB", edgecolor="#333333", linewidth=0.6)
    ax0.set_title("Global MSE component", fontsize=10)
    ax0.set_ylabel("Test loss")
    ax0.set_xticks(list(x), labels, rotation=35, ha="right")
    ax0.grid(axis="y", color="#dddddd", linewidth=0.7)

    ax1.bar(x, expl_loss, color="#EE7733", edgecolor="#333333", linewidth=0.6)
    ax1.set_title("Explanation-weighted component", fontsize=10)
    ax1.set_ylabel("Test loss")
    ax1.set_xticks(list(x), labels, rotation=35, ha="right")
    ax1.grid(axis="y", color="#dddddd", linewidth=0.7)

    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Representation-level distillation losses separate the objectives", fontsize=11
    )
    fig.tight_layout()
    _save(fig, "distillation_losses")


def main() -> None:
    rows = _read_metrics()
    shared_primary_rows = _read_shared_valid_primary_metrics(rows)
    probe_rows = _read_probe_metrics()
    distill_rows = _read_distillation_losses()
    plot_primary_preference(shared_primary_rows)
    plot_validity_and_agreement(rows)
    plot_pipeline()
    plot_probe_results(probe_rows)
    plot_distillation_losses(distill_rows)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
