"""Markdown report rendering."""

from pathlib import Path
from typing import Any, List


def save_markdown_report(result: Any, path: str) -> None:
    """Save a benchmark result as a Markdown report.

    Args:
        result: Result object renderable by `render_markdown_report`.
        path: Destination file path.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown_report(result), encoding="utf-8")


def render_markdown_report(result: Any) -> str:
    """Render a benchmark result as Markdown.

    Args:
        result: Benchmark result object.

    Returns:
        Markdown report text.
    """

    data = result.to_dict()
    lines: List[str] = ["# vertebrae benchmark report", ""]
    dataset = data["dataset_summary"]
    lines.extend(
        [
            "## Dataset summary",
            "",
            f"- Samples: {dataset['n_samples']}",
            f"- Classes: {dataset['n_classes']}",
            f"- Modality: {dataset['modality']}",
            "",
            "## Executive summary",
            "",
        ]
    )
    for item in data.get("recommendations", []):
        lines.append(f"- {item}")
    top_ranked = result.ranked_results()[0] if result.extractor_results else None
    if top_ranked and top_ranked.separatix and top_ranked.separatix.ran:
        lines.append(
            "- "
            f"Separatix complexity guidance for the top representation: "
            f"{top_ranked.separatix.recommendation or ''} "
            f"({top_ranked.separatix.confidence or ''} confidence).".strip()
        )
    lines.extend(["", "## Ranking", ""])
    lines.append(
        "| rank | extractor | extractor_type | overlap_macro | stability_interval | "
        "weakest_class | probe_accuracy | embedding_dim | compression | "
        "compressed_dim | recommendation | separatix_recommendation | "
        "separatix_confidence |"
    )
    lines.append(
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |"
    )
    for rank, item in enumerate(result.ranked_results(), start=1):
        interval = _format_interval(item.stability)
        weakest = item.weakest_class if item.weakest_class is not None else ""
        probe_accuracy = _best_probe_accuracy(item.probes)
        embedding_dim = item.embedding_metadata.get("embedding_dim", "")
        compression_method = item.compression_metadata.get("method", "none")
        compressed_dim = item.compression_metadata.get("compressed_dim", embedding_dim)
        separatix_recommendation = ""
        separatix_confidence = ""
        if item.separatix:
            separatix_recommendation = item.separatix.recommendation or ""
            separatix_confidence = item.separatix.confidence or ""
        lines.append(
            f"| {rank} | {item.name} | {item.extractor_type} | "
            f"{item.overlap.macro_score:.4f} | {interval} | {weakest} | "
            f"{probe_accuracy} | {embedding_dim} | {compression_method} | "
            f"{compressed_dim} | {item.recommendation} | {separatix_recommendation} | "
            f"{separatix_confidence} |"
        )

    lines.extend(["", "## Per-extractor details", ""])
    for item in result.ranked_results():
        lines.extend(
            [
                f"### {item.name}",
                "",
                f"- Extractor type: {item.extractor_type}",
                f"- Extractor family: {_extractor_family(item.extractor_type)}",
                f"- Modality: {item.embedding_metadata.get('modality', '')}",
                f"- Embedding dimension: {item.embedding_metadata.get('embedding_dim', '')}",
                f"- Compression method: {item.compression_metadata.get('method', 'none')}",
                f"- Compression precision: {item.compression_metadata.get('precision', '')}",
                f"- Compressed dimension: {item.compression_metadata.get('compressed_dim', '')}",
                f"- Overlap macro: {item.overlap.macro_score:.4f}",
                f"- Weakest class: {item.weakest_class or ''}",
                f"- Recommendation: {item.recommendation}",
                "",
                "#### Recipe summary",
                "",
            ]
        )
        recipe = item.embedding_metadata.get("recipe") or item.embedding_metadata.get(
            "extractor_recipe",
            {},
        )
        if recipe:
            for key, value in recipe.items():
                if key in {"params"} and isinstance(value, dict):
                    lines.append(f"- {key}: {len(value)} captured parameters")
                else:
                    lines.append(f"- {key}: {value}")
        else:
            lines.append("No extractor recipe was captured.")
        lines.append("")
        lines.extend(["#### Compression", ""])
        compression_metadata = item.compression_metadata or {}
        for key in (
            "method",
            "precision",
            "applied",
            "original_dim",
            "compressed_dim",
            "explained_variance_total",
        ):
            if key in compression_metadata:
                lines.append(f"- {key}: {compression_metadata[key]}")
        for warning in compression_metadata.get("warnings", []):
            lines.append(f"- warning: {warning}")
        lines.extend(
            [
                "",
                "#### Per-class scores",
                "",
            ]
        )
        if item.overlap.per_class_scores:
            lines.append("| class | score |")
            lines.append("| --- | ---: |")
            for label, score in item.overlap.per_class_scores.items():
                lines.append(f"| {label} | {_format_float(score)} |")
            lines.append("")
        else:
            lines.append("No per-class scores were returned.")
            lines.append("")

        lines.extend(["#### Stability analysis", ""])
        if item.stability:
            summary = item.stability.get("summary", {})
            lines.append(
                "- "
                f"{item.stability.get('mode')} stability interval: "
                f"{_format_float(summary.get('lower'))} to {_format_float(summary.get('upper'))}; "
                f"mean {_format_float(summary.get('mean'))}."
            )
        else:
            lines.append("Stability analysis was not run.")
        lines.append("")

        lines.extend(["#### Probe results", ""])
        if item.probes and item.probes.get("enabled"):
            lines.append("| method | accuracy | macro_f1 | balanced_accuracy |")
            lines.append("| --- | ---: | ---: | ---: |")
            for method, scores in item.probes.get("results", {}).items():
                lines.append(
                    f"| {method} | {_format_float(scores.get('accuracy'))} | "
                    f"{_format_float(scores.get('macro_f1'))} | "
                    f"{_format_float(scores.get('balanced_accuracy'))} |"
                )
        else:
            lines.append("Probe evaluation was not run.")
        lines.append("")

        lines.extend(["#### Separatix complexity diagnostic", ""])
        if item.separatix is None:
            lines.append("Separatix diagnostics were disabled.")
        elif not item.separatix.ran:
            lines.append(f"- Skipped: {item.separatix.skipped_reason or ''}")
        else:
            lines.append(f"- Recommendation: {item.separatix.recommendation or ''}")
            lines.append(f"- Confidence: {item.separatix.confidence or ''}")
            lines.append(
                f"- Summary: {(item.separatix.recommendation_text or '').strip()}"
            )
            if item.separatix.decision_path:
                lines.append("- Decision path:")
                for step in item.separatix.decision_path:
                    lines.append(f"  - {step}")
            if item.separatix.scores:
                lines.append("")
                lines.append("| score | value |")
                lines.append("| --- | ---: |")
                for key, value in item.separatix.scores.items():
                    lines.append(f"| {key} | {_format_float(value)} |")
            if item.separatix.skipped_diagnostics:
                lines.append("")
                lines.append("- Skipped diagnostics:")
                for entry in item.separatix.skipped_diagnostics:
                    lines.append(
                        f"  - {entry.get('name', '')}: {entry.get('reason', '')}"
                    )
            if item.separatix.warnings:
                lines.append("")
                lines.append("- Warnings:")
                for warning in item.separatix.warnings:
                    lines.append(f"  - {warning}")
        lines.append("")

        warnings = item.warnings
        if warnings:
            lines.extend(["#### Warnings", ""])
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

    lines.extend(["## OverlapIndex configuration", ""])
    scoring_config = data.get("metadata", {}).get("scoring_config", {})
    for key, value in scoring_config.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Reproducibility metadata", ""])
    for key, value in data.get("metadata", {}).items():
        if key != "scoring_config":
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _format_interval(stability: Any) -> str:
    if not stability:
        return ""
    summary = stability.get("summary", {})
    return f"{_format_float(summary.get('lower'))}-{_format_float(summary.get('upper'))}"


def _best_probe_accuracy(probes: Any) -> str:
    if not probes or not probes.get("enabled"):
        return ""
    accuracies = [
        scores.get("accuracy")
        for scores in probes.get("results", {}).values()
        if scores.get("accuracy") is not None
    ]
    if not accuracies:
        return ""
    return f"{max(accuracies):.4f}"


def _format_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return ""


def _extractor_family(extractor_type: str) -> str:
    families = {
        "frozen_pretrained": "frozen pretrained backbone",
        "unsupervised_fitted": "fitted sklearn pipeline",
        "supervised_fitted": "fitted sklearn pipeline",
        "custom_callable": "custom callable extractor",
        "precomputed": "precomputed embeddings",
    }
    return families.get(extractor_type, extractor_type)
