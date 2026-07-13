"""Markdown report rendering."""

from pathlib import Path
from typing import Any, List

from vertebrae.scoring.separatix import probe_summary_for_result


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
    target_type = dataset.get("target_type", "single_label")
    lines.extend(
        [
            "## Dataset summary",
            "",
            f"- Samples: {dataset.get('n_samples', dataset.get('n_images', ''))}",
            (
                f"- Targets: {dataset.get('n_targets', '')}"
                if target_type == "regression"
                else f"- Classes: {dataset.get('n_classes', dataset.get('n_classes_raw', ''))}"
            ),
            f"- Target type: {target_type}",
            f"- Modality: {dataset['modality']}",
        ]
    )
    if dataset.get("target_view"):
        lines.append(f"- Target view: {dataset.get('target_view', {}).get('name', 'primary')}")
    if dataset.get("available_target_views"):
        target_view_names = [item.get("name") for item in dataset.get("available_target_views", [])]
        lines.append(f"- Available target views: {target_view_names}")
    if dataset.get("modality") == "segmentation":
        lines.append(f"- Source images: {dataset.get('n_images', '')}")
        lines.append(
            "- Interpretation: dense semantic representation separation; "
            "this is not IoU, mask accuracy, or boundary accuracy."
        )
    if target_type == "multi_label":
        lines.append(
            "- " f"Mean label cardinality: {_format_float(dataset.get('mean_label_cardinality'))}"
        )
        lines.append(f"- Label density: {_format_float(dataset.get('label_density'))}")
    if target_type == "regression":
        lines.append(f"- Target names: {dataset.get('target_names', [])}")
        constant_targets = dataset.get("constant_targets", [])
        if constant_targets:
            lines.append(f"- Constant targets: {constant_targets}")
    dataset_metadata = dataset.get("metadata", {})
    if dataset_metadata.get("modalities"):
        lines.append(f"- Modalities: {dataset_metadata['modalities']}")
    if dataset_metadata.get("input_fields"):
        lines.append(f"- Input fields: {dataset_metadata['input_fields']}")
    if dataset_metadata.get("relational_unit"):
        lines.append(f"- Relational unit: {dataset_metadata['relational_unit']}")
    units = dataset.get("units")
    if units:
        lines.append(f"- Unit type: {units.get('unit_type', 'unit')}")
    structured_units = dataset.get("structured_units")
    if structured_units:
        lines.append(f"- Structured unit type: {structured_units.get('unit_type', 'unit')}")
        if structured_units.get("task_family"):
            lines.append(f"- Structured task family: {structured_units.get('task_family')}")
        lines.append(f"- Structured parents: {structured_units.get('n_parents', '')}")
        lines.append(f"- Structured units: {structured_units.get('n_units', '')}")
    if dataset_metadata.get("entity_type"):
        lines.append(f"- Entity type: {dataset_metadata['entity_type']}")
    if dataset_metadata.get("composition"):
        lines.append(f"- Embedding composition: {dataset_metadata['composition']}")
    lines.extend(
        [
            "",
            "## Executive summary",
            "",
        ]
    )
    for item in data.get("recommendations", []):
        lines.append(f"- {item}")
    for warning in data.get("metadata", {}).get("label_view_warnings", []):
        lines.append(f"- {warning}")
    for warning in data.get("metadata", {}).get("target_view_warnings", []):
        lines.append(f"- {warning}")
    top_ranked = result.ranked_results()[0] if result.extractor_results else None
    if top_ranked and top_ranked.separatix and top_ranked.separatix.ran:
        lines.append(
            "- "
            f"Separatix complexity guidance for the top representation: "
            f"{top_ranked.separatix.recommendation or ''} "
            f"({top_ranked.separatix.confidence or ''} confidence).".strip()
        )
    structured_outputs = dataset.get("structured_outputs", [])
    if structured_outputs:
        lines.extend(["", "## Structured outputs", ""])
        lines.append(
            "| extractor | output | unit_type | task_family | alignment_mode | alignment_recipe |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for output in structured_outputs:
            lines.append(
                f"| {output.get('extractor', '')} | {output.get('output', '')} | "
                f"{output.get('unit_type', '')} | {output.get('task_family', '')} | "
                f"{output.get('alignment_mode', '')} | "
                f"{_alignment_recipe_label(output.get('alignment_recipe'))} |"
            )
    lines.extend(["", "## Ranking", ""])
    lines.append(
        "| rank | extractor | extractor_type | target_view | label_view | "
        "primary_metric | primary_score | overlap_score | overlap_macro | "
        "overlap_weighted | stability_interval | "
        "weakest_class | best_probe | probe_metric | probe_score | embedding_dim | compression | "
        "compressed_dim | recommendation | separatix_recommendation | "
        "separatix_confidence |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | "
        "--- | --- | ---: | ---: | --- | ---: | --- | --- | --- |"
    )
    for rank, item in enumerate(result.ranked_results(), start=1):
        interval = _format_interval(item.stability)
        weakest = item.weakest_class if item.weakest_class is not None else ""
        probe = probe_summary_for_result(item.separatix)
        primary_probe_metric = probe.get("primary_metric") or {}
        embedding_dim = item.embedding_metadata.get("embedding_dim", "")
        target_view = (item.target_view or {}).get("name", "primary")
        label_view = (item.label_view or {}).get("name", "primary")
        compression_method = item.compression_metadata.get("method", "none")
        compressed_dim = item.compression_metadata.get("compressed_dim", embedding_dim)
        separatix_recommendation = ""
        separatix_confidence = ""
        if item.separatix:
            separatix_recommendation = item.separatix.recommendation or ""
            separatix_confidence = item.separatix.confidence or ""
        lines.append(
            f"| {rank} | {item.name} | {item.extractor_type} | "
            f"{target_view} | {label_view} | {item.primary_metric_name} | "
            f"{_format_float(item.primary_score)} | "
            f"{_format_float(item.overlap.score if item.overlap else None)} | "
            f"{_format_float(item.overlap.macro_score if item.overlap else None)} | "
            f"{_format_float(item.overlap.weighted_score if item.overlap else None)} | "
            f"{interval} | {weakest} | "
            f"{probe.get('best_probe') or ''} | {primary_probe_metric.get('name') or ''} | "
            f"{_format_float(primary_probe_metric.get('value'))} | "
            f"{embedding_dim} | {compression_method} | "
            f"{compressed_dim} | {item.recommendation} | {separatix_recommendation} | "
            f"{separatix_confidence} |"
        )

    profiled_cohort = [item for item in result.quality_cohort() if item.resource_profile]
    if profiled_cohort:
        lines.extend(["", "## Resource profile for quality-similar candidates", ""])
        lines.append(
            "| extractor | first call ms | warm median ms | warm p95 ms | samples/s | "
            "peak host increase | peak device allocated | model bytes | checkpoint bytes | "
            "embedding bytes | batch sizes | sync |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        for item in profiled_cohort:
            profile = item.resource_profile
            assert profile is not None
            inference = profile.inference
            evaluated_bytes = profile.embedding.evaluated_bytes if profile.embedding else None
            lines.append(
                f"| {item.name} | {_format_milliseconds(inference.first_call_seconds)} | "
                f"{_format_milliseconds(inference.warm_median_seconds)} | "
                f"{_format_milliseconds(inference.warm_p95_seconds)} | "
                f"{_format_float(inference.throughput_samples_per_second)} | "
                f"{_format_bytes(profile.host_memory.peak_increase_bytes)} | "
                f"{_format_bytes(profile.device_memory.peak_allocated_bytes)} | "
                f"{_format_bytes(profile.model.in_memory_bytes)} | "
                f"{_format_bytes(profile.model.checkpoint_bytes)} | "
                f"{_format_bytes(evaluated_bytes)} | "
                f"{inference.batch_sizes} | {profile.context.get('synchronization_status', '')} |"
            )

    lines.extend(["", "## Per-extractor details", ""])
    for item in result.ranked_results():
        overlap = item.overlap
        target_metadata = (
            overlap.metadata if overlap else item.metrics[item.primary_metric_name].metadata
        )
        lines.extend(
            [
                f"### {item.name}",
                "",
                f"- Extractor type: {item.extractor_type}",
                f"- Extractor family: {_extractor_family(item.extractor_type)}",
                f"- Target type: {target_metadata.get('target_type', 'single_label')}",
                f"- Target view: {(item.target_view or {}).get('name', 'primary')}",
                f"- Label view: {(item.label_view or {}).get('name', 'primary')}",
                f"- Modality: {item.embedding_metadata.get('modality', '')}",
                (
                    "- Output source: "
                    f"{item.embedding_metadata.get('output_metadata', {}).get('source', '')}"
                ),
                f"- Embedding dimension: {item.embedding_metadata.get('embedding_dim', '')}",
                f"- Compression method: {item.compression_metadata.get('method', 'none')}",
                f"- Compression precision: {item.compression_metadata.get('precision', '')}",
                f"- Compressed dimension: {item.compression_metadata.get('compressed_dim', '')}",
                f"- Primary metric: {item.primary_metric_name}",
                f"- Primary score: {_format_float(item.primary_score)}",
                f"- Weakest class: {item.weakest_class or ''}",
                f"- Recommendation: {item.recommendation}",
            ]
        )
        if item.resource_profile:
            profile = item.resource_profile
            lines.extend(["", "#### Resource profile", ""])
            lines.extend(
                [
                    f"- Status: {profile.status}",
                    f"- Inference status: {profile.inference.status}",
                    "- First call: "
                    f"{_format_milliseconds(profile.inference.first_call_seconds)} ms",
                    f"- First call includes fit: {profile.inference.first_call_includes_fit}",
                    f"- Warm calls: {profile.inference.warm_call_count}",
                    "- Warm median: "
                    f"{_format_milliseconds(profile.inference.warm_median_seconds)} ms",
                    f"- Warm p95: {_format_milliseconds(profile.inference.warm_p95_seconds)} ms",
                    "- Throughput: "
                    f"{_format_float(profile.inference.throughput_samples_per_second)} samples/s",
                    f"- Peak host RSS: {_format_bytes(profile.host_memory.peak_rss_bytes)}",
                    "- Peak host RSS increase: "
                    f"{_format_bytes(profile.host_memory.peak_increase_bytes)}",
                    "- Peak device allocated: "
                    f"{_format_bytes(profile.device_memory.peak_allocated_bytes)}",
                    f"- Parameters: {profile.model.parameter_count or ''}",
                    f"- Parameter bytes: {_format_bytes(profile.model.parameter_bytes)}",
                    f"- Checkpoint bytes: {_format_bytes(profile.model.checkpoint_bytes)}",
                    f"- Synchronization: {profile.context.get('synchronization_status', '')}",
                    f"- Batch sizes: {profile.inference.batch_sizes}",
                ]
            )
            if profile.embedding:
                lines.extend(
                    [
                        f"- Raw embedding bytes: {_format_bytes(profile.embedding.raw_bytes)}",
                        "- Evaluated embedding bytes: "
                        f"{_format_bytes(profile.embedding.evaluated_bytes)}",
                        "- Bytes per embedding: "
                        f"{_format_float(profile.embedding.bytes_per_embedding)}",
                    ]
                )
            for warning in profile.warnings:
                lines.append(f"- Warning: {warning}")
        if overlap:
            lines.extend(
                [
                    f"- Overlap macro: {overlap.macro_score:.4f}",
                    f"- Overlap weighted: {_format_float(overlap.weighted_score)}",
                    f"- Excluded aggregate classes: {overlap.metadata.get('exclude_classes', [])}",
                ]
            )
        segmentation = item.embedding_metadata.get("segmentation")
        if segmentation:
            lines.extend(
                [
                    f"- Source images: {segmentation.get('n_images', '')}",
                    f"- Candidate tokens: {segmentation.get('candidate_tokens', '')}",
                    f"- Retained tokens: {segmentation.get('retained_tokens', '')}",
                    f"- Background tokens: {segmentation.get('background_tokens', '')}",
                    f"- Ignored tokens: {segmentation.get('ignored_tokens', {})}",
                    f"- Spatial layout: {segmentation.get('layout', {})}",
                ]
            )
        structured = item.embedding_metadata.get("structured")
        if structured:
            lines.extend(
                [
                    f"- Structured unit type: {structured.get('unit_type', '')}",
                    f"- Structured task family: {structured.get('task_family', '')}",
                    f"- Alignment mode: {structured.get('alignment_mode', '')}",
                    "- Alignment recipe: "
                    f"{_alignment_recipe_label(structured.get('alignment_recipe'))}",
                    f"- Structured parents: {structured.get('n_parents', '')}",
                    f"- Structured units: {structured.get('n_units', '')}",
                ]
            )
        lines.extend(["", "#### Recipe summary", ""])
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
        lines.extend(["", "#### Metrics", ""])
        for metric_name, metric in item.metrics.items():
            direction = "higher is better" if metric.higher_is_better else "lower is better"
            lines.append(f"- {metric_name}: {_format_float(metric.score)} ({direction})")
            for warning in metric.warnings:
                lines.append(f"  - warning: {warning}")
        lines.extend(["", "#### Per-class scores", ""])
        if overlap is None:
            lines.append("Per-class scores are available only for OverlapIndex scoring.")
            lines.append("")
        elif overlap.metadata.get("target_type") == "regression":
            lines.append("Per-class scores are not defined for regression overlap scoring.")
            lines.append("")
        elif overlap.per_class_scores:
            lines.append("| class | score |")
            lines.append("| --- | ---: |")
            for label, score in overlap.per_class_scores.items():
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

        lines.extend(["#### Separatix complexity diagnostic", ""])
        if item.separatix is None:
            lines.append("Separatix diagnostics were disabled.")
        elif not item.separatix.ran:
            lines.append(f"- Skipped: {item.separatix.skipped_reason or ''}")
        else:
            probe = probe_summary_for_result(item.separatix)
            primary_probe_metric = probe.get("primary_metric") or {}
            comparison = probe.get("comparison") or {}
            evaluation = probe.get("evaluation") or {}
            sampling = evaluation.get("sampling") or {}
            lines.append(f"- Recommendation: {item.separatix.recommendation or ''}")
            lines.append(f"- Recommendation confidence: {item.separatix.confidence or ''}")
            lines.append(f"- Summary: {(item.separatix.recommendation_text or '').strip()}")
            lines.append(f"- Probe status: {probe.get('status', '')}")
            lines.append(f"- Best probe: {probe.get('best_probe') or ''}")
            if primary_probe_metric:
                lines.append(
                    "- Primary probe metric: "
                    f"{primary_probe_metric.get('name')}="
                    f"{_format_float(primary_probe_metric.get('value'))}"
                )
            if probe.get("skip_reason"):
                lines.append(f"- Probe unavailable: {probe.get('skip_reason')}")
            lines.append(f"- Probe evaluation mode: {evaluation.get('mode') or ''}")
            lines.append(f"- Grouped evaluation: {bool(evaluation.get('grouped', False))}")
            if evaluation.get("n_groups") is not None:
                lines.append(f"- Independence groups: {evaluation.get('n_groups')}")
            if sampling:
                lines.append(
                    "- Probe sampling: "
                    f"sampled={sampling.get('sampled', False)}, "
                    f"used={sampling.get('n_used', '')}, "
                    f"original={sampling.get('n_original', '')}"
                )
            if probe.get("metrics"):
                lines.append("")
                lines.append("| probe metric | value |")
                lines.append("| --- | ---: |")
                for key, value in probe["metrics"].items():
                    lines.append(f"| {key} | {_format_float(value)} |")
            if comparison:
                lines.append("")
                lines.append("- Linear/nonlinear probe comparison:")
                lines.append(
                    "  - "
                    f"{comparison.get('linear_probe', 'linear')}: "
                    f"{_format_float(comparison.get('linear_value'))}"
                )
                lines.append(
                    "  - "
                    f"{comparison.get('nonlinear_probe', 'nonlinear')}: "
                    f"{_format_float(comparison.get('nonlinear_value'))}"
                )
                lines.append(
                    "  - "
                    f"Delta ({comparison.get('metric', '')}): "
                    f"{_format_float(comparison.get('delta'))}; "
                    f"favored family={comparison.get('favored_family', '')}"
                )
                if comparison.get("confidence") is not None:
                    lines.append("  - Comparison confidence: " f"{comparison.get('confidence')}")
            mlp = ((item.separatix.report or {}).get("metrics", {}) or {}).get("mlp_probes", {})
            if mlp:
                lines.append(f"- MLP status: {mlp.get('status', '')}")
                if mlp.get("reason"):
                    lines.append(f"- MLP reason: {mlp.get('reason')}")
                trigger = mlp.get("trigger", {}) or {}
                if trigger.get("reason"):
                    lines.append(f"- MLP trigger reason: {trigger.get('reason')}")
                backend = mlp.get("backend", {})
                if backend:
                    lines.append(
                        "- MLP backend: "
                        f"{backend.get('resolved_device') or backend.get('requested_device')}"
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
                    lines.append(f"  - {entry.get('name', '')}: {entry.get('reason', '')}")
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


def _format_milliseconds(value: Any) -> str:
    return "" if value is None else f"{float(value) * 1000.0:.3f}"


def _format_bytes(value: Any) -> str:
    return "" if value is None else str(int(value))


def _alignment_recipe_label(recipe: Any) -> str:
    if not recipe:
        return ""
    if isinstance(recipe, dict):
        name = recipe.get("name") or ""
        recipe_data = recipe.get("recipe_data") or {}
        if isinstance(recipe_data, dict) and recipe_data.get("policy"):
            return f"{name} ({recipe_data.get('policy')})".strip()
        return str(name)
    return str(recipe)


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
