"""Markdown report rendering."""

from pathlib import Path
from typing import Any, List

from vertebrae.profiling import DistributedResourceProfile
from vertebrae.reports._markdown import markdown_text as _markdown_text
from vertebrae.scoring.separatix import probe_summary_for_result
from vertebrae.utils.semantic_labels import label_display


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
            f"- Target type: {_markdown_text(target_type)}",
            f"- Modality: {_markdown_text(dataset['modality'])}",
        ]
    )
    if dataset.get("target_view"):
        target_view_name = dataset.get("target_view", {}).get("name", "primary")
        lines.append(f"- Target view: {_markdown_text(target_view_name)}")
    if dataset.get("available_target_views"):
        target_view_names = [item.get("name") for item in dataset.get("available_target_views", [])]
        lines.append(f"- Available target views: {_markdown_text(target_view_names)}")
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
        lines.append(f"- Target names: {_markdown_text(dataset.get('target_names', []))}")
        constant_targets = dataset.get("constant_targets", [])
        if constant_targets:
            lines.append(f"- Constant targets: {_markdown_text(constant_targets)}")
    dataset_metadata = dataset.get("metadata", {})
    if dataset_metadata.get("modalities"):
        lines.append(f"- Modalities: {_markdown_text(dataset_metadata['modalities'])}")
    if dataset_metadata.get("input_fields"):
        lines.append(f"- Input fields: {_markdown_text(dataset_metadata['input_fields'])}")
    if dataset_metadata.get("relational_unit"):
        lines.append(f"- Relational unit: {_markdown_text(dataset_metadata['relational_unit'])}")
    units = dataset.get("units")
    if units:
        lines.append(f"- Unit type: {_markdown_text(units.get('unit_type', 'unit'))}")
    structured_units = dataset.get("structured_units")
    if structured_units:
        lines.append(
            f"- Structured unit type: {_markdown_text(structured_units.get('unit_type', 'unit'))}"
        )
        if structured_units.get("task_family"):
            lines.append(
                f"- Structured task family: {_markdown_text(structured_units.get('task_family'))}"
            )
        lines.append(f"- Structured parents: {structured_units.get('n_parents', '')}")
        lines.append(f"- Structured units: {structured_units.get('n_units', '')}")
    if dataset_metadata.get("entity_type"):
        lines.append(f"- Entity type: {_markdown_text(dataset_metadata['entity_type'])}")
    if dataset_metadata.get("composition"):
        lines.append(f"- Embedding composition: {_markdown_text(dataset_metadata['composition'])}")
    lines.extend(
        [
            "",
            "## Executive summary",
            "",
        ]
    )
    for item in data.get("recommendations", []):
        lines.append(f"- {_markdown_text(item)}")
    for warning in data.get("metadata", {}).get("label_view_warnings", []):
        lines.append(f"- {_markdown_text(warning)}")
    for warning in data.get("metadata", {}).get("target_view_warnings", []):
        lines.append(f"- {_markdown_text(warning)}")
    ranked_results = result.ranked_results()
    top_ranked = ranked_results[0] if ranked_results else None
    if top_ranked and top_ranked.separatix and top_ranked.separatix.ran:
        lines.append(
            "- "
            f"Separatix complexity guidance for the top representation: "
            f"{_markdown_text(top_ranked.separatix.recommendation or '')} "
            f"({_markdown_text(top_ranked.separatix.confidence or '')} confidence).".strip()
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
                f"| {_markdown_text(output.get('extractor', ''))} | "
                f"{_markdown_text(output.get('output', ''))} | "
                f"{_markdown_text(output.get('unit_type', ''))} | "
                f"{_markdown_text(output.get('task_family', ''))} | "
                f"{_markdown_text(output.get('alignment_mode', ''))} | "
                f"{_markdown_text(_alignment_recipe_label(output.get('alignment_recipe')))} |"
            )
    lines.extend(["", "## Ranking", ""])
    if not ranked_results:
        lines.append("Ranking unavailable because no valid aggregate remains.")
        lines.append("")
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
    for rank, item in enumerate(ranked_results, start=1):
        interval = _format_interval(item.stability)
        weakest = (
            label_display(
                item.weakest_class,
                item.overlap.metadata.get("label_catalog", []) if item.overlap else [],
            )
            if item.weakest_class is not None
            else ""
        )
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
            f"| {rank} | {_markdown_text(item.name)} | {_markdown_text(item.extractor_type)} | "
            f"{_markdown_text(target_view)} | {_markdown_text(label_view)} | "
            f"{_markdown_text(item.primary_metric_name)} | "
            f"{_format_float(item.primary_score)} | "
            f"{_format_float(item.overlap.score if item.overlap else None)} | "
            f"{_format_float(item.overlap.macro_score if item.overlap else None)} | "
            f"{_format_float(item.overlap.weighted_score if item.overlap else None)} | "
            f"{_markdown_text(interval)} | {_markdown_text(weakest)} | "
            f"{_markdown_text(probe.get('best_probe') or '')} | "
            f"{_markdown_text(primary_probe_metric.get('name') or '')} | "
            f"{_format_float(primary_probe_metric.get('value'))} | "
            f"{_markdown_text(embedding_dim)} | {_markdown_text(compression_method)} | "
            f"{_markdown_text(compressed_dim)} | {_markdown_text(item.recommendation)} | "
            f"{_markdown_text(separatix_recommendation)} | "
            f"{_markdown_text(separatix_confidence)} |"
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
            if isinstance(profile, DistributedResourceProfile):
                evaluated_bytes = profile.embedding.evaluated_bytes if profile.embedding else None
                lines.append(
                    f"| {_markdown_text(item.name)} | worker-first | "
                    f"{_format_milliseconds(profile.worker_first_calls.median_seconds)} | "
                    f"{_format_milliseconds(profile.worker_first_calls.p95_seconds)} | "
                    f"{_format_float(profile.aggregate_compute_throughput_samples_per_second)} | "
                    f"{_format_bytes(profile.max_worker_peak_rss_bytes)} | "
                    f"{_format_bytes(profile.max_worker_peak_device_allocated_bytes)} | "
                    f"{_format_bytes(profile.model.in_memory_bytes if profile.model else None)} | "
                    f"{_format_bytes(profile.model.checkpoint_bytes if profile.model else None)} | "
                    f"{_format_bytes(evaluated_bytes)} | workers | aggregate compute |"
                )
                continue
            inference = profile.inference
            evaluated_bytes = profile.embedding.evaluated_bytes if profile.embedding else None
            lines.append(
                f"| {_markdown_text(item.name)} | "
                f"{_format_milliseconds(inference.first_call_seconds)} | "
                f"{_format_milliseconds(inference.warm_median_seconds)} | "
                f"{_format_milliseconds(inference.warm_p95_seconds)} | "
                f"{_format_float(inference.throughput_samples_per_second)} | "
                f"{_format_bytes(profile.host_memory.peak_increase_bytes)} | "
                f"{_format_bytes(profile.device_memory.peak_allocated_bytes)} | "
                f"{_format_bytes(profile.model.in_memory_bytes)} | "
                f"{_format_bytes(profile.model.checkpoint_bytes)} | "
                f"{_format_bytes(evaluated_bytes)} | "
                f"{_markdown_text(inference.batch_sizes)} | "
                f"{_markdown_text(profile.context.get('synchronization_status', ''))} |"
            )

    lines.extend(["", "## Per-extractor details", ""])
    for item in ranked_results:
        overlap = item.overlap
        target_metadata = (
            overlap.metadata if overlap else item.metrics[item.primary_metric_name].metadata
        )
        output_metadata = item.embedding_metadata.get("output_metadata") or {}
        weakest_class = (
            label_display(
                item.weakest_class,
                target_metadata.get("label_catalog", []),
            )
            if item.weakest_class is not None
            else ""
        )
        lines.extend(
            [
                f"### {_markdown_text(item.name)}",
                "",
                f"- Extractor type: {_markdown_text(item.extractor_type)}",
                f"- Extractor family: {_markdown_text(_extractor_family(item.extractor_type))}",
                "- Target type: "
                f"{_markdown_text(target_metadata.get('target_type', 'single_label'))}",
                f"- Target view: {_markdown_text((item.target_view or {}).get('name', 'primary'))}",
                f"- Label view: {_markdown_text((item.label_view or {}).get('name', 'primary'))}",
                f"- Modality: {_markdown_text(item.embedding_metadata.get('modality', ''))}",
                f"- Output source: {_markdown_text(output_metadata.get('source', ''))}",
                f"- Embedding dimension: {item.embedding_metadata.get('embedding_dim', '')}",
                "- Compression method: "
                f"{_markdown_text(item.compression_metadata.get('method', 'none'))}",
                "- Compression precision: "
                f"{_markdown_text(item.compression_metadata.get('precision', ''))}",
                f"- Compressed dimension: {item.compression_metadata.get('compressed_dim', '')}",
                f"- Primary metric: {_markdown_text(item.primary_metric_name)}",
                f"- Primary score: {_format_float(item.primary_score)}",
                f"- Weakest class: {_markdown_text(weakest_class)}",
                f"- Recommendation: {_markdown_text(item.recommendation)}",
            ]
        )
        if item.resource_profile:
            profile = item.resource_profile
            lines.extend(["", "#### Resource profile", ""])
            if isinstance(profile, DistributedResourceProfile):
                lines.extend(_distributed_resource_details(profile))
            else:
                lines.extend(_local_resource_details(profile))
        if overlap:
            lines.extend(
                [
                    f"- Overlap macro: {overlap.macro_score:.4f}",
                    f"- Overlap weighted: {_format_float(overlap.weighted_score)}",
                    "- Excluded aggregate classes: "
                    f"{_markdown_text(overlap.metadata.get('exclude_classes', []))}",
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
                    f"- Ignored tokens: {_markdown_text(segmentation.get('ignored_tokens', {}))}",
                    f"- Spatial layout: {_markdown_text(segmentation.get('layout', {}))}",
                ]
            )
        structured = item.embedding_metadata.get("structured")
        if structured:
            lines.extend(
                [
                    f"- Structured unit type: {_markdown_text(structured.get('unit_type', ''))}",
                    "- Structured task family: "
                    f"{_markdown_text(structured.get('task_family', ''))}",
                    f"- Alignment mode: {_markdown_text(structured.get('alignment_mode', ''))}",
                    "- Alignment recipe: "
                    f"{_markdown_text(_alignment_recipe_label(structured.get('alignment_recipe')))}",
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
                    lines.append(f"- {_markdown_text(key)}: {len(value)} captured parameters")
                else:
                    lines.append(f"- {_markdown_text(key)}: {_markdown_text(value)}")
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
                lines.append(
                    f"- {_markdown_text(key)}: {_markdown_text(compression_metadata[key])}"
                )
        for warning in compression_metadata.get("warnings", []):
            lines.append(f"- warning: {_markdown_text(warning)}")
        lines.extend(["", "#### Metrics", ""])
        for metric_name, metric in item.metrics.items():
            direction = "higher is better" if metric.higher_is_better else "lower is better"
            lines.append(
                f"- {_markdown_text(metric_name)}: {_format_float(metric.score)} ({direction})"
            )
            for warning in metric.warnings:
                lines.append(f"  - warning: {_markdown_text(warning)}")
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
            catalog = overlap.metadata.get("label_catalog", [])
            for label, score in overlap.per_class_scores.items():
                lines.append(
                    f"| {_markdown_text(label_display(label, catalog))} | "
                    f"{_format_float(score)} |"
                )
            lines.append("")
        else:
            lines.append("No per-class scores were returned.")
            lines.append("")

        lines.extend(["#### Stability analysis", ""])
        if item.stability:
            summary = item.stability.get("summary", {})
            lines.append(
                "- "
                f"{_markdown_text(item.stability.get('mode'))} stability interval: "
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
            lines.append(f"- Skipped: {_markdown_text(item.separatix.skipped_reason or '')}")
        else:
            probe = probe_summary_for_result(item.separatix)
            primary_probe_metric = probe.get("primary_metric") or {}
            comparison = probe.get("comparison") or {}
            evaluation = probe.get("evaluation") or {}
            sampling = evaluation.get("sampling") or {}
            lines.append(f"- Recommendation: {_markdown_text(item.separatix.recommendation or '')}")
            lines.append(
                "- Recommendation confidence: " f"{_markdown_text(item.separatix.confidence or '')}"
            )
            guidance = item.separatix.family_guidance or {}
            if guidance:
                lines.append(
                    "- Minimum recommended family: "
                    f"{_markdown_text(guidance.get('minimum_recommended_family') or '')}"
                )
                lines.append(
                    "- Plausible families: "
                    f"{_markdown_text(guidance.get('plausible_families', []))}"
                )
                lines.append(
                    "- Selected family/probe: "
                    f"{_markdown_text(guidance.get('selected_family') or '')}/"
                    f"{_markdown_text(guidance.get('selected_probe') or '')}"
                )
                lines.append(
                    "- Selected recipe id: "
                    f"{_markdown_text(guidance.get('selected_recipe_id') or '')}"
                )
                lines.append("- MLP override: " f"{bool(guidance.get('mlp_override', False))}")
                lines.append(
                    "- Paired evidence: "
                    f"{_markdown_text(guidance.get('paired_status') or '')}"
                    f" ({_markdown_text(guidance.get('paired_method') or '')})"
                )
            lines.append(
                f"- Summary: {_markdown_text((item.separatix.recommendation_text or '').strip())}"
            )
            lines.append(
                "- Sparse diagnostic input: "
                f"{bool(item.separatix.preprocessing.get('is_sparse', False))}"
            )
            lines.append(
                "- Densification policy: "
                f"{_markdown_text(item.separatix.metadata.get('densify_policy', ''))}"
            )
            lines.append(f"- Probe status: {_markdown_text(probe.get('status', ''))}")
            lines.append(f"- Best probe: {_markdown_text(probe.get('best_probe') or '')}")
            if primary_probe_metric:
                lines.append(
                    "- Primary probe metric: "
                    f"{_markdown_text(primary_probe_metric.get('name'))}="
                    f"{_format_float(primary_probe_metric.get('value'))}"
                )
            if probe.get("skip_reason"):
                lines.append(f"- Probe unavailable: {_markdown_text(probe.get('skip_reason'))}")
            lines.append(f"- Probe evaluation mode: {_markdown_text(evaluation.get('mode') or '')}")
            lines.append(
                "- Probe alignment: " f"{_markdown_text(evaluation.get('alignment_status') or '')}"
            )
            lines.append(
                "- Probe CV/cohort: "
                f"{_markdown_text(evaluation.get('cv_method') or '')}; "
                f"{_markdown_text(evaluation.get('cohort_size') or '')} rows; "
                f"{_markdown_text(evaluation.get('n_splits') or '')} splits"
            )
            effective_train_size = evaluation.get("effective_train_size_summary") or {}
            if effective_train_size:
                lines.append(
                    "- Effective probe train size: "
                    f"{_markdown_text(effective_train_size.get('mean') or '')} mean "
                    f"({_markdown_text(effective_train_size.get('basis') or '')})"
                )
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
                    lines.append(f"| {_markdown_text(key)} | {_format_float(value)} |")
            if comparison:
                lines.append("")
                lines.append("- Linear/nonlinear probe comparison:")
                lines.append(
                    "  - "
                    f"{_markdown_text(comparison.get('linear_probe', 'linear'))}: "
                    f"{_format_float(comparison.get('linear_value'))}"
                )
                lines.append(
                    "  - "
                    f"{_markdown_text(comparison.get('nonlinear_probe', 'nonlinear'))}: "
                    f"{_format_float(comparison.get('nonlinear_value'))}"
                )
                lines.append(
                    "  - "
                    f"Delta ({_markdown_text(comparison.get('metric', ''))}): "
                    f"{_format_float(comparison.get('delta'))}; "
                    f"favored family={_markdown_text(comparison.get('favored_family', ''))}"
                )
                if comparison.get("confidence") is not None:
                    lines.append(
                        "  - Comparison confidence: "
                        f"{_markdown_text(comparison.get('confidence'))}"
                    )
            mlp = ((item.separatix.report or {}).get("metrics", {}) or {}).get("mlp_probes", {})
            if mlp:
                lines.append(f"- MLP status: {_markdown_text(mlp.get('status', ''))}")
                if mlp.get("reason"):
                    lines.append(f"- MLP reason: {_markdown_text(mlp.get('reason'))}")
                trigger = mlp.get("trigger", {}) or {}
                if trigger.get("reason"):
                    lines.append(f"- MLP trigger reason: {_markdown_text(trigger.get('reason'))}")
                backend = mlp.get("backend", {})
                if backend:
                    backend_device = backend.get("resolved_device") or backend.get(
                        "requested_device"
                    )
                    lines.append(f"- MLP backend: {_markdown_text(backend_device)}")
            if item.separatix.decision_path:
                lines.append("- Decision path:")
                for step in item.separatix.decision_path:
                    lines.append(f"  - {_markdown_text(step)}")
            if item.separatix.scores:
                lines.append("")
                lines.append("| score | value |")
                lines.append("| --- | ---: |")
                for key, value in item.separatix.scores.items():
                    lines.append(f"| {_markdown_text(key)} | {_format_float(value)} |")
            if item.separatix.skipped_diagnostics:
                lines.append("")
                lines.append("- Skipped diagnostics:")
                for entry in item.separatix.skipped_diagnostics:
                    lines.append(
                        f"  - {_markdown_text(entry.get('name', ''))}: "
                        f"{_markdown_text(entry.get('reason', ''))}"
                    )
            if item.separatix.densification_events:
                lines.append("")
                lines.append("- Densification events:")
                for entry in item.separatix.densification_events:
                    name = entry.get("diagnostic") or entry.get("name") or entry.get("operation")
                    action = entry.get("action") or entry.get("reason") or entry.get("status")
                    lines.append(
                        f"  - {_markdown_text(name or '')}: " f"{_markdown_text(action or '')}"
                    )
            if item.separatix.warnings:
                lines.append("")
                lines.append("- Warnings:")
                for warning in item.separatix.warnings:
                    lines.append(f"  - {_markdown_text(warning)}")
        lines.append("")

        warnings = item.warnings
        if warnings:
            lines.extend(["#### Warnings", ""])
            for warning in warnings:
                lines.append(f"- {_markdown_text(warning)}")
            lines.append("")

    lines.extend(["## OverlapIndex configuration", ""])
    scoring_config = data.get("metadata", {}).get("scoring_config", {})
    for key, value in scoring_config.items():
        lines.append(f"- {_markdown_text(key)}: {_markdown_text(value)}")
    lines.extend(["", "## Reproducibility metadata", ""])
    for key, value in data.get("metadata", {}).items():
        if key != "scoring_config":
            lines.append(f"- {_markdown_text(key)}: {_markdown_text(value)}")
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


def _distributed_resource_details(profile: DistributedResourceProfile) -> List[str]:
    embedding = profile.embedding
    model = profile.model
    persisted = embedding.evaluated_persisted if embedding else None
    return [
        f"- Status: {_markdown_text(profile.status)}",
        "- Scope: independent distributed worker profiling windows",
        f"- Profiled workers: {profile.profiled_shard_count}/{profile.shard_count}",
        f"- Worker-first calls: {profile.worker_first_calls.count}",
        "- Worker-first median: "
        f"{_format_milliseconds(profile.worker_first_calls.median_seconds)} ms",
        f"- Worker-first p95: {_format_milliseconds(profile.worker_first_calls.p95_seconds)} ms",
        f"- Worker-first max: {_format_milliseconds(profile.worker_first_calls.max_seconds)} ms",
        "- Aggregate compute throughput: "
        f"{_format_float(profile.aggregate_compute_throughput_samples_per_second)} samples/s",
        "- Throughput meaning: summed samples divided by summed worker compute seconds; "
        "not cluster wall-clock throughput",
        f"- Maximum worker RSS: {_format_bytes(profile.max_worker_peak_rss_bytes)}",
        "- Maximum worker RSS shard: "
        f"{_markdown_text(profile.max_worker_peak_rss_shard_key or '')}",
        "- Maximum worker device allocation: "
        f"{_format_bytes(profile.max_worker_peak_device_allocated_bytes)}",
        "- Logical evaluated embedding bytes: "
        f"{_format_bytes(embedding.evaluated_bytes if embedding else None)}",
        "- Persisted evaluated embedding bytes: "
        f"{_format_bytes(persisted.bytes if persisted else None)}",
        f"- Persisted evaluated status: {_markdown_text(persisted.status if persisted else '')}",
        f"- Shard persisted bytes: {_format_bytes(profile.shard_persisted_bytes)}",
        f"- Parameters: {model.parameter_count if model else ''}",
        f"- Model in-memory bytes: " f"{_format_bytes(model.in_memory_bytes if model else None)}",
        f"- Checkpoint bytes: {_format_bytes(model.checkpoint_bytes if model else None)}",
        f"- Weight dtypes: {_markdown_text(model.weight_dtypes if model else [])}",
        *[f"- Warning: {_markdown_text(warning)}" for warning in profile.warnings],
    ]


def _local_resource_details(profile: Any) -> List[str]:
    embedding = profile.embedding
    raw_persisted = embedding.raw_persisted if embedding else None
    evaluated_persisted = embedding.evaluated_persisted if embedding else None
    lines = [
        f"- Status: {_markdown_text(profile.status)}",
        f"- Inference status: {_markdown_text(profile.inference.status)}",
        f"- First call: {_format_milliseconds(profile.inference.first_call_seconds)} ms",
        f"- First call includes fit: {profile.inference.first_call_includes_fit}",
        f"- Warm calls: {profile.inference.warm_call_count}",
        f"- Warm median: {_format_milliseconds(profile.inference.warm_median_seconds)} ms",
        f"- Warm p95: {_format_milliseconds(profile.inference.warm_p95_seconds)} ms",
        "- Throughput: "
        f"{_format_float(profile.inference.throughput_samples_per_second)} samples/s",
        f"- Peak host RSS: {_format_bytes(profile.host_memory.peak_rss_bytes)}",
        "- Peak host RSS increase: " f"{_format_bytes(profile.host_memory.peak_increase_bytes)}",
        f"- Peak device allocated: {_format_bytes(profile.device_memory.peak_allocated_bytes)}",
        f"- Model-footprint status: {_markdown_text(profile.model.status)}",
        f"- Parameters: {profile.model.parameter_count or ''}",
        f"- Model in-memory bytes: {_format_bytes(profile.model.in_memory_bytes)}",
        f"- Checkpoint bytes: {_format_bytes(profile.model.checkpoint_bytes)}",
        f"- Synchronization: {_markdown_text(profile.context.get('synchronization_status', ''))}",
        f"- Cache: {_markdown_text(profile.context.get('cache_status', ''))}",
        f"- Measurement scope: {_markdown_text(profile.context.get('measurement_scope', ''))}",
        f"- Batch sizes: {_markdown_text(profile.inference.batch_sizes)}",
    ]
    if embedding:
        lines.extend(
            [
                f"- Raw logical embedding bytes: {_format_bytes(embedding.raw_bytes)}",
                "- Evaluated logical embedding bytes: "
                f"{_format_bytes(embedding.evaluated_bytes)}",
                "- Raw persisted embedding bytes: "
                f"{_format_bytes(raw_persisted.bytes if raw_persisted else None)}",
                "- Evaluated persisted embedding bytes: "
                f"{_format_bytes(evaluated_persisted.bytes if evaluated_persisted else None)}",
            ]
        )
    for artifact in profile.model.artifacts:
        lines.append(
            "- Deployment artifact: "
            f"{_markdown_text(artifact.role)} {_markdown_text(artifact.path)} "
            f"({_markdown_text(artifact.status)}, {artifact.bytes} bytes)"
        )
    lines.extend(f"- Warning: {_markdown_text(warning)}" for warning in profile.warnings)
    return lines


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
