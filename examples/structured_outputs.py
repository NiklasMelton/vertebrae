"""Evaluate raw structured outputs as embedding-efficacy diagnostics."""

import numpy as np
from _common import ensure_cache_dir, ensure_output_dir

from vertebrae import (
    Benchmark,
    BenchmarkDataset,
    DatasetIdentity,
    DetectionLayoutAdapter,
    KeypointAdapter,
    KeypointAnnotation,
    RegionAnnotation,
    SequenceAnnotation,
    SequenceLabelingAdapter,
)
from vertebrae.config import CacheConfig, OverlapScoringConfig, SeparatixConfig, StabilityConfig
from vertebrae.execution import materialize_structured_artifacts
from vertebrae.extractors import CallableStructuredExtractor, StructuredOutputSpec


def main() -> None:
    output_dir = ensure_output_dir()
    cache_dir = ensure_cache_dir()

    for workflow in (
        _ocr_layout_workflow(),
        _asr_token_workflow(),
        _pose_keypoint_workflow(),
    ):
        result = Benchmark(
            dataset=workflow["dataset"],
            extractors=[workflow["extractor"]],
            scoring_config=OverlapScoringConfig(k=2, min_samples_per_cluster=2),
            stability_config=StabilityConfig(repeats=3, random_state=11),
            separatix_config=SeparatixConfig(enabled=False),
            cache_config=CacheConfig(enabled=False, cache_dir=str(cache_dir / workflow["stem"])),
        ).run()
        result.save_json(str(output_dir / f"{workflow['stem']}.json"))
        result.save_markdown(str(output_dir / f"{workflow['stem']}.md"))

        bundle = materialize_structured_artifacts(
            workflow["dataset"],
            workflow["extractor"],
            workflow["store"],
            batch_size=2,
        )
        print(
            f"{workflow['title']}: overlap={result.extractor_results[0].overlap.macro_score:.3f} "
            f"rows={bundle['outputs'][0]['n_samples']} bundle={bundle['output_key']}"
        )

    print(
        "\nThese workflows diagnose representation efficacy for labeled regions, tokens, and "
        "keypoints. They do not compute IoU, WER/CER, OKS, or other task-native metrics."
    )
    print(f"Reports written to {output_dir}")


def _lookup_structured_embeddings(embeddings_by_parent):
    def transform(batch):
        return [embeddings_by_parent[str(item)] for item in np.asarray(batch).tolist()]

    return transform


def _ocr_layout_workflow():
    dataset = BenchmarkDataset.from_arrays(
        X=np.asarray(["page_a", "page_b", "page_c", "page_d"], dtype=object),
        y=["invoice", "invoice", "report", "report"],
        modality="image",
        metadata={"example": "structured_ocr_layout"},
        identity=DatasetIdentity.declared("structured-ocr-layout-example", "1"),
    )
    dataset = DetectionLayoutAdapter(unit_type="document_region").attach(
        dataset,
        [
            RegionAnnotation(
                labels=["header", "table", "table"],
                unit_ids=["page_a:h", "page_a:t0", "page_a:t1"],
                boxes=[
                    [0.05, 0.05, 0.95, 0.18],
                    [0.08, 0.22, 0.92, 0.48],
                    [0.08, 0.52, 0.92, 0.78],
                ],
                page_id=0,
                document_id="page_a",
            ),
            RegionAnnotation(
                labels=["header", "table", "table"],
                unit_ids=["page_b:h", "page_b:t0", "page_b:t1"],
                boxes=[
                    [0.05, 0.05, 0.95, 0.18],
                    [0.08, 0.22, 0.92, 0.48],
                    [0.08, 0.52, 0.92, 0.78],
                ],
                page_id=1,
                document_id="page_b",
            ),
            RegionAnnotation(
                labels=["title", "body", "footer"],
                unit_ids=["page_c:t", "page_c:b", "page_c:f"],
                boxes=[
                    [0.08, 0.06, 0.88, 0.16],
                    [0.08, 0.20, 0.90, 0.74],
                    [0.10, 0.82, 0.84, 0.92],
                ],
                page_id=0,
                document_id="page_c",
            ),
            RegionAnnotation(
                labels=["title", "body", "footer"],
                unit_ids=["page_d:t", "page_d:b", "page_d:f"],
                boxes=[
                    [0.08, 0.06, 0.88, 0.16],
                    [0.08, 0.20, 0.90, 0.74],
                    [0.10, 0.82, 0.84, 0.92],
                ],
                page_id=1,
                document_id="page_d",
            ),
        ],
    )
    embeddings = {
        "page_a": np.asarray([[1.0, 0.0], [0.9, 0.1], [0.85, 0.15]]),
        "page_b": np.asarray([[1.0, 0.0], [0.9, 0.1], [0.85, 0.15]]),
        "page_c": np.asarray([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]),
        "page_d": np.asarray([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]),
    }
    extractor = CallableStructuredExtractor(
        name="layout_regions",
        transform_fn=_lookup_structured_embeddings(embeddings),
        output_specs=[StructuredOutputSpec(name="regions", unit_type="document_region")],
        modality="image",
    )
    return {
        "title": "OCR/layout regions",
        "stem": "structured_ocr_layout",
        "dataset": dataset,
        "extractor": extractor,
        "store": workflow_store("structured_ocr_layout"),
    }


def _asr_token_workflow():
    dataset = BenchmarkDataset.from_arrays(
        X=np.asarray(["utt_a", "utt_b", "utt_c", "utt_d"], dtype=object),
        y=["support", "support", "weather", "weather"],
        modality="audio",
        metadata={"example": "structured_asr_tokens"},
        identity=DatasetIdentity.declared("structured-keypoints-example", "1"),
    )
    dataset = SequenceLabelingAdapter().attach(
        dataset,
        [
            SequenceAnnotation(
                labels=["greeting", "intent", "entity"],
                unit_ids=["utt_a:0", "utt_a:1", "utt_a:2"],
                spans=[[0.0, 0.2], [0.2, 0.5], [0.5, 0.9]],
                tokens=["hello", "need", "billing"],
                utterance_id="a",
            ),
            SequenceAnnotation(
                labels=["greeting", "intent", "entity"],
                unit_ids=["utt_b:0", "utt_b:1", "utt_b:2"],
                spans=[[0.0, 0.2], [0.2, 0.5], [0.5, 0.9]],
                tokens=["hi", "need", "billing"],
                utterance_id="b",
            ),
            SequenceAnnotation(
                labels=["query", "slot", "slot"],
                unit_ids=["utt_c:0", "utt_c:1", "utt_c:2"],
                spans=[[0.0, 0.2], [0.2, 0.5], [0.5, 0.9]],
                tokens=["what", "is", "forecast"],
                utterance_id="c",
            ),
            SequenceAnnotation(
                labels=["query", "slot", "slot"],
                unit_ids=["utt_d:0", "utt_d:1", "utt_d:2"],
                spans=[[0.0, 0.2], [0.2, 0.5], [0.5, 0.9]],
                tokens=["show", "me", "forecast"],
                utterance_id="d",
            ),
        ],
    )
    embeddings = {
        "utt_a": np.asarray([[1.0, 0.1], [0.8, 0.2], [0.7, 0.3]]),
        "utt_b": np.asarray([[1.0, 0.1], [0.8, 0.2], [0.7, 0.3]]),
        "utt_c": np.asarray([[0.2, 1.0], [0.3, 0.8], [0.4, 0.7]]),
        "utt_d": np.asarray([[0.2, 1.0], [0.3, 0.8], [0.4, 0.7]]),
    }
    extractor = CallableStructuredExtractor(
        name="asr_tokens",
        transform_fn=_lookup_structured_embeddings(embeddings),
        output_specs=[StructuredOutputSpec(name="tokens", unit_type="token")],
        modality="audio",
    )
    return {
        "title": "ASR tokens",
        "stem": "structured_asr_tokens",
        "dataset": dataset,
        "extractor": extractor,
        "store": workflow_store("structured_asr_tokens"),
    }


def _pose_keypoint_workflow():
    dataset = BenchmarkDataset.from_arrays(
        X=np.asarray(["frame_a", "frame_b", "frame_c", "frame_d"], dtype=object),
        y=["walk", "walk", "stretch", "stretch"],
        modality="image",
        metadata={"example": "structured_pose_keypoints"},
        identity=DatasetIdentity.declared("structured-sequence-example", "1"),
    )
    dataset = KeypointAdapter().attach(
        dataset,
        [
            KeypointAnnotation(
                labels=["shoulder", "elbow", "wrist"],
                unit_ids=["fa:s", "fa:e", "fa:w"],
                coordinates=[[0.30, 0.20], [0.42, 0.34], [0.55, 0.48]],
                frame_id=0,
                person_id="p0",
            ),
            KeypointAnnotation(
                labels=["shoulder", "elbow", "wrist"],
                unit_ids=["fb:s", "fb:e", "fb:w"],
                coordinates=[[0.31, 0.20], [0.43, 0.34], [0.56, 0.48]],
                frame_id=1,
                person_id="p0",
            ),
            KeypointAnnotation(
                labels=["hip", "knee", "ankle"],
                unit_ids=["fc:h", "fc:k", "fc:a"],
                coordinates=[[0.34, 0.58], [0.42, 0.75], [0.50, 0.92]],
                frame_id=0,
                person_id="p1",
            ),
            KeypointAnnotation(
                labels=["hip", "knee", "ankle"],
                unit_ids=["fd:h", "fd:k", "fd:a"],
                coordinates=[[0.35, 0.58], [0.43, 0.75], [0.51, 0.92]],
                frame_id=1,
                person_id="p1",
            ),
        ],
    )
    embeddings = {
        "frame_a": np.asarray([[0.95, 0.05], [0.85, 0.15], [0.75, 0.25]]),
        "frame_b": np.asarray([[0.95, 0.05], [0.85, 0.15], [0.75, 0.25]]),
        "frame_c": np.asarray([[0.15, 0.85], [0.05, 0.95], [0.10, 0.90]]),
        "frame_d": np.asarray([[0.15, 0.85], [0.05, 0.95], [0.10, 0.90]]),
    }
    extractor = CallableStructuredExtractor(
        name="pose_keypoints",
        transform_fn=_lookup_structured_embeddings(embeddings),
        output_specs=[StructuredOutputSpec(name="keypoints", unit_type="keypoint")],
        modality="image",
    )
    return {
        "title": "Pose keypoints",
        "stem": "structured_pose_keypoints",
        "dataset": dataset,
        "extractor": extractor,
        "store": workflow_store("structured_pose_keypoints"),
    }


def workflow_store(name: str):
    from vertebrae.cache.local_store import LocalArtifactStore

    return LocalArtifactStore(ensure_cache_dir() / name)


if __name__ == "__main__":
    main()
