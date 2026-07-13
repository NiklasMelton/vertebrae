"""Optional OpenCLIP-style vision-language extractor."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vertebrae.extractors._utils import (
    coerce_image,
    ensure_text_sequence,
    iter_chunks,
    maybe_move_to_device,
    resolve_output_specs,
    resolve_output_value,
    spec_to_recipe,
    stack_batch,
    tensor_to_numpy,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec


class OpenCLIPExtractor:
    """Wrap an OpenCLIP-style model with image/text branch outputs."""

    def __init__(
        self,
        name: str,
        model_name: str,
        pretrained: Optional[str] = None,
        input_modalities: Optional[Dict[str, str]] = None,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        batch_size: int = 16,
        image_mode: str = "auto",
        alpha_mode: str = "drop",
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_paths: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.pretrained = pretrained
        self.input_modalities = input_modalities or {"image": "image", "text": "text"}
        self._output_specs = resolve_output_specs(
            outputs or [{"name": "image_branch", "source": "image"}]
        )
        self.batch_size = batch_size
        self.image_mode = image_mode
        self.alpha_mode = alpha_mode
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.checkpoint_paths = tuple(checkpoint_paths or ())
        self.modality = "multimodal"
        self.extractor_type = "openclip"
        self.streaming_safe = True
        self._torch: Any = None
        self._image_module: Any = None
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def fit(self, X: Any, y: Any = None) -> "OpenCLIPExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "OpenCLIPExtractor.transform() is only available when exactly one output is "
                "configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        torch_module, image_module, model, preprocess_fn, tokenizer = self._load_model()
        samples = _normalize_multimodal_samples(X, self.input_modalities)
        model.eval()
        collected: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in self._output_specs}
        with torch_module.no_grad():
            for batch in iter_chunks(samples, self.batch_size):
                projected = self._project_batch(
                    batch=batch,
                    torch_module=torch_module,
                    image_module=image_module,
                    model=model,
                    preprocess_fn=preprocess_fn,
                    tokenizer=tokenizer,
                )
                for spec in self._output_specs:
                    source = spec.metadata.get("source")
                    value = projected.get(source) if source is not None else projected
                    if spec.metadata.get("selector"):
                        value = resolve_output_value(value, spec.metadata.get("selector"))
                    array = np.asarray(tensor_to_numpy(value), dtype=np.float32)
                    if array.ndim == 1:
                        array = array.reshape(1, -1)
                    collected[spec.name].append(array)
        return [
            EmbeddingOutput(
                name=spec.name,
                embeddings=np.vstack(collected[spec.name]).astype(np.float32, copy=False),
                recipe=spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
            for spec in self._output_specs
        ]

    def encode_retrieval(self, X: Any, *, branch: str, modality: str) -> np.ndarray:
        """Encode one independent image or text endpoint for exact retrieval."""
        spec = next((item for item in self._output_specs if item.name == branch), None)
        source: Any
        # Retrieval endpoints are native OpenCLIP capabilities, rather than a
        # declaration that ordinary transform_many() happens to expose.  Keep the
        # historical image-only transform default while always allowing its paired
        # text endpoint for zero-shot/retrieval protocols.
        if spec is None and branch in {"image_branch", "text_branch"}:
            source = "image" if branch == "image_branch" else "text"
        elif spec is None:
            raise ValueError(f"Unknown retrieval branch {branch!r}.")
        else:
            source = spec.metadata.get("source")
        if source not in {"image", "text"} or modality != source:
            raise ValueError(f"Retrieval branch {branch!r} requires modality {source!r}.")
        torch_module, image_module, model, preprocess_fn, tokenizer = self._load_model()
        values = list(X)
        collected = []
        model.eval()
        with torch_module.no_grad():
            for batch in iter_chunks(values, self.batch_size):
                if source == "image":
                    images = [
                        preprocess_fn(
                            coerce_image(value, image_module, self.image_mode, self.alpha_mode)
                        )
                        for value in batch
                    ]
                    encoded = model.encode_image(
                        maybe_move_to_device(
                            stack_batch(images, torch_module),
                            self._device(torch_module),
                            torch_module,
                        )
                    )
                else:
                    encoded = model.encode_text(
                        maybe_move_to_device(
                            tokenizer(ensure_text_sequence(batch, "OpenCLIP retrieval")),
                            self._device(torch_module),
                            torch_module,
                        )
                    )
                collected.append(np.asarray(tensor_to_numpy(encoded), dtype=np.float32))
        return np.vstack(collected)

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "input_modalities": dict(self.input_modalities),
            "outputs": [spec_to_recipe(spec) for spec in self._output_specs],
            "batch_size": self.batch_size,
            "image_mode": self.image_mode,
            "alpha_mode": self.alpha_mode,
            "device": self.device,
            "model_kwargs": self.model_kwargs,
            "checkpoint_paths": list(self.checkpoint_paths),
            "streaming_safe": self.streaming_safe,
        }

    def get_resource_profile_adapter(self) -> Any:
        from vertebrae.profiling import TorchResourceProfileAdapter

        return TorchResourceProfileAdapter(
            self,
            self.checkpoint_paths,
            model_getter=lambda: self._model,
            device_resolver=self._device,
        )

    def _project_batch(
        self,
        batch: List[Dict[str, Any]],
        torch_module: Any,
        image_module: Any,
        model: Any,
        preprocess_fn: Any,
        tokenizer: Any,
    ) -> Dict[str, Any]:
        images = [
            preprocess_fn(
                coerce_image(
                    sample[_first_field(self.input_modalities, "image")],
                    image_module=image_module,
                    image_mode=self.image_mode,
                    alpha_mode=self.alpha_mode,
                )
            )
            for sample in batch
        ]
        texts = [sample[_first_field(self.input_modalities, "text")] for sample in batch]
        image_batch = stack_batch(images, torch_module=torch_module)
        image_batch = maybe_move_to_device(
            image_batch,
            device=self._device(torch_module),
            torch_module=torch_module,
        )
        text_batch = tokenizer(ensure_text_sequence(texts, "OpenCLIPExtractor"))
        text_batch = maybe_move_to_device(
            text_batch,
            device=self._device(torch_module),
            torch_module=torch_module,
        )
        projected: Dict[str, Any] = {}
        if any(spec.metadata.get("source") == "image" for spec in self._output_specs):
            projected["image"] = model.encode_image(image_batch)
        if any(spec.metadata.get("source") == "text" for spec in self._output_specs):
            projected["text"] = model.encode_text(text_batch)
        if any(spec.metadata.get("source") == "fused" for spec in self._output_specs):
            if hasattr(model, "get_logits"):
                projected["fused"] = model.get_logits(image_batch, text_batch)
            else:
                projected["fused"] = {
                    "image": projected.get("image"),
                    "text": projected.get("text"),
                }
        return projected

    def _device(self, torch_module: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import open_clip
                import torch
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "OpenCLIPExtractor requires optional OpenCLIP dependencies. "
                    "Install with `poetry install -E openclip`."
                ) from exc
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                **self.model_kwargs,
            )
            if hasattr(model, "to"):
                moved = model.to(self._device(torch))
                if moved is not None:
                    model = moved
            self._torch = torch
            self._image_module = Image
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        return self._torch, self._image_module, self._model, self._preprocess, self._tokenizer


def _normalize_multimodal_samples(
    X: Any,
    input_modalities: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not isinstance(X, dict):
        raise ValueError("OpenCLIPExtractor expects dict inputs with one entry per modality.")
    lengths = {key: len(value) for key, value in X.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Structured inputs must align in length; found {lengths}.")
    if set(X) != set(input_modalities):
        raise ValueError(
            "OpenCLIPExtractor inputs must match declared input_modalities; "
            f"got {sorted(X)} expected {sorted(input_modalities)}."
        )
    size = next(iter(lengths.values()))
    return [{field: X[field][index] for field in input_modalities} for index in range(size)]


def _first_field(input_modalities: Dict[str, str], modality: str) -> str:
    for field_name, field_modality in input_modalities.items():
        if field_modality == modality:
            return field_name
    raise ValueError(f"input_modalities does not include a {modality!r} field.")
