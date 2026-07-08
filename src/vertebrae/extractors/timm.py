"""Optional timm vision backbone extractor."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vertebrae.extractors._utils import (
    callable_name,
    coerce_image,
    iter_chunks,
    materialize_named_outputs,
    maybe_move_to_device,
    resolve_output_specs,
    spec_to_recipe,
    stack_batch,
)
from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec


class TimmVisionExtractor:
    """Wrap a timm image backbone as a vertebrae extractor."""

    def __init__(
        self,
        name: str,
        model_name: str,
        pretrained: bool = True,
        batch_size: int = 16,
        preprocess_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        image_mode: str = "auto",
        alpha_mode: str = "drop",
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.output_fn = output_fn
        self._output_specs = resolve_output_specs(outputs)
        self.image_mode = image_mode
        self.alpha_mode = alpha_mode
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.data_config = data_config or {}
        self.modality = "image"
        self.extractor_type = "timm"
        self.streaming_safe = True
        self._torch: Any = None
        self._image_module: Any = None
        self._model: Any = None
        self._resolved_preprocess: Optional[Callable[[Any], Any]] = None

    def fit(self, X: Any, y: Any = None) -> "TimmVisionExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = self.transform_many(X)
        if len(outputs) != 1:
            raise ValueError(
                "TimmVisionExtractor.transform() is only available when exactly one output "
                "is configured. Use Benchmark/Evaluator or transform_many()."
            )
        return outputs[0].embeddings

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.transform(X)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return list(self._output_specs)

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        torch_module, image_module, model, preprocess_fn = self._load_model()
        images = list(X)
        model.eval()
        collected: Dict[str, List[np.ndarray]] = {
            spec.name: [] for spec in self._output_specs
        }
        with torch_module.no_grad():
            for chunk in iter_chunks(images, self.batch_size):
                batch = [
                    preprocess_fn(
                        coerce_image(
                            item,
                            image_module=image_module,
                            image_mode=self.image_mode,
                            alpha_mode=self.alpha_mode,
                        )
                    )
                    for item in chunk
                ]
                stacked = stack_batch(batch, torch_module=torch_module)
                stacked = maybe_move_to_device(
                    stacked,
                    device=self._device(torch_module),
                    torch_module=torch_module,
                )
                raw_output = model(stacked)
                projected = self.output_fn(raw_output) if self.output_fn is not None else raw_output
                outputs = materialize_named_outputs(
                    projected,
                    self._output_specs,
                    owner=f"TimmVisionExtractor '{self.name}'",
                    allow_sparse=False,
                )
                for output in outputs:
                    collected[output.name].append(output.embeddings.astype(np.float32, copy=False))
        return [
            EmbeddingOutput(
                name=spec.name,
                embeddings=np.vstack(collected[spec.name]).astype(np.float32, copy=False),
                recipe=spec_to_recipe(spec),
                metadata=dict(spec.metadata),
            )
            for spec in self._output_specs
        ]

    def recipe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extractor_type": self.extractor_type,
            "modality": self.modality,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "batch_size": self.batch_size,
            "preprocess_fn": (
                callable_name(self.preprocess_fn)
                if self.preprocess_fn is not None
                else "<timm-default>"
            ),
            "output_fn": callable_name(self.output_fn) if self.output_fn is not None else None,
            "outputs": [
                {
                    "name": spec.name,
                    "selector": spec.metadata.get("selector"),
                    "flatten": spec.metadata.get("flatten", True),
                }
                for spec in self._output_specs
            ],
            "image_mode": self.image_mode,
            "alpha_mode": self.alpha_mode,
            "device": self.device,
            "model_kwargs": self.model_kwargs,
            "data_config": self.data_config,
            "streaming_safe": self.streaming_safe,
        }

    def _device(self, torch_module: Any) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import timm
                import torch
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "TimmVisionExtractor requires optional timm vision dependencies. "
                    "Install with `poetry install -E timm`."
                ) from exc
            model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                **self.model_kwargs,
            )
            if hasattr(model, "to"):
                moved = model.to(self._device(torch))
                if moved is not None:
                    model = moved
            preprocess_fn = self.preprocess_fn
            if preprocess_fn is None:
                config = dict(self.data_config)
                if (
                    not config
                    and hasattr(timm, "data")
                    and hasattr(timm.data, "resolve_model_data_config")
                ):
                    config = dict(timm.data.resolve_model_data_config(model))
                preprocess_fn = timm.data.create_transform(**config)
            self._torch = torch
            self._image_module = Image
            self._model = model
            self._resolved_preprocess = preprocess_fn
        return self._torch, self._image_module, self._model, self._resolved_preprocess
