"""SigLIP-style image-text extractor."""

from typing import Any, Dict, List, Optional, Sequence

from vertebrae.extractors.base import EmbeddingOutput, EmbeddingOutputSpec
from vertebrae.extractors.huggingface_multimodal import HFMultimodalExtractor


class SigLIPExtractor:
    """Ergonomic wrapper for SigLIP-style Hugging Face image-text models."""

    def __init__(
        self,
        name: str,
        model_id: str,
        outputs: Optional[Sequence[Dict[str, Any]]] = None,
        image_field: str = "image",
        text_field: str = "text",
        processor_id: Optional[str] = None,
        batch_size: int = 16,
        image_mode: str = "auto",
        alpha_mode: str = "drop",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_outputs = list(
            outputs
            or [
                {"name": "image_branch", "source": "image", "model_output": "image_embeds"},
                {"name": "text_branch", "source": "text", "model_output": "text_embeds"},
            ]
        )
        self._delegate = HFMultimodalExtractor(
            name=name,
            model_id=model_id,
            processor_id=processor_id,
            input_modalities={image_field: "image", text_field: "text"},
            outputs=resolved_outputs,
            batch_size=batch_size,
            image_mode=image_mode,
            alpha_mode=alpha_mode,
            device=device,
            revision=revision,
            trust_remote_code=trust_remote_code,
            processor_kwargs=processor_kwargs,
            model_kwargs=model_kwargs,
        )
        self.name = name
        self.model_id = model_id
        self.modality = "multimodal"
        self.extractor_type = "siglip"
        self.streaming_safe = True
        self.image_field = image_field
        self.text_field = text_field

    def fit(self, X: Any, y: Any = None) -> "SigLIPExtractor":
        self._delegate.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        return self._delegate.transform(X)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self._delegate.fit_transform(X, y)

    def output_specs(self) -> List[EmbeddingOutputSpec]:
        return self._delegate.output_specs()

    def transform_many(self, X: Any) -> List[EmbeddingOutput]:
        return self._delegate.transform_many(X)

    def recipe(self) -> Dict[str, Any]:
        recipe = dict(self._delegate.recipe())
        recipe["extractor_type"] = self.extractor_type
        recipe["image_field"] = self.image_field
        recipe["text_field"] = self.text_field
        return recipe
