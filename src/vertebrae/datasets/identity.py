"""Explicit identities for dataset cache and reproducibility keys."""

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from uuid import uuid4

from vertebrae.cache.fingerprint import canonical_json_exact, hash_json_exact

_IDENTITY_SCHEMA_VERSION = 2
_IDENTITY_MODES = {"declared", "manifest", "content", "ephemeral", "derived"}


@dataclass(frozen=True)
class DatasetIdentity:
    """An explicit, serializable policy for identifying a dataset.

    Use one of the named constructors instead of instantiating this class directly.
    Content identities read all identity-bearing dataset values when first resolved.
    Datasets must be treated as immutable after their identity is resolved or derived.
    """

    mode: str
    dataset_id: Optional[str] = None
    revision: Optional[str] = None
    _manifest_json: Optional[str] = None
    _nonce: Optional[str] = None
    _parent_key: Optional[str] = None
    _operation: Optional[str] = None
    _recipe_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in _IDENTITY_MODES:
            raise ValueError(f"Unsupported dataset identity mode {self.mode!r}.")
        if self.mode == "declared":
            object.__setattr__(self, "dataset_id", _nonempty(self.dataset_id, "dataset_id"))
            object.__setattr__(self, "revision", _nonempty(self.revision, "revision"))
        elif self.mode == "manifest":
            object.__setattr__(self, "dataset_id", _nonempty(self.dataset_id, "dataset_id"))
            if not self._manifest_json:
                raise ValueError("Manifest identities must contain a canonical manifest.")
        elif self.mode == "ephemeral" and not self._nonce:
            raise ValueError("Ephemeral identities must contain a UUID nonce.")
        elif self.mode == "derived":
            object.__setattr__(self, "_parent_key", _nonempty(self._parent_key, "parent_key"))
            object.__setattr__(self, "_operation", _nonempty(self._operation, "operation"))
            object.__setattr__(
                self,
                "_recipe_hash",
                _nonempty(self._recipe_hash, "recipe_hash"),
            )

    @classmethod
    def declared(cls, dataset_id: str, revision: str) -> "DatasetIdentity":
        """Identify a caller-managed dataset revision without inspecting its content."""

        return cls(
            mode="declared",
            dataset_id=_nonempty(dataset_id, "dataset_id"),
            revision=_nonempty(revision, "revision"),
        )

    @classmethod
    def from_manifest(cls, dataset_id: str, manifest: Mapping[str, Any]) -> "DatasetIdentity":
        """Identify a dataset from a complete caller-provided manifest."""

        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping.")
        if not manifest:
            raise ValueError("manifest must not be empty.")
        try:
            manifest_json = canonical_json_exact(dict(manifest))
        except TypeError as exc:
            raise ValueError(
                "Dataset identity manifests must have a stable exact JSON representation."
            ) from exc
        return cls(
            mode="manifest",
            dataset_id=_nonempty(dataset_id, "dataset_id"),
            _manifest_json=manifest_json,
        )

    @classmethod
    def from_content(cls) -> "DatasetIdentity":
        """Explicitly identify a dataset by lazily hashing all of its content."""

        return cls(mode="content")

    @classmethod
    def ephemeral(cls) -> "DatasetIdentity":
        """Create an explicit run-local identity that remains stable when serialized."""

        return cls(mode="ephemeral", _nonce=uuid4().hex)

    @classmethod
    def derived(
        cls,
        parent_key: str,
        operation: str,
        recipe: Any,
    ) -> "DatasetIdentity":
        """Create an identity for a deterministic transformation of another dataset."""

        return cls(
            mode="derived",
            _parent_key=_nonempty(parent_key, "parent_key"),
            _operation=_nonempty(operation, "operation"),
            _recipe_hash=_safe_exact_hash(recipe, "derived dataset identity recipe"),
        )

    def resolve(self, content: Any = None) -> str:
        """Resolve this policy to its stable SHA-256 identity key."""

        if self.mode == "declared":
            payload = {
                "schema_version": _IDENTITY_SCHEMA_VERSION,
                "mode": self.mode,
                "dataset_id": self.dataset_id,
                "revision": self.revision,
            }
        elif self.mode == "manifest":
            payload = {
                "schema_version": _IDENTITY_SCHEMA_VERSION,
                "mode": self.mode,
                "dataset_id": self.dataset_id,
                "manifest_sha256": hashlib.sha256(
                    (self._manifest_json or "").encode("utf-8")
                ).hexdigest(),
            }
        elif self.mode == "ephemeral":
            payload = {
                "schema_version": _IDENTITY_SCHEMA_VERSION,
                "mode": self.mode,
                "nonce": self._nonce,
            }
        elif self.mode == "derived":
            payload = {
                "schema_version": _IDENTITY_SCHEMA_VERSION,
                "mode": self.mode,
                "parent_key": self._parent_key,
                "operation": self._operation,
                "recipe_hash": self._recipe_hash,
            }
        else:
            payload = {
                "schema_version": _IDENTITY_SCHEMA_VERSION,
                "mode": self.mode,
                "content_sha256": _safe_exact_hash(content, "dataset content identity"),
            }
        return hash_json_exact(payload)

    def descriptor(self, resolved_key: str) -> dict[str, Any]:
        """Return compact JSON-safe identity metadata without exposing manifest contents."""

        result: dict[str, Any] = {
            "schema_version": _IDENTITY_SCHEMA_VERSION,
            "mode": self.mode,
            "key": resolved_key,
        }
        if self.dataset_id is not None:
            result["dataset_id"] = self.dataset_id
        if self.revision is not None:
            result["revision"] = self.revision
        if self.mode == "manifest":
            result["manifest_sha256"] = hashlib.sha256(
                (self._manifest_json or "").encode("utf-8")
            ).hexdigest()
        return result


def _nonempty(value: Optional[str], name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _safe_exact_hash(value: Any, purpose: str) -> str:
    try:
        return hash_json_exact(value)
    except TypeError as exc:
        raise ValueError(
            f"Cannot compute {purpose}: {exc} Use DatasetIdentity.declared(...) or "
            "DatasetIdentity.from_manifest(...) for values without a stable exact identity."
        ) from exc
