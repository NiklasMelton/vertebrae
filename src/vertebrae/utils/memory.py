"""Memory estimation and admission helpers."""

import os
import pickle
import sqlite3
import sys
import tempfile
import weakref
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import psutil

from vertebrae.config import MemoryConfig
from vertebrae.utils.validation import estimate_dense_nbytes, is_sparse_matrix


@dataclass(frozen=True)
class MemoryBudget:
    """Resolved memory budget from a `MemoryConfig`.

    Attributes:
        total_bytes: Total system memory.
        available_bytes: Currently available system memory.
        reserve_system_bytes: Bytes reserved for system use.
        max_memory_bytes: Maximum bytes vertebrae should use for planned work.
    """

    total_bytes: int
    available_bytes: int
    reserve_system_bytes: int
    max_memory_bytes: int


@dataclass(frozen=True)
class EmbeddingMemoryEstimate:
    """Estimated memory footprint for an embedding artifact.

    Attributes:
        n_samples: Number of embedding rows.
        embedding_dim: Number of embedding columns.
        dtype: Embedding dtype string.
        resident_bytes: Estimated bytes to hold the embedding artifact in memory.
        dense_scoring_bytes: Estimated dense bytes required by scoring.
        batch_embedding_bytes: Estimated bytes for one embedding batch.
        strategy: Planned strategy: `"in_memory"` or `"stream_to_disk"`.
    """

    n_samples: int
    embedding_dim: int
    dtype: str
    resident_bytes: int
    dense_scoring_bytes: int
    batch_embedding_bytes: int
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate for result metadata.

        Returns:
            JSON-compatible memory estimate.
        """

        return {
            "n_samples": self.n_samples,
            "embedding_dim": self.embedding_dim,
            "dtype": self.dtype,
            "resident_bytes": self.resident_bytes,
            "dense_scoring_bytes": self.dense_scoring_bytes,
            "batch_embedding_bytes": self.batch_embedding_bytes,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class MatrixAssembly:
    """A matrix assembled either in memory or through a temporary local memmap."""

    matrix: Any
    strategy: str
    required_bytes: int
    budget_bytes: int
    staging_strategy: str = "none"


@dataclass(frozen=True)
class MatrixRowReference:
    """Opaque reference to one incrementally staged matrix row."""

    token: int
    output_name: str = ""
    width: int = 0
    dtype: str = ""
    sparse: bool = False
    resident_bytes: int = 0
    dense_offset: int = 0
    nnz_offset: int = 0
    nnz: int = 0


@dataclass(frozen=True)
class MetadataRowReference:
    """Opaque reference to one incrementally staged metadata record."""

    token: int


@dataclass
class _MetadataRowEntry:
    output_name: str
    ordinal: int
    row: Dict[str, Any]
    resident_bytes: int
    priority_key: str = ""
    group_key: str = ""
    row_key: int = 0
    column_key: int = 0
    selected: bool = False


@dataclass
class _MatrixRowEntry:
    output_name: str
    width: int
    dtype: np.dtype[Any]
    sparse: bool
    resident_bytes: int
    dense_offset: int = 0
    nnz_offset: int = 0
    nnz: int = 0
    matrix: Any = None


@dataclass
class _DiskOutputStage:
    width: int
    dtype: np.dtype[Any]
    sparse: bool
    values_path: Path
    indices_path: Optional[Path] = None
    total_rows: int = 0
    total_nnz: int = 0


class IncrementalMatrixStager:
    """Stage compatible matrix rows without retaining extractor batch buffers.

    When disk spill is enabled, every row is written immediately to an
    append-only per-output staging file. This keeps memory bounded even when a
    streaming extractor emits several named outputs. When spill is disabled,
    rows remain in memory only while their aggregate footprint fits the
    configured budget; otherwise appending fails before another row is retained.
    """

    def __init__(self, memory_config: MemoryConfig, *, purpose: str) -> None:
        self.memory_config = memory_config
        self.purpose = purpose
        self.budget = resolve_memory_budget(memory_config)
        self._entries: Dict[int, _MatrixRowEntry] = {}
        self._tokens_by_output: Dict[str, List[int]] = {}
        self._disk_outputs: Dict[str, _DiskOutputStage] = {}
        self._resident_bytes = 0
        self._reserved_metadata_bytes = 0
        self._fixed_resident_bytes = int(memory_config.model_memory_bytes) + int(
            memory_config.raw_batch_memory_bytes
        )
        self._next_token = 0
        self._next_stage_id = 0
        self._temporary_directory: Any = None
        self._closed = False
        if self._fixed_resident_bytes > self.budget.max_memory_bytes:
            _raise_assembly_budget_error(
                f"{purpose} model and raw-batch memory hints",
                self._fixed_resident_bytes,
                self.budget.max_memory_bytes,
            )
        if memory_config.allow_disk_spill:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="vertebrae-row-staging-")

    def __enter__(self) -> "IncrementalMatrixStager":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def append(self, output_name: str, row: Any) -> MatrixRowReference:
        """Append one dense or sparse matrix row and return its lightweight reference."""

        if self._closed:
            raise RuntimeError("Cannot append to a closed matrix stager.")
        sparse = is_sparse_matrix(row)
        shape = getattr(row, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[0]) != 1 or int(shape[1]) < 1:
            raise ValueError(f"{self.purpose} rows must have shape [1, embedding_dim].")
        width = int(shape[1])
        dtype = np.dtype(row.dtype if sparse else np.asarray(row).dtype)
        matrix_bytes = estimate_matrix_resident_bytes(row)
        resident_bytes = matrix_bytes + 256
        token = self._next_token
        self._next_token += 1
        entry = _MatrixRowEntry(
            output_name=output_name,
            width=width,
            dtype=dtype,
            sparse=sparse,
            resident_bytes=resident_bytes,
            nnz=int(getattr(row, "nnz", 0)),
        )
        if self.memory_config.allow_disk_spill:
            self._append_to_disk(output_name, row, entry)
        else:
            required = (
                self._fixed_resident_bytes
                + self._resident_bytes
                + self._reserved_metadata_bytes
                + resident_bytes
            )
            if required > self.budget.max_memory_bytes:
                _raise_assembly_budget_error(
                    self.purpose,
                    required,
                    self.budget.max_memory_bytes,
                )
            entry.matrix = _copy_matrix_row(row, sparse=sparse)
            self._resident_bytes += resident_bytes
            self._entries[token] = entry
            self._tokens_by_output.setdefault(output_name, []).append(token)
        return _matrix_reference(token, entry)

    def assemble(
        self,
        output_name: str,
        references: Iterable[MatrixRowReference],
        *,
        purpose: str,
        force_disk: bool = False,
    ) -> MatrixAssembly:
        """Assemble selected rows for one output and release that output's staging data."""

        if force_disk and not self.memory_config.allow_disk_spill:
            raise ValueError(f"{purpose} cannot force disk assembly when disk spill is disabled.")

        if self.memory_config.allow_disk_spill:
            assembly = self._assemble_disk_references(
                output_name,
                references,
                purpose=purpose,
                force_disk=force_disk,
            )
        else:
            assembly = self._assemble_memory_references(
                output_name,
                references,
                purpose=purpose,
            )
        self.discard_output(output_name)
        return assembly

    def reserve_metadata(self, required_bytes: int, *, purpose: str) -> None:
        """Reserve retained metadata bytes against the shared no-spill budget."""

        if self.memory_config.allow_disk_spill:
            return
        required = (
            self._fixed_resident_bytes
            + self._resident_bytes
            + self._reserved_metadata_bytes
            + required_bytes
        )
        if required > self.budget.max_memory_bytes:
            _raise_assembly_budget_error(purpose, required, self.budget.max_memory_bytes)
        self._reserved_metadata_bytes += required_bytes

    def release_metadata(self, released_bytes: int) -> None:
        """Release metadata bytes previously reserved by a paired stager."""

        if self.memory_config.allow_disk_spill:
            return
        self._reserved_metadata_bytes = max(
            0,
            self._reserved_metadata_bytes - int(released_bytes),
        )

    def discard_output(self, output_name: str) -> None:
        """Release all staged candidate rows for one named output."""

        if not self.memory_config.allow_disk_spill:
            for token in self._tokens_by_output.pop(output_name, []):
                entry = self._entries.pop(token, None)
                if entry is not None and entry.matrix is not None:
                    self._resident_bytes -= entry.resident_bytes
        stage = self._disk_outputs.pop(output_name, None)
        if stage is not None:
            stage.values_path.unlink(missing_ok=True)
            if stage.indices_path is not None:
                stage.indices_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Remove all temporary candidate staging files and release row references."""

        if self._closed:
            return
        self._closed = True
        self._entries.clear()
        self._tokens_by_output.clear()
        self._disk_outputs.clear()
        self._resident_bytes = 0
        self._reserved_metadata_bytes = 0
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def _entry(self, reference: MatrixRowReference, output_name: str) -> _MatrixRowEntry:
        if self.memory_config.allow_disk_spill:
            if not isinstance(reference, MatrixRowReference):
                raise ValueError(f"{self.purpose} received an invalid staged row reference.")
            if reference.output_name != output_name:
                raise ValueError(
                    f"Staged row belongs to output {reference.output_name!r}, "
                    f"not {output_name!r}."
                )
            return _MatrixRowEntry(
                output_name=reference.output_name,
                width=reference.width,
                dtype=np.dtype(reference.dtype),
                sparse=reference.sparse,
                resident_bytes=reference.resident_bytes,
                dense_offset=reference.dense_offset,
                nnz_offset=reference.nnz_offset,
                nnz=reference.nnz,
            )
        try:
            entry = self._entries[reference.token]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"{self.purpose} received an unknown staged row reference.") from exc
        if entry.output_name != output_name:
            raise ValueError(
                f"Staged row belongs to output {entry.output_name!r}, not {output_name!r}."
            )
        return entry

    def _assemble_disk_references(
        self,
        output_name: str,
        references: Iterable[MatrixRowReference],
        *,
        purpose: str,
        force_disk: bool,
    ) -> MatrixAssembly:
        stage = self._disk_outputs.get(output_name)
        if stage is None:
            raise ValueError(f"{purpose} requires at least one matrix row.")
        root = self._temporary_directory.name if self._temporary_directory is not None else None
        entry_file = tempfile.TemporaryFile(mode="w+b", dir=root)
        first: Optional[_MatrixRowEntry] = None
        n_rows = 0
        total_nnz = 0
        try:
            for reference in references:
                entry = self._entry(reference, output_name)
                if first is None:
                    first = entry
                else:
                    _validate_entry_pair(first, entry, purpose=purpose)
                pickle.dump(entry, entry_file, protocol=pickle.HIGHEST_PROTOCOL)
                n_rows += 1
                total_nnz += entry.nnz
            if first is None:
                raise ValueError(f"{purpose} requires at least one matrix row.")
            required = _assembled_shape_bytes(first, n_rows=n_rows, total_nnz=total_nnz)
            use_memmap = (
                self._fixed_resident_bytes + self._reserved_metadata_bytes + required
                > self.budget.max_memory_bytes
                or force_disk
            )
            if stage.sparse:
                matrix = _assemble_staged_sparse_entries(
                    stage,
                    _iter_pickled_entries(entry_file),
                    n_rows=n_rows,
                    total_nnz=total_nnz,
                    use_memmap=use_memmap,
                )
            else:
                matrix = _assemble_staged_dense_entries(
                    stage,
                    _iter_pickled_entries(entry_file),
                    n_rows=n_rows,
                    use_memmap=use_memmap,
                )
            return MatrixAssembly(
                matrix=matrix,
                strategy="disk_spill" if use_memmap else "in_memory",
                required_bytes=required,
                budget_bytes=self.budget.max_memory_bytes,
                staging_strategy="disk",
            )
        finally:
            entry_file.close()

    def _assemble_memory_references(
        self,
        output_name: str,
        references: Iterable[MatrixRowReference],
        *,
        purpose: str,
    ) -> MatrixAssembly:
        entries: List[_MatrixRowEntry] = []
        transient_bytes = 0
        first: Optional[_MatrixRowEntry] = None
        try:
            for reference in references:
                entry = self._entry(reference, output_name)
                if first is None:
                    first = entry
                else:
                    _validate_entry_pair(first, entry, purpose=purpose)
                self.reserve_metadata(64, purpose=f"{purpose} row-reference assembly")
                transient_bytes += 64
                entries.append(entry)
            if first is None:
                raise ValueError(f"{purpose} requires at least one matrix row.")
            required = _assembled_entry_bytes(entries)
            peak_required = (
                self._fixed_resident_bytes
                + self._resident_bytes
                + self._reserved_metadata_bytes
                + required
            )
            if peak_required > self.budget.max_memory_bytes:
                _raise_assembly_budget_error(
                    purpose,
                    peak_required,
                    self.budget.max_memory_bytes,
                )
            assembled = assemble_matrix_rows(
                [entry.matrix for entry in entries],
                self.memory_config,
                purpose=purpose,
            )
            return MatrixAssembly(
                matrix=assembled.matrix,
                strategy=assembled.strategy,
                required_bytes=assembled.required_bytes,
                budget_bytes=assembled.budget_bytes,
                staging_strategy="memory",
            )
        finally:
            self.release_metadata(transient_bytes)

    def _append_to_disk(
        self,
        output_name: str,
        row: Any,
        entry: _MatrixRowEntry,
    ) -> None:
        stage = self._disk_outputs.get(output_name)
        if stage is None:
            if self._temporary_directory is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Disk staging directory was not initialized.")
            stage_id = self._next_stage_id
            self._next_stage_id += 1
            root = Path(self._temporary_directory.name)
            values_path = root / f"output-{stage_id}-values.bin"
            values_path.touch(exist_ok=False)
            indices_path = None
            if entry.sparse:
                indices_path = root / f"output-{stage_id}-indices.bin"
                indices_path.touch(exist_ok=False)
            stage = _DiskOutputStage(
                width=entry.width,
                dtype=entry.dtype,
                sparse=entry.sparse,
                values_path=values_path,
                indices_path=indices_path,
            )
            self._disk_outputs[output_name] = stage
        _validate_stage_contract(stage, entry, output_name=output_name)
        if entry.sparse:
            from scipy import sparse as scipy_sparse

            sparse_row = scipy_sparse.csr_matrix(row, dtype=entry.dtype, copy=True)
            sparse_row.sum_duplicates()
            sparse_row.sort_indices()
            values = np.ascontiguousarray(sparse_row.data)
            indices = np.ascontiguousarray(sparse_row.indices, dtype=np.int64)
            entry.nnz_offset = stage.total_nnz
            entry.nnz = int(sparse_row.nnz)
            with stage.values_path.open("ab") as values_file:
                values.tofile(values_file)
            if stage.indices_path is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Sparse staging is missing its index file.")
            with stage.indices_path.open("ab") as indices_file:
                indices.tofile(indices_file)
            stage.total_nnz += entry.nnz
        else:
            values = np.ascontiguousarray(np.asarray(row))
            entry.dense_offset = stage.total_rows
            with stage.values_path.open("ab") as values_file:
                values.tofile(values_file)
        stage.total_rows += 1


class IncrementalMetadataStager:
    """Stage per-row metadata without retaining an unbounded candidate list.

    Disk-spill mode writes each record directly to an append-only pickle stream
    and stores only ordering and selection indexes in SQLite. Iteration loads one
    record at a time. No-spill mode retains records only after reserving their
    estimated resident size against the paired matrix stager's memory budget.
    """

    _VALID_ORDERS = {"ordinal", "priority", "final"}

    def __init__(
        self,
        memory_config: MemoryConfig,
        *,
        purpose: str,
        matrix_stager: IncrementalMatrixStager,
    ) -> None:
        self.memory_config = memory_config
        self.purpose = purpose
        self.matrix_stager = matrix_stager
        self.strategy = "disk" if memory_config.allow_disk_spill else "memory"
        self._entries: Dict[int, _MetadataRowEntry] = {}
        self._tokens_by_output: Dict[str, List[int]] = {}
        self._resident_by_output: Dict[str, int] = {}
        self._memory_counters: Dict[Tuple[str, str, str], int] = {}
        self._memory_members: Dict[Tuple[str, str, str], set[str]] = {}
        self._next_token = 0
        self._next_ordinal_by_output: Dict[str, int] = {}
        self._temporary_directory: Any = None
        self._connection: Optional[sqlite3.Connection] = None
        self._data_file: Any = None
        self._closed = False
        if memory_config.allow_disk_spill:
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="vertebrae-metadata-staging-"
            )
            root = Path(self._temporary_directory.name)
            self._data_file = (root / "records.pkl").open("w+b")
            self._connection = sqlite3.connect(root / "index.sqlite3")
            self._initialize_database()

    def __enter__(self) -> "IncrementalMetadataStager":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def resident_bytes(self) -> int:
        """Estimated retained metadata bytes in no-spill mode."""

        return sum(self._resident_by_output.values())

    def append(
        self,
        output_name: str,
        row: Dict[str, Any],
        *,
        priority_key: str = "",
        group_key: str = "",
        row_key: int = 0,
        column_key: int = 0,
    ) -> MetadataRowReference:
        """Append one metadata record with optional deterministic ordering keys."""

        if self._closed:
            raise RuntimeError("Cannot append to a closed metadata stager.")
        token = self._next_token
        self._next_token += 1
        ordinal = self._next_ordinal_by_output.get(output_name, 0)
        self._next_ordinal_by_output[output_name] = ordinal + 1
        reference = MetadataRowReference(token=token)
        if self.memory_config.allow_disk_spill:
            connection, data_file = self._disk_resources()
            data_file.seek(0, os.SEEK_END)
            offset = int(data_file.tell())
            pickle.dump(row, data_file, protocol=pickle.HIGHEST_PROTOCOL)
            connection.execute(
                """
                INSERT INTO records(
                    token, output_name, ordinal, offset, priority_key,
                    group_key, row_key, column_key, selected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    token,
                    output_name,
                    ordinal,
                    offset,
                    str(priority_key),
                    str(group_key),
                    int(row_key),
                    int(column_key),
                ),
            )
        else:
            entry = _MetadataRowEntry(
                output_name=output_name,
                ordinal=ordinal,
                row=row,
                resident_bytes=0,
                priority_key=str(priority_key),
                group_key=str(group_key),
                row_key=int(row_key),
                column_key=int(column_key),
            )
            entry.resident_bytes = estimate_object_resident_bytes(entry) + 192
            self._reserve(output_name, entry.resident_bytes)
            self._entries[token] = entry
            self._tokens_by_output.setdefault(output_name, []).append(token)
        return reference

    def iter_rows(
        self,
        output_name: str,
        *,
        order: str = "ordinal",
        selected_only: bool = False,
    ) -> Iterator[Tuple[MetadataRowReference, Dict[str, Any]]]:
        """Iterate records one at a time in insertion, priority, or final order."""

        if order not in self._VALID_ORDERS:
            raise ValueError(f"Unknown metadata staging order {order!r}.")
        if self.memory_config.allow_disk_spill:
            connection, data_file = self._disk_resources()
            data_file.flush()
            predicates = "output_name = ?"
            parameters: List[Any] = [output_name]
            if selected_only:
                predicates += " AND selected = 1"
            order_sql = {
                "ordinal": "ordinal",
                "priority": "priority_key, ordinal",
                "final": "group_key, row_key, column_key, ordinal",
            }[order]
            cursor = connection.execute(
                f"SELECT token, offset FROM records WHERE {predicates} ORDER BY {order_sql}",
                parameters,
            )
            for token, offset in cursor:
                data_file.seek(int(offset))
                row = pickle.load(data_file)
                yield MetadataRowReference(int(token)), row
            return

        entries: Iterable[Tuple[int, _MetadataRowEntry]] = (
            (token, self._entries[token])
            for token in self._tokens_by_output.get(output_name, [])
            if not selected_only or self._entries[token].selected
        )
        transient_sort_bytes = 0
        if order != "ordinal":
            transient_sort_bytes = (
                self.count_rows(
                    output_name,
                    selected_only=selected_only,
                )
                * 80
            )
            self.matrix_stager.reserve_metadata(
                transient_sort_bytes,
                purpose=f"{self.purpose} deterministic ordering",
            )

            def sort_key(item: Tuple[int, _MetadataRowEntry]) -> Tuple[Any, ...]:
                if order == "priority":
                    return (item[1].priority_key, item[1].ordinal)
                return (
                    item[1].group_key,
                    item[1].row_key,
                    item[1].column_key,
                    item[1].ordinal,
                )

            entries = sorted(entries, key=sort_key)
        try:
            for token, entry in entries:
                yield MetadataRowReference(token), entry.row
        finally:
            if transient_sort_bytes:
                self.matrix_stager.release_metadata(transient_sort_bytes)

    def count_rows(self, output_name: str, *, selected_only: bool = False) -> int:
        """Return the number of staged records for an output."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            predicate = "output_name = ?"
            if selected_only:
                predicate += " AND selected = 1"
            row = connection.execute(
                f"SELECT COUNT(*) FROM records WHERE {predicate}",
                (output_name,),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        return sum(
            1
            for token in self._tokens_by_output.get(output_name, [])
            if not selected_only or self._entries[token].selected
        )

    def mark_selected(self, reference: MetadataRowReference, output_name: str) -> None:
        """Mark one staged record as retained by deterministic filtering."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            cursor = connection.execute(
                "UPDATE records SET selected = 1 WHERE token = ? AND output_name = ?",
                (reference.token, output_name),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"{self.purpose} received an unknown metadata reference.")
            return
        entry = self._entries.get(reference.token)
        if entry is None or entry.output_name != output_name:
            raise ValueError(f"{self.purpose} received an unknown metadata reference.")
        entry.selected = True

    def counter_value(self, output_name: str, namespace: str, key: str) -> int:
        """Read a bounded or disk-backed selection counter."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            row = connection.execute(
                """
                SELECT value FROM counters
                WHERE output_name = ? AND namespace = ? AND key = ?
                """,
                (output_name, namespace, key),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        return self._memory_counters.get((output_name, namespace, key), 0)

    def increment_counter(
        self,
        output_name: str,
        namespace: str,
        key: str,
        amount: int = 1,
    ) -> int:
        """Increment a bounded or disk-backed selection counter."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            connection.execute(
                """
                INSERT INTO counters(output_name, namespace, key, value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(output_name, namespace, key)
                DO UPDATE SET value = value + excluded.value
                """,
                (output_name, namespace, key, int(amount)),
            )
            return self.counter_value(output_name, namespace, key)
        state_key = (output_name, namespace, key)
        if state_key not in self._memory_counters:
            self._reserve(output_name, estimate_object_resident_bytes(state_key) + 64)
        value = self._memory_counters.get(state_key, 0) + int(amount)
        self._memory_counters[state_key] = value
        return value

    def add_member(
        self,
        output_name: str,
        namespace: str,
        group_key: str,
        member_key: str,
    ) -> bool:
        """Add one exact membership value and report whether it was new."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO members(
                    output_name, namespace, group_key, member_key
                ) VALUES (?, ?, ?, ?)
                """,
                (output_name, namespace, group_key, member_key),
            )
            return cursor.rowcount == 1
        state_key = (output_name, namespace, group_key)
        members = self._memory_members.get(state_key)
        if members is None:
            self._reserve(output_name, estimate_object_resident_bytes(state_key) + 216)
            members = set()
            self._memory_members[state_key] = members
        if member_key in members:
            return False
        self._reserve(output_name, estimate_object_resident_bytes(member_key) + 32)
        members.add(member_key)
        return True

    def has_member(
        self,
        output_name: str,
        namespace: str,
        group_key: str,
        member_key: str,
    ) -> bool:
        """Return whether an exact membership value has been observed."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            row = connection.execute(
                """
                SELECT 1 FROM members WHERE output_name = ? AND namespace = ?
                AND group_key = ? AND member_key = ?
                """,
                (output_name, namespace, group_key, member_key),
            ).fetchone()
            return row is not None
        return member_key in self._memory_members.get(
            (output_name, namespace, group_key),
            set(),
        )

    def member_count(self, output_name: str, namespace: str, group_key: str = "") -> int:
        """Count exact members in one state namespace and group."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            row = connection.execute(
                """
                SELECT COUNT(*) FROM members WHERE output_name = ?
                AND namespace = ? AND group_key = ?
                """,
                (output_name, namespace, group_key),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        return len(self._memory_members.get((output_name, namespace, group_key), set()))

    def discard_output(self, output_name: str) -> None:
        """Release all records and selection state for one output."""

        if self.memory_config.allow_disk_spill:
            connection, _ = self._disk_resources()
            connection.execute("DELETE FROM records WHERE output_name = ?", (output_name,))
            connection.execute("DELETE FROM counters WHERE output_name = ?", (output_name,))
            connection.execute("DELETE FROM members WHERE output_name = ?", (output_name,))
            connection.commit()
        else:
            for token in self._tokens_by_output.pop(output_name, []):
                self._entries.pop(token, None)
            for state_key in [key for key in self._memory_counters if key[0] == output_name]:
                self._memory_counters.pop(state_key, None)
            for state_key in [key for key in self._memory_members if key[0] == output_name]:
                self._memory_members.pop(state_key, None)
            released = self._resident_by_output.pop(output_name, 0)
            self.matrix_stager.release_metadata(released)
        self._next_ordinal_by_output.pop(output_name, None)

    def close(self) -> None:
        """Release retained records and remove all temporary spill files."""

        if self._closed:
            return
        self._closed = True
        if not self.memory_config.allow_disk_spill:
            released = sum(self._resident_by_output.values())
            self.matrix_stager.release_metadata(released)
        self._entries.clear()
        self._tokens_by_output.clear()
        self._resident_by_output.clear()
        self._memory_counters.clear()
        self._memory_members.clear()
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def _reserve(self, output_name: str, required_bytes: int) -> None:
        self.matrix_stager.reserve_metadata(required_bytes, purpose=self.purpose)
        self._resident_by_output[output_name] = (
            self._resident_by_output.get(output_name, 0) + required_bytes
        )

    def _disk_resources(self) -> Tuple[sqlite3.Connection, Any]:
        if self._connection is None or self._data_file is None:
            raise RuntimeError("Disk metadata staging is unavailable.")
        return self._connection, self._data_file

    def _initialize_database(self) -> None:
        connection, _ = self._disk_resources()
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = FILE;
            PRAGMA cache_size = -64;
            CREATE TABLE records(
                token INTEGER PRIMARY KEY,
                output_name TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                priority_key TEXT NOT NULL,
                group_key TEXT NOT NULL,
                row_key INTEGER NOT NULL,
                column_key INTEGER NOT NULL,
                selected INTEGER NOT NULL
            );
            CREATE INDEX records_ordinal
                ON records(output_name, ordinal);
            CREATE INDEX records_priority
                ON records(output_name, priority_key, ordinal);
            CREATE INDEX records_final
                ON records(output_name, selected, group_key, row_key, column_key, ordinal);
            CREATE TABLE counters(
                output_name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value INTEGER NOT NULL,
                PRIMARY KEY(output_name, namespace, key)
            );
            CREATE TABLE members(
                output_name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                group_key TEXT NOT NULL,
                member_key TEXT NOT NULL,
                PRIMARY KEY(output_name, namespace, group_key, member_key)
            );
            """
        )


class IncrementalMatrixReferenceStager:
    """Store ordered matrix-row references without an unbounded Python list.

    Spill-enabled runs keep reference payloads and ordering indexes in the
    metadata stager's disk-backed SQLite/pickle storage. No-spill runs charge
    reference records, exact-position membership, and deterministic sorting to
    the paired matrix stager's memory budget.
    """

    _POSITION_NAMESPACE = "matrix_row_position"

    def __init__(
        self,
        memory_config: MemoryConfig,
        *,
        purpose: str,
        matrix_stager: IncrementalMatrixStager,
    ) -> None:
        self.purpose = purpose
        self.matrix_stager = matrix_stager
        self._metadata_stager = IncrementalMetadataStager(
            memory_config,
            purpose=f"{purpose} row-reference ordering",
            matrix_stager=matrix_stager,
        )

    def __enter__(self) -> "IncrementalMatrixReferenceStager":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def strategy(self) -> str:
        """Return ``disk`` or ``memory`` for reference bookkeeping."""

        return self._metadata_stager.strategy

    @property
    def resident_bytes(self) -> int:
        """Return charged resident reference bytes in no-spill mode."""

        return self._metadata_stager.resident_bytes

    def count_rows(self, output_name: str) -> int:
        """Return the number of staged references for one output."""

        return self._metadata_stager.count_rows(output_name)

    def append(
        self,
        output_name: str,
        position: int,
        reference: MatrixRowReference,
    ) -> None:
        """Stage one unique matrix reference at its final sample position."""

        if isinstance(position, bool) or not isinstance(position, Integral):
            raise TypeError(f"{self.purpose} row positions must be integers.")
        normalized_position = int(position)
        if normalized_position < 0:
            raise ValueError(f"{self.purpose} row positions must be non-negative.")
        if not isinstance(reference, MatrixRowReference):
            raise TypeError(f"{self.purpose} requires MatrixRowReference values.")
        if reference.output_name and reference.output_name != output_name:
            raise ValueError(
                f"Staged row belongs to output {reference.output_name!r}, " f"not {output_name!r}."
            )
        position_key = str(normalized_position)
        if not self._metadata_stager.add_member(
            output_name,
            self._POSITION_NAMESPACE,
            "",
            position_key,
        ):
            raise ValueError(
                "Duplicate embedding rows for sample index "
                f"{normalized_position} in output {output_name!r}."
            )
        self._metadata_stager.append(
            output_name,
            {
                "position": normalized_position,
                "reference": reference,
            },
            row_key=normalized_position,
        )

    def iter_references(
        self,
        output_name: str,
        *,
        expected_rows: int,
    ) -> Iterator[MatrixRowReference]:
        """Yield references in exact sample order after validating full coverage."""

        if isinstance(expected_rows, bool) or not isinstance(expected_rows, Integral):
            raise TypeError(f"{self.purpose} expected_rows must be an integer.")
        normalized_expected = int(expected_rows)
        if normalized_expected < 1:
            raise ValueError(f"{self.purpose} expected_rows must be >= 1.")
        expected_position = 0
        for _, row in self._metadata_stager.iter_rows(output_name, order="final"):
            position = row.get("position")
            reference = row.get("reference")
            if isinstance(position, bool) or not isinstance(position, Integral):
                raise ValueError(f"{self.purpose} contains an invalid staged row position.")
            normalized_position = int(position)
            if normalized_position >= normalized_expected:
                raise ValueError(
                    f"{self.purpose} output {output_name!r} row position "
                    f"{normalized_position} is outside the expected range "
                    f"[0, {normalized_expected})."
                )
            if normalized_position != expected_position:
                if normalized_position < expected_position:
                    raise ValueError(
                        f"{self.purpose} output {output_name!r} contains duplicate or "
                        f"out-of-order row position {normalized_position}."
                    )
                raise ValueError(
                    f"{self.purpose} output {output_name!r} did not cover every sample; missing "
                    f"{_position_preview(expected_position, normalized_position)}."
                )
            if not isinstance(reference, MatrixRowReference):
                raise ValueError(f"{self.purpose} contains an invalid matrix row reference.")
            yield reference
            expected_position += 1
        if expected_position != normalized_expected:
            raise ValueError(
                f"{self.purpose} output {output_name!r} did not cover every sample; missing "
                f"{_position_preview(expected_position, normalized_expected)}."
            )

    def assemble(
        self,
        output_name: str,
        *,
        expected_rows: int,
        purpose: str,
        force_disk: bool = False,
    ) -> MatrixAssembly:
        """Assemble one output in sample order and release its reference state."""

        try:
            return self.matrix_stager.assemble(
                output_name,
                self.iter_references(output_name, expected_rows=expected_rows),
                purpose=purpose,
                force_disk=force_disk,
            )
        finally:
            self._metadata_stager.discard_output(output_name)

    def close(self) -> None:
        """Release all in-memory or disk-backed reference state."""

        self._metadata_stager.close()


def _position_preview(start: int, stop: int, *, limit: int = 10) -> List[int]:
    return list(range(start, min(stop, start + limit)))


def _copy_matrix_row(row: Any, *, sparse: bool) -> Any:
    if sparse:
        return row.tocsr(copy=True)
    return np.array(row, copy=True, order="C")


def _matrix_reference(token: int, entry: _MatrixRowEntry) -> MatrixRowReference:
    return MatrixRowReference(
        token=token,
        output_name=entry.output_name,
        width=entry.width,
        dtype=str(entry.dtype),
        sparse=entry.sparse,
        resident_bytes=entry.resident_bytes,
        dense_offset=entry.dense_offset,
        nnz_offset=entry.nnz_offset,
        nnz=entry.nnz,
    )


def _validate_stage_contract(
    stage: _DiskOutputStage,
    entry: _MatrixRowEntry,
    *,
    output_name: str,
) -> None:
    if stage.width != entry.width or stage.dtype != entry.dtype or stage.sparse != entry.sparse:
        raise ValueError(
            f"Output {output_name!r} changed staged matrix contract; expected width "
            f"{stage.width}, dtype {stage.dtype}, and sparse={stage.sparse}, received "
            f"width {entry.width}, dtype {entry.dtype}, and sparse={entry.sparse}."
        )


def _validate_entry_pair(
    first: _MatrixRowEntry,
    entry: _MatrixRowEntry,
    *,
    purpose: str,
) -> None:
    if entry.width != first.width or entry.dtype != first.dtype or entry.sparse != first.sparse:
        raise ValueError(f"{purpose} rows must have one consistent matrix contract.")


def _assembled_entry_bytes(entries: List[_MatrixRowEntry]) -> int:
    first = entries[0]
    nnz = sum(entry.nnz for entry in entries)
    return _assembled_shape_bytes(first, n_rows=len(entries), total_nnz=nnz)


def _assembled_shape_bytes(
    first: _MatrixRowEntry,
    *,
    n_rows: int,
    total_nnz: int,
) -> int:
    if not first.sparse:
        return n_rows * first.width * first.dtype.itemsize
    index_bytes = np.dtype(np.int64).itemsize
    return total_nnz * (first.dtype.itemsize + index_bytes) + (n_rows + 1) * index_bytes


def _iter_pickled_entries(entry_file: Any) -> Iterator[_MatrixRowEntry]:
    entry_file.seek(0)
    while True:
        try:
            entry = pickle.load(entry_file)
        except EOFError:
            return
        if not isinstance(entry, _MatrixRowEntry):  # pragma: no cover - internal invariant
            raise RuntimeError("Matrix row staging index contains an invalid entry.")
        yield entry


def _assemble_staged_dense_entries(
    stage: _DiskOutputStage,
    entries: Iterable[_MatrixRowEntry],
    *,
    n_rows: int,
    use_memmap: bool,
) -> np.ndarray:
    shape = (n_rows, stage.width)
    final_path = None
    matrix: np.ndarray
    if use_memmap:
        matrix, final_path = _new_npy_memmap(dtype=stage.dtype, shape=shape)
    else:
        matrix = np.empty(shape, dtype=stage.dtype)
    source = None
    try:
        source = np.memmap(
            stage.values_path,
            mode="r",
            dtype=stage.dtype,
            shape=(stage.total_rows, stage.width),
        )
        for row_index, entry in enumerate(entries):
            matrix[row_index] = source[entry.dense_offset]
        _flush_memmaps(matrix)
    except Exception:
        if final_path is not None:
            final_path.unlink(missing_ok=True)
        raise
    finally:
        if source is not None:
            del source
    if final_path is not None:
        weakref.finalize(matrix, final_path.unlink, missing_ok=True)
    return matrix


def _assemble_staged_sparse_entries(
    stage: _DiskOutputStage,
    entries: Iterable[_MatrixRowEntry],
    *,
    n_rows: int,
    total_nnz: int,
    use_memmap: bool,
) -> Any:
    paths: Tuple[Path, ...] = ()
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    if use_memmap:
        data, indices, indptr, paths = _new_sparse_memmap_components(
            dtype=stage.dtype,
            n_rows=n_rows,
            nnz=total_nnz,
        )
    else:
        data = np.empty(total_nnz, dtype=stage.dtype)
        indices = np.empty(total_nnz, dtype=np.int64)
        indptr = np.empty(n_rows + 1, dtype=np.int64)

    source_data = None
    source_indices = None
    try:
        if stage.total_nnz:
            source_data = np.memmap(
                stage.values_path,
                mode="r",
                dtype=stage.dtype,
                shape=(stage.total_nnz,),
            )
            if stage.indices_path is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Sparse staging is missing its index file.")
            source_indices = np.memmap(
                stage.indices_path,
                mode="r",
                dtype=np.int64,
                shape=(stage.total_nnz,),
            )
        offset = 0
        indptr[0] = 0
        for row_index, entry in enumerate(entries):
            next_offset = offset + entry.nnz
            if entry.nnz:
                if source_data is None or source_indices is None:  # pragma: no cover
                    raise RuntimeError("Sparse staging data is unavailable.")
                source_slice = slice(entry.nnz_offset, entry.nnz_offset + entry.nnz)
                data[offset:next_offset] = source_data[source_slice]
                indices[offset:next_offset] = source_indices[source_slice]
            indptr[row_index + 1] = next_offset
            offset = next_offset
        _flush_memmaps(data, indices, indptr)
        matrix = _csr_from_components(
            data,
            indices,
            indptr,
            shape=(n_rows, stage.width),
            dtype=stage.dtype,
        )
    except Exception:
        _unlink_paths(paths)
        raise
    finally:
        if source_data is not None:
            del source_data
        if source_indices is not None:
            del source_indices
    if paths:
        weakref.finalize(matrix, _unlink_paths, paths)
    return matrix


def resolve_memory_budget(config: MemoryConfig) -> MemoryBudget:
    """Resolve an effective memory budget using psutil.

    Args:
        config: Memory configuration.

    Returns:
        Resolved memory budget.
    """

    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    reserve = (
        int(config.reserve_system_bytes)
        if config.reserve_system_bytes is not None
        else _default_reserve_bytes(total)
    )
    if config.max_memory_bytes is not None:
        limit = int(config.max_memory_bytes)
    else:
        headroom = available - reserve
        if headroom > 0:
            limit = int(min(available * config.max_fraction, headroom))
        else:
            # Some constrained or containerized environments report very low
            # currently available memory relative to total system reserve.
            # Fall back to a fraction of available memory instead of collapsing
            # the budget to a single byte.
            limit = int(max(1, available * config.max_fraction))
    return MemoryBudget(
        total_bytes=total,
        available_bytes=available,
        reserve_system_bytes=reserve,
        max_memory_bytes=max(1, limit),
    )


def estimate_embedding_from_probe(
    probe_embeddings: Any,
    n_samples: int,
    batch_size: int,
    memory_config: MemoryConfig,
) -> EmbeddingMemoryEstimate:
    """Estimate full embedding memory from a probe batch.

    Args:
        probe_embeddings: Dense or sparse probe embedding batch.
        n_samples: Full dataset sample count.
        batch_size: Planned embedding batch size.
        memory_config: Memory configuration.

    Returns:
        Estimated embedding footprint and strategy.
    """

    if is_sparse_matrix(probe_embeddings):
        dim = int(probe_embeddings.shape[1])
        dtype = str(probe_embeddings.dtype)
        density = _safe_density(probe_embeddings)
        resident = estimate_sparse_bytes(
            n_samples=n_samples,
            n_features=dim,
            dtype=np.dtype(probe_embeddings.dtype),
            density=density,
        )
        dense_scoring = n_samples * dim * np.dtype(probe_embeddings.dtype).itemsize
        batch_bytes = estimate_sparse_bytes(
            n_samples=batch_size,
            n_features=dim,
            dtype=np.dtype(probe_embeddings.dtype),
            density=density,
        )
    else:
        arr = np.asarray(probe_embeddings)
        dim = int(arr.shape[1])
        dtype = str(arr.dtype)
        resident = n_samples * dim * np.dtype(arr.dtype).itemsize
        dense_scoring = resident
        batch_bytes = batch_size * dim * np.dtype(arr.dtype).itemsize
    budget = resolve_memory_budget(memory_config)
    strategy = (
        "stream_to_disk"
        if memory_config.allow_disk_spill and resident > budget.max_memory_bytes
        else "in_memory"
    )
    return EmbeddingMemoryEstimate(
        n_samples=n_samples,
        embedding_dim=dim,
        dtype=dtype,
        resident_bytes=int(resident),
        dense_scoring_bytes=int(dense_scoring),
        batch_embedding_bytes=int(batch_bytes),
        strategy=strategy,
    )


def estimate_matrix_resident_bytes(matrix: Any) -> int:
    """Estimate memory required to hold a dense or sparse matrix.

    Args:
        matrix: Dense or sparse matrix.

    Returns:
        Estimated resident bytes.
    """

    if is_sparse_matrix(matrix):
        return sparse_matrix_nbytes(matrix)
    return estimate_dense_nbytes(np.asarray(matrix))


def estimate_object_resident_bytes(value: Any, _seen: Optional[set[int]] = None) -> int:
    """Conservatively estimate the retained size of nested metadata."""

    seen = _seen if _seen is not None else set()
    identifier = id(value)
    if identifier in seen:
        return 0
    seen.add(identifier)
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            size += sum(estimate_object_resident_bytes(item, seen) for item in value.flat)
        else:
            size += int(value.nbytes)
        return size
    if isinstance(value, np.generic):
        return size
    if isinstance(value, Mapping):
        return size + sum(
            estimate_object_resident_bytes(key, seen) + estimate_object_resident_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(estimate_object_resident_bytes(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(
            estimate_object_resident_bytes(getattr(value, item.name), seen)
            for item in fields(value)
        )
    return size


def estimate_final_row_metadata_bytes(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_rows: int,
    expansion_factor: float,
    purpose: str,
) -> int:
    """Estimate final row-aligned metadata before allocating retained containers.

    Staged rows are intentionally read one at a time. ``expansion_factor`` accounts
    for the distinct labels, IDs, groups, annotations, and provenance containers
    built from each staged row by a materializer.
    """

    if expected_rows < 1:
        raise ValueError(f"{purpose} expected_rows must be >= 1.")
    if not np.isfinite(expansion_factor) or expansion_factor < 1.0:
        raise ValueError(f"{purpose} expansion_factor must be finite and >= 1.")
    observed = 0
    staged_bytes = 0
    for observed, row in enumerate(rows, start=1):
        if observed > expected_rows:
            raise RuntimeError(f"{purpose} staging produced extra final rows.")
        staged_bytes += estimate_object_resident_bytes(row)
    if observed != expected_rows:
        raise RuntimeError(f"{purpose} staging produced {observed} rows; expected {expected_rows}.")
    # Include final list/array pointer tables and fixed dataset/result metadata.
    container_bytes = expected_rows * 128 + 4096
    return int(staged_bytes * expansion_factor) + container_bytes


def admit_final_metadata(
    stager: IncrementalMatrixStager,
    required_bytes: int,
    *,
    purpose: str,
    retained_bytes: int = 0,
    transient_bytes: int = 0,
) -> int:
    """Admit retained row metadata and the current output's transient peak."""

    budget_bytes = stager.budget.max_memory_bytes
    cumulative_bytes = (
        int(stager.memory_config.model_memory_bytes)
        + int(stager.memory_config.raw_batch_memory_bytes)
        + retained_bytes
        + required_bytes
        + transient_bytes
    )
    if cumulative_bytes > budget_bytes:
        raise ValueError(
            f"{purpose} requires an estimated cumulative {cumulative_bytes} bytes including "
            "fixed model/raw-batch memory and final row metadata, "
            f"but the memory budget is {budget_bytes} bytes. Final labels, IDs, groups, "
            "annotations, and provenance remain resident even when disk spill is enabled; "
            "increase max_memory_bytes or reduce the retained row caps/subsample rate."
        )
    if not stager.memory_config.allow_disk_spill:
        stager.reserve_metadata(required_bytes, purpose=purpose)
    return retained_bytes + required_bytes


def estimate_metadata_resident_bytes(metadata: dict[str, Any]) -> Optional[int]:
    """Estimate resident bytes from embedding metadata.

    Args:
        metadata: Embedding metadata dictionary.

    Returns:
        Estimated bytes, or `None` if metadata is incomplete.
    """

    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if not shape or len(shape) != 2 or dtype is None:
        return None
    if metadata.get("sparse"):
        nnz = metadata.get("nnz")
        if nnz is None:
            return None
        return sparse_nbytes_from_nnz(
            nnz=int(nnz),
            n_rows=int(shape[0]),
            dtype=np.dtype(dtype),
        )
    return int(shape[0]) * int(shape[1]) * np.dtype(dtype).itemsize


def estimate_metadata_dense_scoring_bytes(metadata: dict[str, Any]) -> Optional[int]:
    """Estimate dense bytes needed for scoring from embedding metadata.

    Args:
        metadata: Embedding metadata dictionary.

    Returns:
        Estimated dense scoring bytes, or `None` if metadata is incomplete.
    """

    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if not shape or len(shape) != 2 or dtype is None:
        return None
    return int(shape[0]) * int(shape[1]) * np.dtype(dtype).itemsize


def assert_within_memory(
    required_bytes: int,
    memory_config: MemoryConfig,
    purpose: str,
) -> MemoryBudget:
    """Fail fast when a planned allocation exceeds the memory budget.

    Args:
        required_bytes: Estimated required bytes.
        memory_config: Memory configuration.
        purpose: Human-readable description of the planned work.

    Returns:
        Resolved memory budget.

    Raises:
        ValueError: If `required_bytes` exceeds the configured budget.
    """

    budget = resolve_memory_budget(memory_config)
    if memory_config.fail_fast and required_bytes > budget.max_memory_bytes:
        raise ValueError(
            f"{purpose} is estimated to require {required_bytes} bytes, exceeding "
            f"the memory budget of {budget.max_memory_bytes} bytes. Increase "
            "MemoryConfig.max_memory_bytes, enable disk spill where applicable, "
            "reduce batch size, or run fewer concurrent jobs."
        )
    return budget


def assemble_matrix_rows(
    rows: list[Any],
    memory_config: MemoryConfig,
    *,
    purpose: str,
) -> MatrixAssembly:
    """Stack compatible matrix rows without forcing an over-budget dense allocation.

    Dense and sparse results that exceed the resolved memory budget are assembled
    into temporary disk-backed arrays when spill is enabled. Staging files are
    removed automatically once the returned matrix is no longer referenced.
    """

    if not rows:
        raise ValueError(f"{purpose} requires at least one matrix row.")
    sparse_flags = [is_sparse_matrix(row) for row in rows]
    if any(flag != sparse_flags[0] for flag in sparse_flags[1:]):
        raise ValueError(f"{purpose} cannot mix dense and sparse matrix rows.")
    required = sum(estimate_matrix_resident_bytes(row) for row in rows)
    budget = resolve_memory_budget(memory_config)
    over_budget = required > budget.max_memory_bytes

    if sparse_flags[0]:
        sparse_rows = _validated_sparse_rows(rows, purpose=purpose)
        nnz = sum(int(row.nnz) for row in sparse_rows)
        total_rows = sum(int(row.shape[0]) for row in sparse_rows)
        index_bytes = np.dtype(np.int64).itemsize
        required = (
            nnz * (np.dtype(sparse_rows[0].dtype).itemsize + index_bytes)
            + (total_rows + 1) * index_bytes
        )
        over_budget = required > budget.max_memory_bytes
        if over_budget and not memory_config.allow_disk_spill:
            _raise_assembly_budget_error(
                purpose,
                required,
                budget.max_memory_bytes,
            )
        from scipy import sparse as scipy_sparse

        return MatrixAssembly(
            matrix=(
                _assemble_sparse_memmap_rows(sparse_rows)
                if over_budget
                else scipy_sparse.vstack(sparse_rows, format="csr")
            ),
            strategy="disk_spill" if over_budget else "in_memory",
            required_bytes=required,
            budget_bytes=budget.max_memory_bytes,
        )

    arrays = [np.asarray(row) for row in rows]
    first = arrays[0]
    if first.ndim != 2:
        raise ValueError(f"{purpose} rows must be two-dimensional matrices.")
    width = int(first.shape[1])
    dtype = first.dtype
    for row in arrays[1:]:
        if row.ndim != 2 or int(row.shape[1]) != width or row.dtype != dtype:
            raise ValueError(f"{purpose} rows must have one consistent matrix contract.")

    if over_budget and not memory_config.allow_disk_spill:
        _raise_assembly_budget_error(
            purpose,
            required,
            budget.max_memory_bytes,
        )
    matrix: Any
    if over_budget:
        matrix = _assemble_dense_memmap(arrays, width=width, dtype=dtype)
        strategy = "disk_spill"
    else:
        matrix = np.vstack(arrays)
        strategy = "in_memory"
    return MatrixAssembly(
        matrix=matrix,
        strategy=strategy,
        required_bytes=required,
        budget_bytes=budget.max_memory_bytes,
    )


def _assemble_dense_memmap(
    rows: list[np.ndarray],
    *,
    width: int,
    dtype: np.dtype[Any],
) -> np.memmap:
    descriptor, staging_path = tempfile.mkstemp(
        prefix="vertebrae-materialization-",
        suffix=".npy",
    )
    os.close(descriptor)
    path = Path(staging_path)
    try:
        total_rows = sum(int(row.shape[0]) for row in rows)
        matrix = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(total_rows, width),
        )
        offset = 0
        for row in rows:
            next_offset = offset + int(row.shape[0])
            matrix[offset:next_offset] = row
            offset = next_offset
        matrix.flush()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    weakref.finalize(matrix, path.unlink, missing_ok=True)
    return matrix


def _validated_sparse_rows(rows: List[Any], *, purpose: str) -> List[Any]:
    sparse_rows = []
    first = rows[0]
    if getattr(first, "ndim", None) != 2:
        raise ValueError(f"{purpose} rows must be two-dimensional matrices.")
    width = int(first.shape[1])
    dtype = np.dtype(first.dtype)
    for row in rows:
        if (
            getattr(row, "ndim", None) != 2
            or int(row.shape[1]) != width
            or np.dtype(row.dtype) != dtype
        ):
            raise ValueError(f"{purpose} rows must have one consistent matrix contract.")
        sparse_rows.append(row.tocsr(copy=False))
    return sparse_rows


def _assemble_sparse_memmap_rows(rows: List[Any]) -> Any:
    total_rows = sum(int(row.shape[0]) for row in rows)
    total_nnz = sum(int(row.nnz) for row in rows)
    width = int(rows[0].shape[1])
    dtype = np.dtype(rows[0].dtype)
    data, indices, indptr, paths = _new_sparse_memmap_components(
        dtype=dtype,
        n_rows=total_rows,
        nnz=total_nnz,
    )
    try:
        row_offset = 0
        nnz_offset = 0
        indptr[0] = 0
        for row in rows:
            csr = row.tocsr(copy=False)
            next_nnz = nnz_offset + int(csr.nnz)
            next_row = row_offset + int(csr.shape[0])
            data[nnz_offset:next_nnz] = csr.data
            indices[nnz_offset:next_nnz] = csr.indices
            indptr[row_offset + 1 : next_row + 1] = csr.indptr[1:] + nnz_offset
            nnz_offset = next_nnz
            row_offset = next_row
        _flush_memmaps(data, indices, indptr)
        matrix = _csr_from_components(
            data,
            indices,
            indptr,
            shape=(total_rows, width),
            dtype=dtype,
        )
    except Exception:
        _unlink_paths(paths)
        raise
    weakref.finalize(matrix, _unlink_paths, paths)
    return matrix


def _new_npy_memmap(
    *,
    dtype: np.dtype[Any],
    shape: Tuple[int, ...],
) -> Tuple[np.memmap, Path]:
    descriptor, staging_path = tempfile.mkstemp(
        prefix="vertebrae-materialization-",
        suffix=".npy",
    )
    os.close(descriptor)
    path = Path(staging_path)
    try:
        values = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return values, path


def _new_sparse_memmap_components(
    *,
    dtype: np.dtype[Any],
    n_rows: int,
    nnz: int,
) -> Tuple[np.memmap, np.memmap, np.memmap, Tuple[Path, ...]]:
    paths: List[Path] = []
    try:
        data, data_path = _new_npy_memmap(dtype=dtype, shape=(nnz,))
        paths.append(data_path)
        indices, indices_path = _new_npy_memmap(dtype=np.dtype(np.int64), shape=(nnz,))
        paths.append(indices_path)
        indptr, indptr_path = _new_npy_memmap(dtype=np.dtype(np.int64), shape=(n_rows + 1,))
        paths.append(indptr_path)
    except Exception:
        _unlink_paths(tuple(paths))
        raise
    return data, indices, indptr, tuple(paths)


def _csr_from_components(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    *,
    shape: Tuple[int, int],
    dtype: np.dtype[Any],
) -> Any:
    from scipy import sparse as scipy_sparse

    matrix = scipy_sparse.csr_matrix(shape, dtype=dtype)
    # Assigning the validated arrays directly avoids scipy narrowing int64 index
    # memmaps into newly allocated resident int32 arrays.
    matrix.data = data
    matrix.indices = indices
    matrix.indptr = indptr
    matrix._shape = shape
    return matrix


def _flush_memmaps(*values: np.ndarray) -> None:
    for value in values:
        flush = getattr(value, "flush", None)
        if callable(flush):
            flush()


def _unlink_paths(paths: Tuple[Path, ...]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _raise_assembly_budget_error(
    purpose: str,
    required_bytes: int,
    budget_bytes: int,
) -> None:
    raise ValueError(
        f"{purpose} is estimated to require {required_bytes} bytes, exceeding the memory "
        f"budget of {budget_bytes} bytes. Enable MemoryConfig.allow_disk_spill or increase "
        "max_memory_bytes."
    )


def largest_fitting_subsample_rate(
    required_bytes: int,
    memory_config: MemoryConfig,
) -> float:
    """Estimate the largest sample fraction that fits the memory budget.

    Args:
        required_bytes: Estimated bytes for the full sample set.
        memory_config: Memory configuration.

    Returns:
        Fraction in `(0, 1]` that should fit the configured budget.
    """

    if required_bytes < 1:
        return 1.0
    budget = resolve_memory_budget(memory_config)
    return min(1.0, max(0.0, budget.max_memory_bytes / float(required_bytes)))


def sparse_matrix_nbytes(matrix: Any) -> int:
    """Estimate resident bytes for an existing scipy sparse matrix."""

    total = int(matrix.data.nbytes)
    for attr in ("indices", "indptr", "row", "col"):
        values = getattr(matrix, attr, None)
        if values is not None:
            total += int(values.nbytes)
    return total


def sparse_nbytes_from_nnz(nnz: int, n_rows: int, dtype: np.dtype) -> int:
    """Estimate CSR sparse bytes from non-zero count and row count."""

    index_bytes = np.dtype(np.int32).itemsize
    return int(nnz) * (np.dtype(dtype).itemsize + index_bytes) + (int(n_rows) + 1) * index_bytes


def estimate_sparse_bytes(
    n_samples: int,
    n_features: int,
    dtype: np.dtype,
    density: float,
) -> int:
    """Estimate sparse matrix bytes from shape, dtype, and density."""

    nnz = int(round(n_samples * n_features * density))
    return sparse_nbytes_from_nnz(nnz=nnz, n_rows=n_samples, dtype=dtype)


def _default_reserve_bytes(total_bytes: int) -> int:
    gib = 1024**3
    return int(min(8 * gib, max(1 * gib, total_bytes * 0.2)))


def _safe_density(matrix: Any) -> float:
    total = int(matrix.shape[0]) * int(matrix.shape[1])
    if total == 0:
        return 0.0
    return min(1.0, max(0.0, float(matrix.nnz) / float(total)))
