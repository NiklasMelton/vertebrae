"""Shared Markdown-safety helpers for generated reports."""

from typing import Any, Iterable


def markdown_text(value: Any) -> str:
    """Escape one dynamic value before inserting it into Markdown."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return (
        text.replace("\\", "\\\\")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def markdown_table_row(values: Iterable[Any]) -> str:
    """Render a table row while escaping every cell through one boundary."""

    return "| " + " | ".join(markdown_text(value) for value in values) + " |"
