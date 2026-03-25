import os

from typing import Dict, Any

from src.agents.reader.utils.reader_error import FileMissingError


class ParserEngine:
    """Normalizes input files into a universal intermediate text representation."""

    def parse(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileMissingError(file_path)

        _, ext = os.path.splitext(file_path.lower())
        encoding = "utf-8"

        if ext in {".txt", ".md", ".html", ".xml", ".json", ".csv"}:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()
        else:
            # Fallback for binary/unknown documents, preserving raw bytes safely.
            with open(file_path, "rb") as f:
                raw = f.read()
            content = raw.decode("utf-8", errors="replace")

        return {
            "source": file_path,
            "extension": ext,
            "content": content,
            "metadata": {"size": os.path.getsize(file_path)},
        }