import os

from typing import Dict, Any


class ConversionEngine:
    """Writes parsed intermediate content to a target format."""

    def convert(self, parsed_doc: Dict[str, Any], target_format: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(parsed_doc["source"]))[0]
        target_ext = target_format.lower().strip(".")
        out_path = os.path.join(output_dir, f"{base_name}.{target_ext}")

        content = parsed_doc.get("content", "")
        if target_ext in {"txt", "md", "html", "xml", "json"}:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            # Store as utf-8 bytes for unknown targets as a safe baseline.
            with open(out_path, "wb") as f:
                f.write(content.encode("utf-8", errors="replace"))

        return {
            "output_path": out_path,
            "target_format": target_ext,
            "source": parsed_doc["source"],
        }

    def merge(self, parsed_docs: list[Dict[str, Any]], output_format: str, output_dir: str, filename: str = "merged") -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        target_ext = output_format.lower().strip(".")
        out_path = os.path.join(output_dir, f"{filename}.{target_ext}")

        merged_content = []
        for doc in parsed_docs:
            merged_content.append(f"\n\n# Source: {doc['source']}\n")
            merged_content.append(doc.get("content", ""))

        final_text = "\n".join(merged_content)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return {"output_path": out_path, "output_format": target_ext, "inputs": [d["source"] for d in parsed_docs]}
