from __future__ import annotations

import base64
import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def path_to_data_url(path: str | Path) -> str:
    file_path = Path(path)
    mime_type = _mime_type_for_path(file_path)
    data = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def bytes_to_data_url(data: bytes, suffix: str = ".png") -> str:
    mime_type = _mime_type_for_path(Path(f"image{suffix}"))
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"

