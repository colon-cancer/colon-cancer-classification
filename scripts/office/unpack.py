"""Unpack a .docx (or any Office Open XML) file into a directory."""
import sys
import zipfile
from pathlib import Path


def unpack(docx_path: str, out_dir: str) -> None:
    src = Path(docx_path)
    dst = Path(out_dir)
    if not src.exists():
        sys.exit(f"Error: file not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dst)
    print(f"Unpacked {src.name} -> {dst}")
    for p in sorted(dst.rglob("*")):
        print(" ", p.relative_to(dst))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: unpack.py <file.docx> <output_dir>")
    unpack(sys.argv[1], sys.argv[2])
