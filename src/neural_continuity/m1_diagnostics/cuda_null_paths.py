"""Path-boundary checks shared by CUDA-null authority and model-free replay."""

from __future__ import annotations

import os
import stat
from pathlib import Path

_WINDOWS = os.name == "nt"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
lstat = os.lstat


def _windows_reparse(path: Path) -> bool:
    attributes = getattr(lstat(path), "st_file_attributes", None)
    if not isinstance(attributes, int):
        raise OSError(f"Windows file attributes unavailable: {path}")
    return bool(attributes & _REPARSE_POINT)


def path_is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    return _WINDOWS and _windows_reparse(path)


def has_linked_ancestor(path: Path) -> bool:
    return any(path_is_link_or_reparse(part) for part in (path, *path.parents))


def snapshot_file_inventory(root: Path) -> set[str]:
    """Enumerate files without traversing linked snapshot directories."""
    if has_linked_ancestor(root):
        raise ValueError("snapshot root contains a link or reparse point")
    files: set[str] = set()
    pending = [root]
    while pending:
        for entry in pending.pop().iterdir():
            if entry.is_dir():
                if path_is_link_or_reparse(entry):
                    raise ValueError(f"linked snapshot directory: {entry}")
                pending.append(entry)
            elif entry.is_file() or entry.is_symlink():
                files.add(entry.relative_to(root).as_posix())
            else:
                raise ValueError(f"unsupported snapshot entry: {entry}")
    return files
