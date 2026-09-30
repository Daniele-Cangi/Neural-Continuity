"""Non-authoritative progress output for long model-free corpus verification."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor

_KIND = "m1_cuda_full_corpus_progress"
_EPOCH_PHASES = {"corpus_aggregation", "corpus_replay", "journal_epoch_replay"}
_progress_path: Path | None = None
_started = monotonic()


@dataclass
class _Phase:
    started: float
    completed: set[int] = field(default_factory=set)


_phases: dict[str, _Phase] = {}


def configure_progress(path: Path | None = None, *, reset: bool = True) -> None:
    """Attach an operational snapshot outside evidence; never overwrite unrelated files."""
    global _progress_path, _started
    _progress_path = None
    if reset:
        _phases.clear()
        _started = monotonic()
    if path is not None:
        try:
            path = path.absolute()
            if has_linked_ancestor(path):
                raise ValueError("progress path contains a link")
            if path.exists():
                prior = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(prior, dict) or prior.get("kind") != _KIND:
                    raise ValueError("existing progress path contains unrelated data")
            _progress_path = path
        except (OSError, ValueError) as exc:
            _warn(str(exc))


def _warn(reason: str) -> None:
    _emit({"kind": "m1_cuda_full_corpus_progress_warning", "reason": reason})


def _emit(record: dict[str, Any]) -> None:
    # Losing the terminal must not interrupt evidence verification.
    with suppress(OSError):
        print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def _snapshot(record: dict[str, Any]) -> None:
    if _progress_path is None:
        return
    pending: str | None = None
    try:
        if has_linked_ancestor(_progress_path):
            raise OSError("progress path contains a link")
        descriptor, pending = tempfile.mkstemp(
            prefix=f".{_progress_path.name}.", suffix=".tmp", dir=_progress_path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(pending, _progress_path)
    except OSError as exc:
        _warn(f"progress snapshot unavailable: {exc}")
    finally:
        if pending is not None and os.path.exists(pending):
            try:
                os.unlink(pending)
            except OSError as exc:
                _warn(f"progress temporary-file cleanup unavailable: {exc}")


def _duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    return f"{seconds_int // 3600:02d}:{seconds_int // 60 % 60:02d}:{seconds_int % 60:02d}"


def report_progress(phase: str, event: str, *, epoch: int | None = None) -> None:
    now = monotonic()
    if event == "started" or phase not in _phases:
        _phases[phase] = _Phase(now)
    state = _phases[phase]
    if event == "epoch_completed" and epoch is not None:
        state.completed.add(epoch)
    elapsed = max(0.0, now - state.started)
    record: dict[str, Any] = {
        "kind": _KIND,
        "timestamp": datetime.now(UTC).isoformat(),
        "process_id": os.getpid(),
        "phase": phase,
        "event": event,
        "run_elapsed_seconds": round(max(0.0, now - _started), 2),
        "phase_elapsed_seconds": round(elapsed, 2),
        "state": (
            "completed"
            if phase == "final_anchor" and event == "completed"
            else (
                event if event in {"blocked", "interrupted", "error", "checkpointed"} else "running"
            )
        ),
    }
    message = f"{phase}: {event} | elapsed {_duration(elapsed)}"
    if phase in _EPOCH_PHASES:
        count = len(state.completed)
        # ETA applies only to the fixed 120-epoch corpus phases, not to the
        # journal's variable number of previously completed epoch packages.
        eta = elapsed / count * (120 - count) if count and phase != "journal_epoch_replay" else None
        record.update(
            completed_epochs=count,
            epoch_total=120,
            percent=round(count / 120 * 100, 2),
            phase_eta_seconds=round(eta, 2) if eta is not None else None,
        )
        message = f"{phase}: {count}/120 ({count / 120 * 100:.1f}%) | elapsed {_duration(elapsed)}"
        if eta is not None:
            message += f" | estimated phase remaining {_duration(eta)}"
    if epoch is not None:
        record["epoch"] = epoch
        if event == "epoch_started":
            message += f" | processing epoch {epoch}"
    record["message"] = message
    _emit(record)
    _snapshot(record)
