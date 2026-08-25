"""Transactional artifact persistence for SLAI tuning results.

Artifacts are staged in a private sibling directory and published by one
directory rename.  A pre-existing run directory is never overwritten.  The
writer separates compact run summary, trial evidence, and redacted config
snapshot so the same payload is not duplicated across files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tuning_types import *
from .tuning_validation import *
from .utils.tuning_errors import *
from .utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Tuning Artifact")
printer = PrettyPrinter()



@dataclass(frozen=True, slots=True)
class TuningArtifactConfig:
    output_dir: Path
    write_summary: bool = True
    write_trials: bool = True
    write_config_snapshot: bool = True
    indent: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        for name in ("write_summary", "write_trials", "write_config_snapshot"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not any(
            (self.write_summary, self.write_trials, self.write_config_snapshot)
        ):
            raise ValueError("At least one artifact payload must be enabled")
        if isinstance(self.indent, bool) or not isinstance(self.indent, int):
            raise TypeError("indent must be an integer")
        if self.indent < 0:
            raise ValueError("indent must be non-negative")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TuningArtifactConfig":
        report = validate_artifact_config(config)
        report.raise_if_invalid(
            message="Invalid tuning artifact configuration.",
            context=TuningErrorContext(
                component="TuningArtifactWriter", operation="load_config"
            ),
        )
        try:
            return cls(
                output_dir=Path(str(config["output_dir"])),
                write_summary=coerce_bool(
                    config.get("write_summary", True), name="write_summary"
                ),
                write_trials=coerce_bool(
                    config.get("write_trials", True), name="write_trials"
                ),
                write_config_snapshot=coerce_bool(
                    config.get("write_config_snapshot", True),
                    name="write_config_snapshot",
                ),
                indent=int(config.get("indent", 2)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningPersistenceError(
                "Unable to construct artifact configuration.",
                context=TuningErrorContext(
                    component="TuningArtifactWriter", operation="load_config"
                ),
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc


class TuningArtifactWriter:
    """Write one immutable, checksummed artifact bundle per tuning run."""

    MANIFEST_SCHEMA_VERSION = 1

    def __init__(self, config: TuningArtifactConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, TuningArtifactConfig)
            else TuningArtifactConfig.from_mapping(config)
        )

    def __call__(self, result: TuningResult) -> tuple[ArtifactRecord, ...]:
        if not isinstance(result, TuningResult):
            raise TuningPersistenceError("Artifact writer requires TuningResult.")
        base = self.config.output_dir.resolve()
        run_segment = self._run_segment(result.run_id)
        target = base / run_segment
        context = TuningErrorContext(
            run_id=result.run_id,
            component=self.__class__.__name__,
            operation="write_bundle",
            output_path=str(target),
        )
        if target.exists():
            raise TuningPersistenceError(
                "Artifact bundle already exists; existing run evidence is immutable.",
                context=context,
            )
        stage: Path | None = None
        try:
            base.mkdir(parents=True, exist_ok=True)
            stage = Path(
                tempfile.mkdtemp(prefix=f".{run_segment}.", suffix=".staging", dir=base)
            )
            written: list[tuple[str, str, Path]] = []
            if self.config.write_summary:
                path = atomic_write_json(
                    stage / "summary.json",
                    self._summary_payload(result),
                    indent=self.config.indent,
                    redact_sensitive=True,
                )
                written.append(("summary", "summary.json", path))
            if self.config.write_trials:
                path = atomic_write_json(
                    stage / "trials.json",
                    self._trials_payload(result),
                    indent=self.config.indent,
                    redact_sensitive=True,
                )
                written.append(("trials", "trials.json", path))
            if self.config.write_config_snapshot:
                path = atomic_write_json(
                    stage / "config.json",
                    {
                        "config_fingerprint": result.request.config_fingerprint,
                        "config": dict(result.request.config),
                    },
                    indent=self.config.indent,
                    redact_sensitive=True,
                )
                written.append(("config", "config.json", path))

            manifest_files = [
                {
                    "kind": kind,
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
                for kind, relative, path in written
            ]
            manifest = atomic_write_json(
                stage / "manifest.json",
                {
                    "schema_version": self.MANIFEST_SCHEMA_VERSION,
                    "run_id": result.run_id,
                    "status": result.status.value,
                    "created_at": utc_iso(result.completed_at),
                    "config_fingerprint": result.request.config_fingerprint,
                    "files": manifest_files,
                },
                indent=self.config.indent,
                redact_sensitive=True,
            )
            written.append(("manifest", "manifest.json", manifest))
            self._fsync_directory(stage)
            os.replace(stage, target)
            stage = None
            self._fsync_directory(base)

            records: list[ArtifactRecord] = []
            for kind, relative, _staged_path in written:
                final_path = target / relative
                records.append(
                    ArtifactRecord(
                        kind=kind,
                        status=ArtifactStatus.WRITTEN,
                        path=final_path,
                        checksum=self._sha256(final_path),
                        metadata={
                            "size_bytes": final_path.stat().st_size,
                            "relative_path": relative,
                        },
                    )
                )
            return tuple(records)
        except TuningError:
            raise
        except Exception as exc:
            raise wrap_exception(
                exc,
                message="Unable to persist tuning artifact bundle.",
                error_cls=TuningPersistenceError,
                context=context,
            ) from exc
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _run_segment(run_id: str) -> str:
        safe = sanitize_identifier(run_id, fallback="tuning-run", max_length=96)
        # The digest prevents two distinct unsafe identifiers from collapsing
        # to the same sanitized directory name.
        return f"{safe}-{stable_fingerprint(run_id)[:12]}"

    @staticmethod
    def _summary_payload(result: TuningResult) -> dict[str, Any]:
        search = result.search_result
        best = result.best_trial
        trial_counts = (
            Counter(trial.status.value for trial in search.trials)
            if search is not None
            else Counter()
        )
        return to_json_safe(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "strategy": result.strategy,
                "model_type": result.model_type,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "duration_seconds": result.duration_seconds,
                "config_fingerprint": result.request.config_fingerprint,
                "objective": (
                    None if search is None else search.objective.to_dict()
                ),
                "best_trial": (
                    None
                    if best is None
                    else {
                        "trial_id": best.trial_id,
                        "parameters": dict(best.parameters),
                        "objective_value": best.objective_value,
                        "metrics": dict(best.metrics),
                        "constraints": [
                            item.to_dict() for item in best.constraints
                        ],
                    }
                ),
                "trial_counts": dict(trial_counts),
                "promotion": (
                    None if result.promotion is None else result.promotion.to_dict()
                ),
                "warnings": list(result.warnings),
                "error": None if result.error is None else result.error.to_dict(),
            },
            redact_sensitive=True,
        )

    @staticmethod
    def _trials_payload(result: TuningResult) -> dict[str, Any]:
        search = result.search_result
        return {
            "run_id": result.run_id,
            "objective": None if search is None else search.objective.to_dict(),
            "trials": [] if search is None else [trial.to_dict() for trial in search.trials],
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["TuningArtifactConfig", "TuningArtifactWriter"]

if __name__ == "__main__":
    import json
    from collections.abc import Callable
    from tempfile import TemporaryDirectory
    print("\n=== Running Tuning Artifacts Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting tuning artifact tests", "info")

    _failures: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def _run_test(name: str, test: Callable[[], None]) -> None:
        try:
            test()
            printer.status("TEST", name, "success")
        except Exception as exc:
            _failures.append(f"{name}: {type(exc).__name__}: {exc}")
            printer.status("TEST", _failures[-1], "error")

    def _result() -> TuningResult:
        objective = MetricSpec("loss", ObjectiveDirection.MINIMIZE)
        started = utc_now()
        settings = TunerSettings(
            strategy=TuningStrategy.GRID,
            model_type="SelfTestModel",
            allow_generate=False,
        )
        request = TuningRunRequest(
            run_id="artifact-self-test",
            settings=settings,
            config={"token": "secret-value", "schema_version": 2},
            strategy_config={"fail_fast": False},
            search_space=(
                {"name": "x", "type": "integer", "values": [1]},
            ),
            config_fingerprint="self-test-fingerprint",
            objective=objective,
        )
        trial = TrialRecord(
            trial_id="trial-1",
            run_id=request.run_id,
            status=TrialStatus.SUCCEEDED,
            parameters={"x": 1},
            started_at=started,
            completed_at=utc_now(),
            metrics={"loss": 1.0},
            objective_value=1.0,
        )
        search = SearchResult(
            run_id=request.run_id,
            strategy=TuningStrategy.GRID,
            status=RunStatus.SUCCEEDED,
            objective=objective,
            trials=(trial,),
            started_at=started,
            completed_at=utc_now(),
            best_trial_id=trial.trial_id,
        )
        return TuningResult(
            request=request,
            status=RunStatus.SUCCEEDED,
            started_at=started,
            completed_at=utc_now(),
            search_result=search,
        )

    def _test_transactional_bundle() -> None:
        with TemporaryDirectory(prefix="slai-tuning-artifacts-") as directory:
            output = Path(directory) / "reports"
            writer = TuningArtifactWriter(
                TuningArtifactConfig(output_dir=output, indent=2)
            )
            result = _result()
            records = writer(result)
            _check(len(records) == 4, "expected three payloads and one manifest")
            _check(
                all(record.status is ArtifactStatus.WRITTEN for record in records),
                "artifact record did not report a successful write",
            )
            _check(
                all(record.path is not None and record.path.is_file() for record in records),
                "artifact path is missing",
            )
            _check(
                all(record.checksum == writer._sha256(record.path) for record in records if record.path is not None),
                "artifact checksum mismatch",
            )
            manifest_record = next(record for record in records if record.kind == "manifest")
            _check(manifest_record.path is not None, "manifest path is missing")
            assert manifest_record.path is not None
            manifest = json.loads(manifest_record.path.read_text(encoding="utf-8"))
            _check(manifest["run_id"] == result.run_id, "manifest run_id mismatch")
            _check(len(manifest["files"]) == 3, "manifest payload list is incomplete")
            config_record = next(record for record in records if record.kind == "config")
            _check(config_record.path is not None, "config artifact path is missing")
            assert config_record.path is not None
            _check(
                "secret-value" not in config_record.path.read_text(encoding="utf-8"),
                "sensitive configuration value was not redacted",
            )
            _check(
                not any(path.name.endswith(".staging") for path in output.iterdir()),
                "staging directory leaked after publication",
            )
            try:
                writer(result)
            except TuningPersistenceError:
                return
            raise AssertionError("existing immutable run bundle was overwritten")

    def _test_configuration_validation() -> None:
        try:
            TuningArtifactConfig(
                output_dir=Path("reports"),
                write_summary=False,
                write_trials=False,
                write_config_snapshot=False,
            )
        except ValueError:
            return
        raise AssertionError("empty artifact bundle configuration was accepted")

    _run_test("transactional immutable artifact bundle", _test_transactional_bundle)
    _run_test("artifact configuration invariants", _test_configuration_validation)

    _all_passed = not _failures
    printer.status(
        "",
        f"{2 - len(_failures)}/2 tuning artifact tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All tuning artifact tests passed ===\n")