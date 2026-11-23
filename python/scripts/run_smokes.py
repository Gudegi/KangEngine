"""Run explicit smoke suites against the current build without configuring it."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

from smoke_cases import CASES, MANUAL_CASES, SUITES, Case


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Result:
    case_id: str
    status: str
    seconds: float
    log: str


def select_cases(suites: list[str], case_ids: list[str]) -> list[Case]:
    by_id = {case.id: case for case in CASES}
    selected = [case_id for suite in suites for case_id in SUITES[suite]]
    selected.extend(case_ids)
    return [by_id[case_id] for case_id in dict.fromkeys(selected)]


def run_case(case: Case, *, root: Path, build_dir: Path, log_dir: Path) -> Result:
    name = case.id.replace(":", "__")
    log_path = log_dir / f"{name}.log"
    args = [arg.format(root=root, output=log_dir / name) for arg in case.args]
    if case.source.endswith(".py"):
        command = [sys.executable, str(root / case.source), *args]
    else:
        command = [str(build_dir / Path(case.source).stem), *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root / "python"), env.get("PYTHONPATH"))))
    env["PYTHONUNBUFFERED"] = "1"

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        try:
            completed = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=case.timeout, check=False,
            )
            status = "PASS" if completed.returncode == 0 else "FAIL"
            log.write(f"\nExit code: {completed.returncode}\n")
        except subprocess.TimeoutExpired:
            status = "FAIL"
            log.write(f"\nTimeout after {case.timeout:g} seconds\n")
        except OSError as error:
            # Missing/unexecutable native targets must not count as passing tests.
            status = "NOT_RUN"
            log.write(f"\nCould not start: {error}\n")

    output = log_path.read_text(encoding="utf-8", errors="replace")
    if status == "PASS" and re.search(r"^\s*SKIP\b", output, re.MULTILINE):
        status = "SKIP"
    return Result(case.id, status, time.monotonic() - started, str(log_path))


def run_cases(cases: list[Case], *, root: Path, build_dir: Path, log_dir: Path) -> int:
    results = []
    for case in cases:
        print(f"RUN  {case.id}", flush=True)
        result = run_case(case, root=root, build_dir=build_dir, log_dir=log_dir)
        results.append(result)
        print(f"{result.status:7} {result.case_id} ({result.seconds:.2f}s)", flush=True)
        if result.status != "PASS":
            print(f"  log: {result.log}", flush=True)

    counts = Counter(result.status for result in results)
    summary = {
        "counts": dict(counts),
        "results": [vars(result) for result in results],
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(" | ".join(f"{status}: {counts[status]}" for status in ("PASS", "FAIL", "SKIP", "NOT_RUN")))
    print(f"Logs: {log_dir}")
    return int(any(result.status != "PASS" for result in results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=sorted(SUITES), default=[])
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES], default=[])
    parser.add_argument("--list", action="store_true", help="list cases and requirements without initializing the engine")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/release")
    args = parser.parse_args()

    suites = args.suite
    if not suites and not args.case:
        suites = ["all"]
    cases = select_cases(suites, args.case)
    if args.list:
        for case in cases:
            requirements = ",".join(case.requirements) or "none"
            print(f"{case.id}  [{','.join(case.suites)}]  requires={requirements}")
            print(f"  {case.source} {shlex.join(case.args)}".rstrip())
        if not args.suite and not args.case:
            for source, reason in MANUAL_CASES.items():
                print(f"MANUAL {source}: {reason}")
        return 0

    log_root = ROOT / "build/smoke-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=log_root))
    return run_cases(cases, root=ROOT, build_dir=args.build_dir.resolve(), log_dir=log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
