"""Small TURBOTEST-style executor used by each directory's local test script."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _arguments(defaults):
    parser = argparse.ArgumentParser(
        description="Run input.py, create output.test.txt, and compare selected values."
    )
    parser.add_argument("--with-pyscf", action="store_true")
    parser.add_argument("--ncores", type=int)
    parser.add_argument("--output", default=defaults.get("output", "output.test.txt"))
    parser.add_argument("--report", default=defaults.get("report", "test.report.txt"))
    parser.add_argument(
        "--reference", default=defaults.get("reference", "output.ref.txt")
    )
    parser.add_argument("--update-reference", action="store_true")
    return parser.parse_args()


def _environment(ncores):
    env = os.environ.copy()
    if ncores is not None:
        if ncores < 1:
            raise ValueError("ncores must be at least one")
        env["PYFOCK_TEST_NCORES"] = str(ncores)
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[variable] = str(ncores)
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + previous if previous else ""
    )
    return env


def _active_checks(checks, with_pyscf):
    return [
        check
        for check in checks
        if check.get("when", "always") == "always" or with_pyscf
    ]


def _extract(text, check):
    matches = list(re.finditer(check["match"], text, re.MULTILINE))
    if not matches:
        raise ValueError("pattern was not found")
    match = matches[-1] if check.get("occurrence", "last") == "last" else matches[0]
    if check.get("kind") == "text":
        return "found"
    raw_value = match.group(1)
    if check.get("kind") == "integer":
        return int(raw_value)
    return float(raw_value)


def _render_report(test, checks, reference_text, output_text, command):
    lines = [
        "=" * 108,
        "TEST: " + test["name"],
        "DESCRIPTION: " + test["description"],
        "COMMAND: " + " ".join(command),
        "REFERENCE: " + test.get("reference", "output.ref.txt"),
        "OUTPUT: " + test.get("output", "output.test.txt"),
        "=" * 108,
        "{:<32} {:>19} {:>19} {:>13} {:>11} {:>8}".format(
            "Quantity", "Reference", "Test", "Abs. diff", "Tolerance", "Result"
        ),
        "-" * 108,
    ]
    passed = True
    for check in checks:
        try:
            reference_value = _extract(reference_text, check)
            output_value = _extract(output_text, check)
            if check.get("kind") == "text":
                difference = "-"
                tolerance = "-"
                result = "PASS"
            elif check.get("kind") == "integer":
                difference_value = abs(output_value - reference_value)
                difference = str(difference_value)
                tolerance = "0"
                result = "PASS" if difference_value == 0 else "FAIL"
            else:
                tolerance_value = float(check["tol"])
                difference_value = abs(output_value - reference_value)
                difference = "{:.3e}".format(difference_value)
                tolerance = "{:.1e}".format(tolerance_value)
                result = "PASS" if difference_value <= tolerance_value else "FAIL"
            if result == "FAIL":
                passed = False
            reference_rendered = (
                str(reference_value)
                if isinstance(reference_value, str)
                else "{:.15g}".format(reference_value)
            )
            output_rendered = (
                str(output_value)
                if isinstance(output_value, str)
                else "{:.15g}".format(output_value)
            )
        except Exception as exc:
            passed = False
            reference_rendered = "-"
            output_rendered = "-"
            difference = "-"
            tolerance = str(check.get("tol", "-"))
            result = "MISSING"
            lines.append("  extraction error for {}: {}".format(check["name"], exc))
        lines.append(
            "{:<32} {:>19} {:>19} {:>13} {:>11} {:>8}".format(
                check["name"],
                reference_rendered,
                output_rendered,
                difference,
                tolerance,
                result,
            )
        )
    lines.extend(("-" * 108, "OVERALL RESULT: " + ("PASS" if passed else "FAIL")))
    return passed, "\n".join(lines) + "\n"


def run_test(TEST, CHECKS):
    args = _arguments(TEST)
    case_directory = Path(sys.argv[0]).resolve().parent

    def case_path(value):
        path = Path(value)
        return path if path.is_absolute() else case_directory / path

    output_path = case_path(args.output)
    reference_path = case_path(args.reference)
    report_path = case_path(args.report)
    input_name = TEST.get("input", "input.py")
    command = [sys.executable, input_name]
    if args.with_pyscf:
        command.append("--with-pyscf")

    with output_path.open("w", encoding="utf-8") as output_handle:
        completed = subprocess.run(
            command,
            cwd=str(case_directory),
            env=_environment(args.ncores),
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    active_checks = _active_checks(CHECKS, args.with_pyscf)
    output_text = output_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        report = "TEST: {}\nCOMMAND FAILED WITH STATUS {}\nSee {}\n".format(
            TEST["name"], completed.returncode, output_path
        )
        report_path.write_text(report, encoding="utf-8")
        print(report, end="")
        return 1

    if args.update_reference:
        if not args.with_pyscf:
            report = "Reference updates require --with-pyscf.\n"
            report_path.write_text(report, encoding="utf-8")
            print(report, end="")
            return 2
        missing = []
        for check in active_checks:
            try:
                _extract(output_text, check)
            except Exception as exc:
                missing.append("{}: {}".format(check["name"], exc))
        if missing:
            report = "Reference was not updated:\n" + "\n".join(missing) + "\n"
            report_path.write_text(report, encoding="utf-8")
            print(report, end="")
            return 1
        shutil.copyfile(str(output_path), str(reference_path))

    if not reference_path.is_file():
        report = "Missing reference file: {}\n".format(reference_path)
        report_path.write_text(report, encoding="utf-8")
        print(report, end="")
        return 1

    reference_text = reference_path.read_text(encoding="utf-8", errors="replace")
    passed, report = _render_report(
        TEST, active_checks, reference_text, output_text, command
    )
    if args.update_reference:
        report = "REFERENCE UPDATED\n" + report
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if passed else 1
