#!/usr/bin/env python3
"""Run the local test script in selected PyFock case directories."""

import argparse
import sys

from tests.regression import (
    SUITES,
    discover_cases,
    format_case_summary,
    matches,
    run_case,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", default=["short"])
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="glob matched against suite/directory or descriptive test name",
    )
    parser.add_argument("--with-pyscf", action="store_true")
    parser.add_argument("--update-references", action="store_true")
    parser.add_argument("--ncores", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    suites = list(SUITES) if "all" in args.suites else args.suites
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        print("Unknown suites: {}".format(", ".join(unknown)), file=sys.stderr)
        return 2
    if args.update_references and not args.with_pyscf:
        print("--update-references requires --with-pyscf", file=sys.stderr)
        return 2

    cases = [case for case in discover_cases(suites) if matches(case, args.case)]
    if not cases:
        print("No matching tests.", file=sys.stderr)
        return 2
    if args.list:
        for case in cases:
            print(format_case_summary(case))
        return 0

    failed = 0
    for index, case in enumerate(cases, start=1):
        print("[{}/{}] {}".format(index, len(cases), format_case_summary(case)))
        try:
            report_path = run_case(
                case,
                with_pyscf=args.with_pyscf,
                ncores_override=args.ncores,
                update_reference=args.update_references,
            )
            status = "UPDATED" if args.update_references else "PASS"
            print("  {}  {}".format(status, report_path))
        except Exception as exc:
            failed += 1
            print("  FAIL")
            print("  " + str(exc).replace("\n", "\n  "))
            if args.fail_fast:
                break

    print("\n{} passed/updated; {} failed".format(len(cases) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
