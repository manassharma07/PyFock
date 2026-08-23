import subprocess
import sys
from pathlib import Path


def test_pyfock_import_does_not_require_legacy_re_t():
    """Python 3.13 removed the deprecated ``re.T`` alias."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import re; delattr(re, 'T') if hasattr(re, 'T') else None; import pyfock",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
