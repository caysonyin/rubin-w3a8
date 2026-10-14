"""Record independent Plan A engineering evidence without model/data resources."""
from __future__ import annotations

import hashlib
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from scripts.diagnose_layers import ROOT, dependency_check, source_fingerprints, write_json, relative_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/layerwise_diagnostics")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "implementation_tests.log"
    status, reason, versions = dependency_check()
    if status != "PASS":
        write_json(output / "implementation_acceptance.json",
                   dict(status="NOT_RUN", reason=reason, software_versions=versions))
        print(reason)
        return 2
    before = source_fingerprints()
    with tempfile.TemporaryDirectory(prefix="rubin-implementation-tests-") as temporary:
        junit = Path(temporary) / "tests.xml"
        with log.open("w") as handle:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-ra", "--junitxml", str(junit)],
                                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        suites = ET.parse(junit).getroot() if junit.is_file() else None
        cases = list(suites.iter("testcase")) if suites is not None else []
        plotted = [case for case in cases if case.get("name") == "test_canonical_coverage_and_png_pdf"]
        plot_passed = len(plotted) == 1 and not any(plotted[0].find(tag) is not None for tag in ("skipped", "failure", "error"))
        accepted = result.returncode == 0 and plot_passed and before == source_fingerprints()
        evidence = dict(status="PASS" if accepted else "FAIL", returncode=result.returncode,
                        test_cases=len(cases), skipped=sum(case.find("skipped") is not None for case in cases),
                        plotting_test_passed=plot_passed, software_versions=versions,
                        source_fingerprints=before, log=relative_path(log),
                        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        write_json(output / "implementation_acceptance.json", evidence)
    print(log.read_text())
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
