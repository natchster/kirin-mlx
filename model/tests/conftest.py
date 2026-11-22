import os

import pytest

# KIRIN_REQUIRE_FIXTURES=1 turns every skip in model/tests (missing checkpoints, golden fixtures,
# the Metal GPU, torch) into a failure, so a full run cannot pass by skipping.
REQUIRE_FIXTURES = os.environ.get("KIRIN_REQUIRE_FIXTURES", "").strip().lower() not in ("", "0", "false", "no")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if REQUIRE_FIXTURES and report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"skipped, but KIRIN_REQUIRE_FIXTURES is set: {reason}"
