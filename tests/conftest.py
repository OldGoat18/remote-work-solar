"""Test hygiene: a data-RUN test that REBUILDS a pipeline product (rather than reading its output) is both
slow and a CLOBBER risk — a smoke/partial rebuild can overwrite the full production artifact. The pipeline is
responsible for building products; tests verify them. So by default we SKIP any test whose body calls a build/
rebuild entry point. Run them explicitly with `WLPR_DATA_TESTS=1 pytest` (e.g. in a throwaway working tree).

Read-only data tests (which read a built scorecard/product and assert on it) are NOT affected — they carry no
build call and run normally once data is staged.
"""
import inspect
import os

import pytest

_REBUILD_CALLS = (
    "build_component_product", "build_owner_ledger", "build_obs_stock_kernel", "build_baseline_p0",
    "build_fields(", "run_decomposition(", "build_dashboard(", "emit_main(", "emit_adwfh", ".build()",
    "mode=\"smoke\"", "mode='smoke'",
)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("WLPR_DATA_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="heavy data-RUN test rebuilds a product (clobber risk); set WLPR_DATA_TESTS=1")
    for item in items:
        fn = getattr(item, "function", None)
        if fn is None:
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if any(tok in src for tok in _REBUILD_CALLS):
            item.add_marker(skip)
