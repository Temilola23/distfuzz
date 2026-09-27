import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DISTFUZZ_MULTIRANK") == "1":
        return
    skip = pytest.mark.skip(reason="starts several torch.distributed ranks; run inside Docker with `make docker-test`")
    for item in items:
        if "multirank" in item.keywords:
            item.add_marker(skip)
