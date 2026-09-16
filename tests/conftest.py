import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "lightning: tests that require PyTorch Lightning")


def pytest_collection_modifyitems(config, items):
    try:
        import lightning  # noqa: F401
    except ImportError:
        skip_lightning = pytest.mark.skip(reason="PyTorch Lightning not installed")
        for item in items:
            if "lightning" in item.keywords:
                item.add_marker(skip_lightning)
