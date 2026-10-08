"""Shared pytest configuration."""

# Windows caps environment variables at 32,767 characters, and pytest copies each test's id into
# PYTEST_CURRENT_TEST. A parametrize case built from large data gets an id that long and errors on
# Windows only. Fail on every platform instead, so it is caught before it reaches a Windows machine.
MAX_TEST_ID_CHARS = 500


def pytest_collection_modifyitems(config, items):
    too_long = [item.nodeid[:80] + "..." for item in items if len(item.nodeid) > MAX_TEST_ID_CHARS]
    if too_long:
        raise ValueError(f"Test ids longer than {MAX_TEST_ID_CHARS} chars (add ids=[...]): {too_long}")
