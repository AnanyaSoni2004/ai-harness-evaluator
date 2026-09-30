"""Keep tests hermetic: provider keys or a model choice in the developer's shell must not change results."""
import pytest

from harness.providers import KEY_ENV_VARS


@pytest.fixture(autouse=True)
def _no_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*KEY_ENV_VARS, "HARNESS_MODEL"):
        if name != "AI_API_KEY":
            monkeypatch.delenv(name, raising=False)
