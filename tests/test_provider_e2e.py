import os

import pytest

from src.main import generate_answer


@pytest.mark.provider
def test_provider_returns_answer():
    if os.getenv("RUN_E2E") != "1":
        pytest.skip("Set RUN_E2E=1 to run the live provider smoke test.")

    required = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    missing = [name for name in required if not os.getenv(name)]
    assert not missing, f"Missing provider configuration: {', '.join(missing)}"

    answer = generate_answer(
        "Berapa lama link reset password berlaku?",
        ["Password reset link expires after 30 minutes."],
    )

    assert isinstance(answer, str)
    assert answer.strip()
