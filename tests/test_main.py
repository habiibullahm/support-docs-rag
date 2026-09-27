from unittest.mock import Mock, patch

import numpy as np

from src.main import chunk_document, cosine_similarity, generate_answer


def test_chunk_document_groups_two_sections():
    text = "Title one\nBody one\n\nTitle two\nBody two\n\nTitle three\nBody three"

    chunks = chunk_document(text)

    assert len(chunks) == 2
    assert "Title one" in chunks[0]
    assert "Title two" in chunks[0]
    assert "Title three" in chunks[1]


def test_cosine_similarity_for_identical_vectors():
    score = cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    assert score == 1.0


@patch("src.main.OpenAI")
def test_generate_answer_uses_persona_and_low_temperature(openai_class):
    response = Mock()
    response.choices = [Mock(message=Mock(content="Jawaban singkat."))]
    openai_class.return_value.chat.completions.create.return_value = response

    answer = generate_answer("Berapa lama?", ["Link berlaku 30 menit."])

    request = openai_class.return_value.chat.completions.create.call_args.kwargs
    assert answer == "Jawaban singkat."
    assert request["temperature"] == 0.2
    assert request["messages"][0]["role"] == "system"
    assert "bahasa Indonesia" in request["messages"][0]["content"]
    assert "Link berlaku 30 menit." in request["messages"][1]["content"]
