from rag_evaluation.config import Settings


def test_default_settings():
    settings = Settings(_env_file=None)

    assert settings.llm_model == "qwen2.5:1.5b"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 3
