import pytest
from llm_cerebras.cerebras import CerebrasModel
from unittest.mock import patch, MagicMock
import llm

@pytest.fixture
def cerebras_model():
    return CerebrasModel("cerebras-llama3.1-8b")  # Use the full model ID with prefix

def test_cerebras_model_initialization(cerebras_model):
    assert cerebras_model.model_id == "cerebras-llama3.1-8b"
    assert cerebras_model.can_stream == True
    assert cerebras_model.api_base == "https://api.cerebras.ai/v1"

def test_build_messages(cerebras_model):
    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    conversation = None
    messages = cerebras_model._build_messages(prompt, conversation)
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "Test prompt"}


def make_prompt(**overrides):
    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.schema = None
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = None
    prompt.options.top_p = 1
    prompt.options.seed = None
    prompt.options.reasoning_effort = None
    prompt.options.disable_reasoning = None
    for key, value in overrides.items():
        setattr(prompt.options, key, value)
    return prompt


@patch('llm_cerebras.cerebras.httpx.post')
@patch('llm_cerebras.cerebras.CerebrasModel.get_key')
def test_execute_non_streaming(mock_get_key, mock_post, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value = mock_response

    prompt = make_prompt()

    response = MagicMock()
    conversation = None

    result = list(cerebras_model.execute(prompt, False, response, conversation))

    assert result == ["Test response"]
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["timeout"] == 120


@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_execute_without_api_key_raises_model_error(mock_get_key, cerebras_model):
    mock_get_key.side_effect = llm.NeedsKeyException("missing")

    with pytest.raises(Exception, match="No Cerebras API key configured"):
        list(cerebras_model.execute(make_prompt(), False, MagicMock(), None))


@pytest.fixture
def gpt_oss_model():
    return CerebrasModel("cerebras-gpt-oss-120b")


@pytest.fixture
def glm_model():
    return CerebrasModel("cerebras-zai-glm-4.7")


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_reasoning_effort_is_sent_for_gpt_oss(mock_get_key, mock_post, gpt_oss_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value = mock_response

    prompt = make_prompt(reasoning_effort="high")

    with patch.object(
        type(gpt_oss_model),
        "model_map",
        new=property(lambda self: {"cerebras-gpt-oss-120b": "gpt-oss-120b"}),
    ):
        result = list(gpt_oss_model.execute(prompt, False, MagicMock(), None))

    assert result == ["Test response"]
    payload = mock_post.call_args.kwargs["json"]
    assert payload["reasoning_effort"] == "high"
    assert "disable_reasoning" not in payload


@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_reasoning_effort_wrong_model_raises(mock_get_key, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    prompt = make_prompt(reasoning_effort="medium")

    with patch.object(
        type(cerebras_model),
        "model_map",
        new=property(lambda self: {"cerebras-llama3.1-8b": "llama3.1-8b"}),
    ):
        with pytest.raises(
            Exception,
            match="reasoning_effort can only be used with the gpt-oss-120b model",
        ):
            list(cerebras_model.execute(prompt, False, MagicMock(), None))


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_disable_reasoning_is_sent_for_glm(mock_get_key, mock_post, glm_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value = mock_response

    prompt = make_prompt(disable_reasoning=True)

    with patch.object(
        type(glm_model),
        "model_map",
        new=property(lambda self: {"cerebras-zai-glm-4.7": "zai-glm-4.7"}),
    ):
        result = list(glm_model.execute(prompt, False, MagicMock(), None))

    assert result == ["Test response"]
    payload = mock_post.call_args.kwargs["json"]
    assert payload["disable_reasoning"] is True
    assert "reasoning_effort" not in payload


@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_disable_reasoning_wrong_model_raises(mock_get_key, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    prompt = make_prompt(disable_reasoning=True)

    with patch.object(
        type(cerebras_model),
        "model_map",
        new=property(lambda self: {"cerebras-llama3.1-8b": "llama3.1-8b"}),
    ):
        with pytest.raises(
            Exception,
            match="disable_reasoning can only be used with the zai-glm-4.7 model",
        ):
            list(cerebras_model.execute(prompt, False, MagicMock(), None))


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_reasoning_options_omitted_when_none(mock_get_key, mock_post, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value = mock_response

    with patch.object(
        type(cerebras_model),
        "model_map",
        new=property(lambda self: {"cerebras-llama3.1-8b": "llama3.1-8b"}),
    ):
        list(cerebras_model.execute(make_prompt(), False, MagicMock(), None))

    payload = mock_post.call_args.kwargs["json"]
    assert "reasoning_effort" not in payload
    assert "disable_reasoning" not in payload


@patch("llm_cerebras.cerebras.httpx.stream")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_execute_streaming_uses_bounded_timeout(mock_get_key, mock_stream, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None
    mock_response.iter_lines.return_value = ["data: {\"choices\":[{\"delta\":{\"content\":\"Test\"}}]}", "data: [DONE]"]
    mock_response.raise_for_status.return_value = None
    mock_stream.return_value = mock_response

    result = list(cerebras_model.execute(make_prompt(), True, MagicMock(), None))

    assert result == ["Test"]
    assert mock_stream.call_args.kwargs["timeout"] == 120

if __name__ == "__main__":
    pytest.main()
