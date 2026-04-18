import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_cerebras.cerebras import CerebrasModel


@pytest.fixture
def cerebras_model():
    return CerebrasModel("cerebras-llama3.1-8b")


def make_prompt(schema=None, prompt_text="Generate a person"):
    prompt = MagicMock()
    prompt.prompt = prompt_text
    prompt.schema = schema
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = None
    prompt.options.top_p = 1
    prompt.options.seed = None
    prompt.options.reasoning_effort = None
    prompt.options.disable_reasoning = None
    prompt.options.clear_thinking = None
    prompt.options.model_fields_set = set()
    return prompt


def test_schema_flag_enabled(cerebras_model):
    assert cerebras_model.supports_schema is True


def test_process_schema_dict(cerebras_model):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    assert cerebras_model._process_schema(schema) == schema


def test_process_schema_json_string(cerebras_model):
    processed = cerebras_model._process_schema(
        json.dumps(
            {
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            }
        )
    )
    assert processed["properties"]["age"]["type"] == "integer"


def test_process_schema_concise_compatibility(cerebras_model):
    processed = cerebras_model._process_schema("name, age int, bio")
    assert processed == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "bio": {"type": "string"},
        },
        "required": ["name", "age", "bio"],
    }


def test_normalize_schema_for_cerebras_adds_additional_properties_false(
    cerebras_model,
):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "pet": {
                "type": "object",
                "properties": {"species": {"type": "string"}},
                "required": ["species"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            },
        },
        "required": ["name", "pet", "items"],
    }

    normalized = cerebras_model._normalize_schema_for_cerebras(schema)

    assert normalized["additionalProperties"] is False
    assert normalized["properties"]["pet"]["additionalProperties"] is False
    assert (
        normalized["properties"]["items"]["items"]["additionalProperties"] is False
    )


def test_normalize_schema_requires_top_level_object(cerebras_model):
    with pytest.raises(Exception, match="top-level schema type to be object"):
        cerebras_model._normalize_schema_for_cerebras(
            {"type": "array", "items": {"type": "string"}}
        )


def test_build_response_format_uses_native_json_schema(cerebras_model):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    response_format = cerebras_model._build_response_format(schema)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"].startswith("cerebras_")
    assert response_format["json_schema"]["schema"] == schema


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_execute_with_schema_uses_native_json_schema(mock_get_key, mock_post, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"name": "Alice", "age": 30}'}}]
    }
    mock_post.return_value = mock_response

    prompt = make_prompt(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
    )

    with patch.object(
        cerebras_model,
        "_build_messages",
        return_value=[{"role": "user", "content": "Generate a person"}],
    ):
        result = list(cerebras_model.execute(prompt, False, MagicMock(), None))

    assert json.loads(result[0]) == {"name": "Alice", "age": 30}
    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is False
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_execute_with_schema_disables_streaming(mock_get_key, mock_post, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"name": "Bob"}'}}]
    }
    mock_post.return_value = mock_response

    prompt = make_prompt(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )

    list(cerebras_model.execute(prompt, True, MagicMock(), None))

    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is False
    assert payload["response_format"]["type"] == "json_schema"


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_execute_with_invalid_schema_json_response_raises(
    mock_get_key, mock_post, cerebras_model
):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "not-json"}}]
    }
    mock_post.return_value = mock_response

    prompt = make_prompt(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )

    with pytest.raises(Exception, match="invalid JSON"):
        list(cerebras_model.execute(prompt, False, MagicMock(), None))


@patch("llm_cerebras.cerebras.httpx.post")
@patch("llm_cerebras.cerebras.CerebrasModel.get_key")
def test_execute_with_schema_http_422_surfaces_error(
    mock_get_key, mock_post, cerebras_model
):
    mock_get_key.return_value = "fake-api-key"
    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(
        422,
        request=request,
        json={
            "message": "schema rejected",
            "type": "invalid_request_error",
            "code": "wrong_api_format",
        },
    )
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unprocessable Entity", request=request, response=response
    )
    mock_post.return_value = mock_response

    prompt = make_prompt(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )

    with pytest.raises(httpx.HTTPStatusError):
        list(cerebras_model.execute(prompt, False, MagicMock(), None))


def test_normalize_schema_for_cerebras_sets_additional_properties_on_defs(
    cerebras_model,
):
    schema = {
        "type": "object",
        "properties": {
            "person": {"$ref": "#/$defs/person"},
        },
        "required": ["person"],
        "$defs": {
            "person": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
    }

    normalized = cerebras_model._normalize_schema_for_cerebras(schema)

    assert normalized["additionalProperties"] is False
    assert normalized["$defs"]["person"]["additionalProperties"] is False
