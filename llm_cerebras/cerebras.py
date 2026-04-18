import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import httpx
import llm
from llm.utils import make_schema_id, schema_dsl
from pydantic import Field


DEFAULT_HTTP_TIMEOUT = 120
GPT_OSS_MODEL_ID = "gpt-oss-120b"
GLM_47_MODEL_ID = "zai-glm-4.7"
DEPRECATED_MODEL_IDS = {
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b",
    "llama-4-maverick-17b-128e-instruct",
    "llama-4-scout-17b-16e-instruct",
    "qwen-3-32b",
}
FALLBACK_MODELS = {
    "cerebras-llama3.1-8b": "llama3.1-8b",
    "cerebras-gpt-oss-120b": GPT_OSS_MODEL_ID,
    "cerebras-qwen-3-235b-a22b-instruct-2507": "qwen-3-235b-a22b-instruct-2507",
    "cerebras-zai-glm-4.7": GLM_47_MODEL_ID,
}
GLM_DEFAULTS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_completion_tokens": 32768,
    "clear_thinking": False,
}


@llm.hookimpl
def register_models(register):
    model_map = CerebrasModel.get_models()
    for model_id in model_map.keys():
        register(CerebrasModel(model_id), aliases=tuple())


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def cerebras():
        "Commands relating to the llm-cerebras plugin"

    @cerebras.command()
    def refresh():
        "Refresh Cerebras models from API"
        try:
            models = CerebrasModel.refresh_models()
            print(f"Refreshed {len(models)} Cerebras models:")
            for model_id in sorted(models.keys()):
                print(f"  - {model_id}")
        except Exception as e:
            print(f"Error refreshing models: {e}")
            return 1
        return 0


class CerebrasModel(llm.Model):
    can_stream = True
    model_id: str
    api_base = "https://api.cerebras.ai/v1"
    supports_schema = True
    needs_key = "cerebras"
    key_env_var = "CEREBRAS_API_KEY"

    _cache_file = None
    _cache_duration = 24 * 60 * 60

    @classmethod
    def get_cache_file(cls):
        """Get the path to the models cache file."""
        if cls._cache_file is None:
            cls._cache_file = llm.user_dir() / "cerebras_models.json"
        return cls._cache_file

    @classmethod
    def load_cached_models(cls):
        """Load models from cache if available and not expired."""
        cache_file = cls.get_cache_file()

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            cache_time = cache_data.get("timestamp", 0)
            if time.time() - cache_time > cls._cache_duration:
                return None

            return cls._normalize_model_map(cache_data.get("models", {}))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logging.warning(f"Failed to load cached models: {e}")
            return None

    @classmethod
    def save_models_to_cache(cls, models):
        """Save models to cache with timestamp."""
        models = cls._normalize_model_map(models)
        cache_file = cls.get_cache_file()
        cache_data = {"timestamp": time.time(), "models": models}

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except OSError as e:
            logging.warning(f"Failed to save models to cache: {e}")

    @classmethod
    def _normalize_model_map(cls, models):
        """Filter out models that Cerebras now documents as deprecated."""
        normalized = {}
        for prefixed_id, api_id in (models or {}).items():
            resolved_id = api_id or prefixed_id.removeprefix("cerebras-")
            if resolved_id in DEPRECATED_MODEL_IDS:
                continue
            normalized[prefixed_id] = api_id
        return normalized

    @classmethod
    def fetch_models_from_api(cls):
        """Fetch available models from Cerebras API."""
        try:
            api_key = llm.get_key("", "cerebras", "CEREBRAS_API_KEY")
            if not api_key:
                logging.warning("No Cerebras API key found, using fallback models")
                raise ValueError("No API key available")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            url = f"{cls.api_base}/models"
            logging.info(f"Fetching models from {url}")
            response = httpx.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            api_data = response.json()
            models = {}

            if "data" in api_data:
                for model in api_data["data"]:
                    model_id = model.get("id", "")
                    if model_id:
                        prefixed_id = f"cerebras-{model_id}"
                        models[prefixed_id] = model_id

                logging.info(f"Successfully fetched {len(models)} models from API")
            else:
                logging.warning("No 'data' field in API response")

            return cls._normalize_model_map(models)

        except Exception as e:
            logging.error(f"Failed to fetch models from API: {e}")
            fallback_models = dict(FALLBACK_MODELS)
            logging.info(f"Using fallback models: {list(fallback_models.keys())}")
            return fallback_models

    @classmethod
    def get_models(cls, refresh=False):
        """Get models from cache or API."""
        if not refresh:
            cached_models = cls.load_cached_models()
            if cached_models:
                return cached_models

        models = cls.fetch_models_from_api()
        cls.save_models_to_cache(models)
        return models

    @classmethod
    def refresh_models(cls):
        """Force refresh models from API."""
        return cls.get_models(refresh=True)

    @property
    def model_map(self):
        """Get the current model mapping."""
        return self.get_models()

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description="What sampling temperature to use, between 0 and 1.5.",
            ge=0,
            le=1.5,
            default=0.7,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description="An alternative to sampling with temperature, called nucleus sampling.",
            ge=0,
            le=1,
            default=1,
        )
        seed: Optional[int] = Field(
            description="If specified, our system will make a best effort to sample deterministically.",
            default=None,
        )
        reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = Field(
            description="Reasoning effort level for models that support it.",
            default=None,
        )
        disable_reasoning: Optional[bool] = Field(
            description="Disable reasoning for models that support it. Deprecated on zai-glm-4.7.",
            default=None,
        )
        clear_thinking: Optional[bool] = Field(
            description=(
                "Preserve previous reasoning context on zai-glm-4.7 when set to false."
            ),
            default=None,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        messages = self._build_messages(prompt, conversation)
        try:
            api_key = self.get_key()
        except llm.NeedsKeyException as ex:
            raise llm.ModelError("No Cerebras API key configured") from ex
        if not api_key:
            raise llm.ModelError("No Cerebras API key configured")
        has_schema = bool(getattr(prompt, "schema", None))
        should_stream = stream and not has_schema
        api_model = self.model_map.get(self.model_id, self.model_id)
        option_fields_set = self._option_fields_set(prompt.options)

        self._validate_model_specific_options(prompt.options, api_model)

        data = {
            "model": api_model,
            "messages": messages,
            "stream": should_stream,
        }
        data.update(
            self._build_request_options(prompt.options, api_model, option_fields_set)
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if has_schema:
            schema = self._normalize_schema_for_cerebras(
                self._process_schema(prompt.schema)
            )
            data["response_format"] = self._build_response_format(schema)

        url = f"{self.api_base}/chat/completions"
        timeout = DEFAULT_HTTP_TIMEOUT

        if should_stream:
            with httpx.stream(
                "POST", url, json=data, headers=headers, timeout=timeout
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk != "[DONE]":
                            content = json.loads(chunk)["choices"][0]["delta"].get(
                                "content"
                            )
                            if content:
                                yield content
            return

        r = httpx.post(url, json=data, headers=headers, timeout=timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        if has_schema:
            try:
                content = json.dumps(json.loads(content))
            except json.JSONDecodeError as ex:
                raise llm.ModelError(
                    "Cerebras returned invalid JSON for a schema request"
                ) from ex
        yield content

    def _option_fields_set(self, options) -> set[str]:
        for attr in ("model_fields_set", "__fields_set__"):
            fields_set = getattr(options, attr, None)
            if isinstance(fields_set, set):
                return set(fields_set)
            if isinstance(fields_set, (list, tuple)):
                return set(fields_set)
        return set()

    def _validate_model_specific_options(self, options, api_model: str) -> None:
        reasoning_effort = options.reasoning_effort
        disable_reasoning = options.disable_reasoning
        clear_thinking = options.clear_thinking

        if reasoning_effort is not None:
            if api_model == GPT_OSS_MODEL_ID:
                if reasoning_effort not in {"low", "medium", "high"}:
                    raise llm.ModelError(
                        "reasoning_effort for gpt-oss-120b must be one of: low, medium, high"
                    )
            elif api_model == GLM_47_MODEL_ID:
                if reasoning_effort != "none":
                    raise llm.ModelError(
                        'reasoning_effort for zai-glm-4.7 must be "none"'
                    )
            else:
                raise llm.ModelError(
                    "reasoning_effort can only be used with the gpt-oss-120b or zai-glm-4.7 models"
                )

        if disable_reasoning is not None and api_model != GLM_47_MODEL_ID:
            raise llm.ModelError(
                "disable_reasoning can only be used with the zai-glm-4.7 model"
            )
        if clear_thinking is not None and api_model != GLM_47_MODEL_ID:
            raise llm.ModelError(
                "clear_thinking can only be used with the zai-glm-4.7 model"
            )
        if (
            api_model == GLM_47_MODEL_ID
            and reasoning_effort is not None
            and disable_reasoning is not None
            and disable_reasoning != (reasoning_effort == "none")
        ):
            raise llm.ModelError(
                "disable_reasoning conflicts with reasoning_effort for the zai-glm-4.7 model"
            )

    def _build_request_options(
        self, options, api_model: str, option_fields_set
    ) -> Dict[str, Any]:
        temperature = options.temperature
        top_p = options.top_p
        max_completion_tokens = options.max_tokens
        clear_thinking = options.clear_thinking

        if api_model == GLM_47_MODEL_ID:
            if "temperature" not in option_fields_set:
                temperature = GLM_DEFAULTS["temperature"]
            if "top_p" not in option_fields_set:
                top_p = GLM_DEFAULTS["top_p"]
            if "max_tokens" not in option_fields_set:
                max_completion_tokens = GLM_DEFAULTS["max_completion_tokens"]
            if "clear_thinking" not in option_fields_set:
                clear_thinking = GLM_DEFAULTS["clear_thinking"]

        data: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_completion_tokens is not None:
            data["max_completion_tokens"] = max_completion_tokens
        if options.seed is not None:
            data["seed"] = options.seed
        if options.reasoning_effort is not None:
            data["reasoning_effort"] = options.reasoning_effort
        if options.disable_reasoning is not None:
            data["disable_reasoning"] = options.disable_reasoning
        if clear_thinking is not None:
            data["clear_thinking"] = clear_thinking
        return data

    def _build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                messages.extend(
                    [
                        {"role": "user", "content": response.prompt.prompt},
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def _process_schema(self, schema) -> Dict[str, Any]:
        """
        Normalize direct schema input for non-CLI callers.

        With llm >= 0.30 the CLI resolves schema DSL, file paths, template
        references, and saved schema IDs before the plugin sees them. This
        method mainly exists as a compatibility shim for direct callers.
        """
        if isinstance(schema, dict):
            return schema

        if isinstance(schema, str):
            stripped = schema.strip()
            if not stripped:
                raise llm.ModelError("Schema cannot be empty")
            if stripped.startswith("{"):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError as ex:
                    raise llm.ModelError("Schema string contained invalid JSON") from ex
            return schema_dsl(schema)

        raise llm.ModelError(f"Unsupported schema type: {type(schema)!r}")

    def _normalize_schema_for_cerebras(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Cerebras strict-mode requirements to a JSON schema."""

        def normalize(node):
            if isinstance(node, list):
                return [normalize(item) for item in node]
            if not isinstance(node, dict):
                return node

            normalized = {key: normalize(value) for key, value in node.items()}
            if "properties" in normalized and "type" not in normalized:
                normalized["type"] = "object"
            if normalized.get("type") == "object":
                normalized["additionalProperties"] = False
            return normalized

        normalized = normalize(schema)
        if normalized.get("type") != "object":
            raise llm.ModelError(
                "Cerebras structured outputs require the top-level schema type to be object"
            )
        return normalized

    def _build_response_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        schema_id, _ = make_schema_id(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name(schema_id),
                "strict": True,
                "schema": schema,
            },
        }

    def _schema_name(self, schema_id: str) -> str:
        return f"cerebras_{schema_id[:24]}"
