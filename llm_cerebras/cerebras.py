import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
import llm
from llm.utils import make_schema_id, schema_dsl
from pydantic import Field


DEFAULT_HTTP_TIMEOUT = 120


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

            return cache_data.get("models", {})
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logging.warning(f"Failed to load cached models: {e}")
            return None

    @classmethod
    def save_models_to_cache(cls, models):
        """Save models to cache with timestamp."""
        cache_file = cls.get_cache_file()
        cache_data = {"timestamp": time.time(), "models": models}

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except OSError as e:
            logging.warning(f"Failed to save models to cache: {e}")

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

            return models

        except Exception as e:
            logging.error(f"Failed to fetch models from API: {e}")
            fallback_models = {
                "cerebras-llama3.1-8b": "llama3.1-8b",
                "cerebras-llama3.3-70b": "llama-3.3-70b",
                "cerebras-llama-4-scout-17b-16e-instruct": "llama-4-scout-17b-16e-instruct",
                "cerebras-deepseek-r1-distill-llama-70b": "DeepSeek-R1-Distill-Llama-70B",
            }
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
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
            description="Reasoning effort level for models that support it.",
            default=None,
        )
        disable_reasoning: Optional[bool] = Field(
            description="Disable reasoning for models that support it.",
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

        if (
            prompt.options.reasoning_effort is not None
            and api_model != "gpt-oss-120b"
        ):
            raise llm.ModelError(
                "reasoning_effort can only be used with the gpt-oss-120b model"
            )
        if (
            prompt.options.disable_reasoning is not None
            and api_model != "zai-glm-4.7"
        ):
            raise llm.ModelError(
                "disable_reasoning can only be used with the zai-glm-4.7 model"
            )

        data = {
            "model": api_model,
            "messages": messages,
            "stream": should_stream,
            "temperature": prompt.options.temperature,
            "max_tokens": prompt.options.max_tokens,
            "top_p": prompt.options.top_p,
            "seed": prompt.options.seed,
        }
        if prompt.options.reasoning_effort is not None:
            data["reasoning_effort"] = prompt.options.reasoning_effort
        if prompt.options.disable_reasoning is not None:
            data["disable_reasoning"] = prompt.options.disable_reasoning
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
