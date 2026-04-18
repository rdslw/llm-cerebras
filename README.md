# llm-cerebras

This is a plugin for [LLM](https://llm.datasette.io/) that adds support for the Cerebras inference API.

This fork targets `llm>=0.30` and `Python>=3.10`.

## Installation

Install this plugin in the same environment as LLM. The recommended workflow uses `uv`.

```bash
uv tool install --with llm-cerebras llm
```

## Configuration

You'll need to provide an API key for Cerebras.

The April 2026 Cerebras docs still describe a single Inference API key and the
same `https://api.cerebras.ai/v1` base URL for coding integrations and paid
Code plans.

```bash
llm keys set cerebras
```

## Listing available models

The plugin automatically fetches the latest available models from the Cerebras API and caches them for 24 hours.

```bash
llm models list | grep cerebras
# CerebrasModel: cerebras-llama3.1-8b
# CerebrasModel: cerebras-gpt-oss-120b
# CerebrasModel: cerebras-qwen-3-235b-a22b-instruct-2507
# CerebrasModel: cerebras-zai-glm-4.7
```

## Refreshing models

To get the latest models from the Cerebras API and update the cache:

```bash
llm cerebras refresh
```

This will fetch the current list of available models and save them to the cache. The models are automatically cached for 24 hours, so you typically don't need to refresh manually unless you want to check for newly released models.

## Model-specific defaults

For `cerebras-zai-glm-4.7`, the plugin applies coding-oriented defaults unless you override them with `-o` options:

- `temperature=0.9`
- `top_p=0.95`
- `max_tokens=32768` (sent to the API as `max_completion_tokens`)
- `clear_thinking=false`

The plugin also supports `reasoning_effort=none` for `cerebras-zai-glm-4.7`. The older `disable_reasoning` parameter remains available for compatibility, but Cerebras documents it as deprecated as of March 24, 2026.

## Schema Support

The llm-cerebras plugin supports schemas for structured output using Cerebras native `json_schema` mode in strict mode.
For compatibility with Cerebras structured outputs, object schemas are normalized to set `additionalProperties: false`.

You can use either compact schema syntax or full JSON Schema:

```bash
# Using compact schema syntax
llm -m cerebras-llama3.1-8b 'invent a dog' --schema 'name, age int, breed'

# Using multi-item schema for lists
llm -m cerebras-llama3.1-8b 'invent three dogs' --schema-multi 'name, age int, breed'

# Using full JSON Schema 
llm -m cerebras-llama3.1-8b 'invent a dog' --schema '{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "breed": {"type": "string"}
  },
  "required": ["name", "age", "breed"]
}'
```

### Schema with Descriptions

You can add descriptions to your schema fields to guide the model:

```bash
llm -m cerebras-llama3.1-8b 'invent a famous scientist' --schema '
name: the full name including any titles
field: their primary field of study
year_born int: year of birth
year_died int: year of death, can be null if still alive
achievements: a list of their major achievements
'
```

### Creating Schema Templates

You can save schemas as templates for reuse:

```bash
# Create a template
llm -m cerebras-llama3.1-8b --schema 'title, director, year int, genre' --save movie_template

# Use the template
llm -t movie_template 'suggest a sci-fi movie from the 1980s'
```

## Development

This project now uses a `uv`-first workflow for local development. `uv sync` will create and manage the virtual environment for you.

```bash
cd llm-cerebras
uv sync --group dev
```

Run the core test suite:

```bash
uv run pytest tests/test_cerebras.py tests/test_schema_support.py
```

Run integration tests if you have a valid `CEREBRAS_API_KEY`:

```bash
uv run pytest tests/test_integration.py
```

Run automated user workflow tests:

```bash
uv run pytest tests/test_automated_user.py
```

Filter by pytest marker when needed:

```bash
uv run pytest -m "integration"  # Run only integration tests
uv run pytest -m "user"         # Run only user workflow tests
```

## License

Apache 2.0
