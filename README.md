# llm-cerebras

This is a plugin for [LLM](https://llm.datasette.io/) that adds support for the Cerebras inference API.

## Installation

Install this plugin in the same environment as LLM. The recommended workflow uses `uv`.

```bash
uv tool install --with llm-cerebras llm
```

## Configuration

You'll need to provide an API key for Cerebras.

```bash
llm keys set cerebras
```

## Listing available models

The plugin automatically fetches the latest available models from the Cerebras API and caches them for 24 hours.

```bash
llm models list | grep cerebras
# CerebrasModel: cerebras-llama3.1-8b
# CerebrasModel: cerebras-llama3.3-70b
# CerebrasModel: cerebras-llama-4-scout-17b-16e-instruct
# CerebrasModel: cerebras-qwen-3-32b
```

## Refreshing models

To get the latest models from the Cerebras API and update the cache:

```bash
llm cerebras refresh
```

This will fetch the current list of available models and save them to the cache. The models are automatically cached for 24 hours, so you typically don't need to refresh manually unless you want to check for newly released models.

## Schema Support

The llm-cerebras plugin supports schemas for structured output. You can use either compact schema syntax or full JSON Schema:

```bash
# Using compact schema syntax
llm -m cerebras-llama3.3-70b 'invent a dog' --schema 'name, age int, breed'

# Using multi-item schema for lists
llm -m cerebras-llama3.3-70b 'invent three dogs' --schema-multi 'name, age int, breed'

# Using full JSON Schema 
llm -m cerebras-llama3.3-70b 'invent a dog' --schema '{
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
llm -m cerebras-llama3.3-70b 'invent a famous scientist' --schema '
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
llm -m cerebras-llama3.3-70b --schema 'title, director, year int, genre' --save movie_template

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
