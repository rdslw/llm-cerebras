"""
Automated user tests that simulate typical user workflows with llm-cerebras.
These tests require a properly installed llm environment with llm-cerebras.
"""

import os
import pytest
import json
import subprocess
import tempfile
import sqlite3
from pathlib import Path
from functools import lru_cache

# Skip tests if SKIP_USER_TESTS is set
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_USER_TESTS") == "1",
    reason="SKIP_USER_TESTS is set"
)

def run_command(cmd):
    """Run a shell command and return output"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode

@lru_cache(maxsize=1)
def get_available_cerebras_model():
    """Return an available Cerebras model ID, preferring smaller models."""
    stdout, stderr, returncode = run_command("llm models list")
    if returncode != 0:
        return None

    model_ids = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("CerebrasModel: "):
            model_ids.append(line.split(": ", 1)[1].strip())

    if not model_ids:
        return None

    preferred = [
        "cerebras-llama3.1-8b",
        "cerebras-qwen-3-32b",
        "cerebras-llama3.3-70b",
    ]
    for model_id in preferred:
        if model_id in model_ids:
            return model_id
    return model_ids[0]


@lru_cache(maxsize=1)
def has_working_cerebras_prompt():
    """Check that prompting actually works, not just model listing."""
    model_id = get_available_cerebras_model()
    if not model_id:
        return False

    stdout, stderr, returncode = run_command(
        f"llm -m {model_id} --no-stream -o max_tokens 12 'Say ok'"
    )
    return returncode == 0 and bool(stdout.strip())


def require_working_cerebras_model():
    """Return a working Cerebras model or skip if the environment is not ready."""
    model_id = get_available_cerebras_model()
    if not model_id:
        pytest.skip("cerebras models not available")
    if not has_working_cerebras_prompt():
        pytest.skip("cerebras prompting not available in this environment")
    return model_id

@pytest.mark.user
def test_plugin_installation():
    """Test that the plugin is properly installed and recognized by llm"""
    # Skip if pytest is run from development directory
    if os.path.exists("pyproject.toml") and "llm-cerebras" in open("pyproject.toml").read():
        pytest.skip("Running from development directory")
    
    # Check if plugin is listed
    stdout, stderr, returncode = run_command("llm plugins")
    assert returncode == 0, f"llm plugins command failed: {stderr}"
    assert "cerebras" in stdout, "cerebras plugin not found in llm plugins list"

@pytest.mark.user
def test_models_listing():
    """Test that cerebras models are listed by llm"""
    model_id = get_available_cerebras_model()
    if not model_id:
        pytest.skip("cerebras models not available")
    
    stdout, stderr, returncode = run_command("llm models list | grep -i cerebras")
    assert returncode == 0, "No cerebras models found"
    
    models = stdout.strip().split("\n")
    assert len(models) >= 1, "No cerebras models found"
    
    model_ids = [line.split(": ", 1)[1].strip() for line in models if ": " in line]
    assert model_id in model_ids, f"Expected model {model_id} not present in list"

@pytest.mark.user
def test_workflow_basic_prompt():
    """Test a basic user workflow with a simple prompt"""
    model_id = require_working_cerebras_model()
    
    stdout, stderr, returncode = run_command(
        f"llm -m {model_id} --no-stream 'Write a haiku about programming'"
    )
    assert returncode == 0, f"Command failed: {stderr}"
    assert len(stdout) > 10, "Response too short"
    
    # Haikus typically have three lines
    lines = [line for line in stdout.split("\n") if line.strip()]
    assert 2 <= len(lines) <= 5, f"Response doesn't look like a haiku: {stdout}"

@pytest.mark.user
def test_workflow_schema_prompt():
    """Test a user workflow with a schema prompt"""
    model_id = require_working_cerebras_model()
    
    stdout, stderr, returncode = run_command(
        f"llm -m {model_id} --no-stream --schema 'title, year int, director, genre' 'Suggest a sci-fi movie'"
    )
    assert returncode == 0, f"Command failed: {stderr}"
    
    # Try to parse as JSON
    try:
        data = json.loads(stdout)
        assert "title" in data, "Response missing title"
        assert "year" in data, "Response missing year"
        assert "director" in data, "Response missing director"
        assert "genre" in data, "Response missing genre"
        assert isinstance(data["year"], int), "Year is not an integer"
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {stdout}")

@pytest.mark.user
def test_workflow_conversation():
    """Test a conversational workflow with follow-up questions"""
    model_id = require_working_cerebras_model()
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.db', delete=False) as f:
        database_path = f.name
    
    try:
        cmd1 = (
            f"llm -m {model_id} --no-stream --log --database {database_path} "
            "'What are the three laws of robotics?'"
        )
        stdout1, stderr1, returncode1 = run_command(cmd1)
        assert returncode1 == 0, f"Command failed: {stderr1}"
        assert "law" in stdout1.lower() and "robot" in stdout1.lower(), "Response doesn't mention laws or robots"

        with sqlite3.connect(database_path) as conn:
            conversation_id = conn.execute(
                "select id from conversations order by id desc limit 1"
            ).fetchone()[0]

        cmd2 = (
            f"llm --no-stream --database {database_path} --cid {conversation_id} "
            "'Who created these laws?'"
        )
        stdout2, stderr2, returncode2 = run_command(cmd2)
        assert returncode2 == 0, f"Command failed: {stderr2}"
        assert "asimov" in stdout2.lower(), "Response doesn't mention Asimov"
    finally:
        if os.path.exists(database_path):
            os.unlink(database_path)

@pytest.mark.user
def test_workflow_schema_template():
    """Test creating and using a schema template"""
    model_id = require_working_cerebras_model()
    
    # Create a schema template
    template_name = "test_movie_schema"
    
    # Remove template if it exists
    run_command(f"llm templates rm {template_name} 2>/dev/null || true")
    
    try:
        # Create template
        cmd1 = (
            f"llm -m {model_id} --no-stream --schema "
            "'title: the movie title\nyear int: release year\ndirector: the director\ngenre: the primary genre' "
            f"--system 'You are a helpful assistant that recommends movies' --save {template_name}"
        )
        stdout1, stderr1, returncode1 = run_command(cmd1)
        assert returncode1 == 0, f"Template creation failed: {stderr1}"
        
        # Check template exists
        cmd2 = f"llm templates show {template_name}"
        stdout2, stderr2, returncode2 = run_command(cmd2)
        assert returncode2 == 0, f"Template check failed: {stderr2}"
        assert "title" in stdout2, "Template doesn't contain expected schema"
        
        # Use template
        cmd3 = f"llm -m {model_id} --no-stream -t {template_name} 'Suggest a comedy movie'"
        stdout3, stderr3, returncode3 = run_command(cmd3)
        assert returncode3 == 0, f"Template use failed: {stderr3}"
        
        # Try to parse as JSON
        try:
            data = json.loads(stdout3)
            assert "title" in data, "Response missing title"
            assert "year" in data, "Response missing year"
            assert "director" in data, "Response missing director"
            assert "genre" in data, "Response missing genre"
            assert isinstance(data["year"], int), "Year is not an integer"
            assert data["genre"].lower() == "comedy", f"Genre is not comedy: {data['genre']}"
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {stdout3}")
    finally:
        # Clean up template
        run_command(f"llm templates rm {template_name} 2>/dev/null || true")
