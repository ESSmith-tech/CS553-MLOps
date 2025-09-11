import os, json, sys, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_manager import ConfigManager

@pytest.fixture
def setup_cm_files(tmp_path):
    """Create fake cm files in a temporary directory"""
    cm_dir = tmp_path / "cm"
    cm_dir.mkdir()

    # Fake JSON config
    config_data = {"model": {"local_model_name": "fake-model"}}
    config_file = cm_dir / "app_config.json"
    config_file.write_text(json.dumps(config_data))

    # Fake JSON prompts
    prompts_data = {"greeting": "Hello!"}
    prompts_file = cm_dir / "app_prompts.json"
    prompts_file.write_text(json.dumps(prompts_data))

    # Fake CSS
    css_data = "body { background: black; }"
    css_file = cm_dir / "app_ui.css"
    css_file.write_text(css_data)

    # Point ConfigManager to use tmp_path/src as script_dir
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    script_dir = str(src_dir)

    return {
        "script_dir": script_dir,
        "config_data": config_data,
        "prompts_data": prompts_data,
        "css_data": css_data,
    }


def test_load_config(setup_cm_files):
    cm = ConfigManager(script_dir=setup_cm_files["script_dir"])
    config = cm.load_config()
    assert config == setup_cm_files["config_data"]


def test_load_prompts(setup_cm_files):
    cm = ConfigManager(script_dir=setup_cm_files["script_dir"])
    prompts = cm.load_prompts()
    assert prompts == setup_cm_files["prompts_data"]


def test_load_css(setup_cm_files):
    cm = ConfigManager(script_dir=setup_cm_files["script_dir"])
    css = cm.load_css()
    assert css == setup_cm_files["css_data"]
