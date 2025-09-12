import os, json, pytest

def test_cm_directory_exists():
    """Test that the cm directory exists."""
    cm_dir = os.path.join(os.path.dirname(__file__), '..', 'cm')
    
    assert os.path.exists(cm_dir) and os.path.isdir(cm_dir), \
        f"cm directory does not exist at: {cm_dir}"


def test_app_config_json_exists():
    """Test that app_config.json exists and is valid JSON."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'cm', 'app_config.json')
    
    assert os.path.exists(config_path) and os.path.isfile(config_path), \
        f"app_config.json does not exist at: {config_path}"
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    assert isinstance(config_data, dict), "app_config.json should contain a JSON object"


def test_app_ui_css_exists():
    """Test that app_ui.css exists and is readable."""
    css_path = os.path.join(os.path.dirname(__file__), '..', 'cm', 'app_ui.css')
    
    assert os.path.exists(css_path) and os.path.isfile(css_path), \
        f"app_ui.css does not exist at: {css_path}"
    
    with open(css_path, 'r') as f:
        css_content = f.read()
    assert len(css_content.strip()) > 0, "app_ui.css should not be empty"


def test_app_prompts_json_exists():
    """Test that app_prompts.json exists and is valid JSON."""
    prompts_path = os.path.join(os.path.dirname(__file__), '..', 'cm', 'app_prompts.json')
    
    assert os.path.exists(prompts_path) and os.path.isfile(prompts_path), \
        f"app_prompts.json does not exist at: {prompts_path}"
    
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
    assert isinstance(prompts_data, dict), "app_prompts.json should contain a JSON object"

def test_ui_scraper_config_json_exists():
    """Test that ui_scraper_config.json exists and is valid JSON."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'cm', 'ui_scraper_config.json')
    
    assert os.path.exists(config_path) and os.path.isfile(config_path), \
        f"ui_scraper_config.json does not exist at: {config_path}"
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    assert isinstance(config_data, dict), "ui_scraper_config.json should contain a JSON object"

def test_ui_scraper_config_is_valid():
    prompts_path = os.path.join(os.path.dirname(__file__), '..', 'cm', 'app_prompts.json')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'cm', 'ui_scraper_config.json')

    assert os.path.exists(prompts_path) and os.path.isfile(prompts_path), \
        f"app_prompts.json does not exist at: {prompts_path}"
    
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)

    assert os.path.exists(config_path) and os.path.isfile(config_path), \
        f"ui_scraper_config.json does not exist at: {config_path}"
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    for image in config_data['image_data']:
        filename = image['filename']
        key = os.path.basename(filename).rsplit('.', 1)[0]
        assert key in prompts_data.keys()