import os, pytest, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_manager import ConfigManager
from src.model_manager import ModelManager
from src.chat_handler import ChatHandler

class DummyToken:
    def __init__(self, token): self.token = token

@pytest.fixture
def chat_handler():
    config = ConfigManager().load_config()
    model_manager = ModelManager(config)
    prompts = ConfigManager().load_prompts()
    return ChatHandler(model_manager, config, prompts)

def test_api_requires_token(chat_handler):
    # Should yield login_required message if token is missing
    gen = chat_handler.respond(
        message="Hi",
        history=[],
        gallery=None,
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        hf_token=None,
        use_local_model=False
    )
    first = next(gen)
    assert "log in" in first.lower() or "required" in first.lower()

def test_api_with_token(chat_handler):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN not set in environment")
    gen = chat_handler.respond(
        message="Hi",
        history=[],
        gallery=None,
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        hf_token=DummyToken(hf_token),
        use_local_model=False
    )
    first = next(gen)
    assert isinstance(first, str)