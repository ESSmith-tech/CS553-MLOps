import pytest
from src.config_manager import ConfigManager
from src.model_manager import ModelManager
from src.chat_handler import ChatHandler

class DummyToken:
    def __init__(self, token): self.token = token

@pytest.fixture
def config():
    return ConfigManager().load_config()

@pytest.fixture
def model_manager(config):
    return ModelManager(config)

@pytest.fixture
def chat_handler(config):
    model_manager = ModelManager(config)
    prompts = ConfigManager().load_prompts()
    return ChatHandler(model_manager, config, prompts)

def test_build_messages(chat_handler):
    msg = "hello"
    history = [{"role": "user", "content": "hi"}]
    system_prompt = "sys"
    messages = chat_handler.build_messages(msg, history, system_prompt)
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "hello"

def test_respond_login_required(chat_handler, config):
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
    assert config["messages"]["login_required"].split()[0] in first

def test_respond_local_model_not_ready(chat_handler, config):
    # Force local model to not be ready
    chat_handler.model_manager.local_model._ready = False
    chat_handler.model_manager.local_model._loading = False
    gen = chat_handler.respond(
        message="Hi",
        history=[],
        gallery=None,
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        hf_token=None,
        use_local_model=True
    )
    first = next(gen)
    assert config["messages"]["model_load_failed"] in first

def test_queue_and_process_messages(model_manager):
    msg = {"foo": "bar"}
    model_manager.queue_message(msg)
    assert model_manager.has_queued_messages()
    processed = model_manager.process_queued_messages()
    assert processed == 1

def test_local_model_ready_flag(model_manager):
    local_model = model_manager.local_model
    local_model._ready = True
    assert local_model.is_ready()
    local_model._ready = False
    assert not local_model.is_ready()

def test_local_model_loading_flag(model_manager):
    local_model = model_manager.local_model
    local_model._loading = True
    assert local_model.is_loading()
    local_model._loading = False
    assert not local_model.is_loading()
