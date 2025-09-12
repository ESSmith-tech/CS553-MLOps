from src.config_manager import ConfigManager
from src.model_manager import ModelManager
from src.chat_handler import ChatHandler
from src.ui_factory import UIFactory

def test_create_chatbot_interface():
    config = ConfigManager().load_config()
    prompts = ConfigManager().load_prompts()
    model_manager = ModelManager(config)
    handler = ChatHandler(model_manager, config, prompts)
    chatbot = UIFactory.create_chatbot_interface(handler, config)
    assert hasattr(chatbot, "render")