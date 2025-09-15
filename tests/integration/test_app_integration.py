import os, time, pytest, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.app import ChatApp

@pytest.mark.skip("API cannot be tested outside of HuggingFace spaces due to OAuth restrictions")
def test_app_api_model_response():
    """Test ChatApp end-to-end with API model using HF_TOKEN from environment."""
    class DummyToken:
        def __init__(self, token): self.token = token
    
    hf_token = os.environ.get("HF_TOKEN")
    token_obj = DummyToken(hf_token)
    
    app = ChatApp()
    
    message = "Hello, who are you?"
    history = []
    gallery = "Diogenes"
    max_tokens = 128
    temperature = 0.2
    top_p = 0.9
    
    gen = app.chat_handler.respond(
        message=message,
        history=history,
        gallery=gallery,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        hf_token=token_obj,
        use_local_model=False
    )

    chunks = []
    for chunk in gen:
        assert isinstance(chunk, str)
        chunks.append(chunk)
    
    full_response = ''.join(chunks)
    assert len(full_response) > 0, f"Response was empty: {chunks}"



def test_app_local_model_response():
    """Test ChatApp end-to-end with local model, waiting for model readiness."""
    app = ChatApp()
    app.model_manager.start_model_loading()
    
    local_model = app.model_manager.local_model
    timeout = 120
    interval = 2
    waited = 0
    while not local_model.is_ready():
        if local_model.is_loading():
            time.sleep(interval)
            waited += interval
            if waited > timeout:
                pytest.fail("Local model did not become ready in time")
        else:
            pytest.fail("Local model is neither loading nor ready")

    # Build a simple chat message
    message = "Hello, local model!"
    history = []
    gallery = None
    max_tokens = 8
    temperature = 0.2
    top_p = 0.9
    # Use local model
    gen = app.chat_handler.respond(
        message=message,
        history=history,
        gallery=gallery,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        hf_token=None,
        use_local_model=True
    )
    # Get first response (should not be error message)
    first = next(gen)
    assert isinstance(first, str)
    # Should not be a model load failure message
    assert app.config["messages"]["model_load_failed"] not in first
    # Should not be a loading message
    assert app.config["messages"]["loading_message"] not in first
    # Should not be a model ready message (should be actual model output)
    assert app.config["messages"]["model_ready"] not in first
    # Should be a non-empty string
    assert len(first.strip()) > 0
