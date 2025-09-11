from typing import List, Dict, Generator, Optional, Any
import gradio as gr
from src.model_manager import ModelManager
import time

class ChatHandler:
    """Handles chat interactions and response generation"""
    
    def __init__(self, model_manager: ModelManager, config: Dict[str, Any]):
        self.model_manager = model_manager
        self.config = config
    
    def build_messages(self, message: str, history: List[Dict[str, str]], 
                      system_message: str) -> List[Dict[str, str]]:
        """Build message list from history and current message"""
        messages = [{"role": "system", "content": system_message}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        return messages
    
    def respond(self, message: str, history: List[Dict[str, str]], system_message: str,
               max_tokens: int, temperature: float, top_p: float, 
               hf_token: Optional[gr.OAuthToken], use_local_model: bool) -> Generator[str, None, None]:
        """Generate response to user message"""
        
        messages = self.build_messages(message, history, system_message)
        
        if use_local_model:
            yield from self._handle_local_model(messages, max_tokens, temperature, top_p)
        else:
            yield from self._handle_api_model(messages, max_tokens, temperature, top_p, hf_token)
    
    def _handle_local_model(self, messages: List[Dict[str, str]], max_tokens: int, 
                           temperature: float, top_p: float) -> Generator[str, None, None]:
        """Handle local model response generation"""
        print("[MODE] local")
        
        local_model = self.model_manager.local_model
        
        # Check if model is still loading
        if local_model.is_loading():
            # Queue the message for later processing
            queued_data = {
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'use_local_model': True
            }
            self.model_manager.queue_message(queued_data)
            
            yield self.config["messages"]["loading_message"]
            
            # Wait for model to be ready
            while local_model.is_loading():
                time.sleep(1)
            
            if not local_model.is_ready():
                yield self.config["messages"]["model_load_failed"]
                return
            
            # Clear the loading message and process normally
            yield self.config["messages"]["model_ready"]
        
        elif not local_model.is_ready():
            yield self.config["messages"]["model_load_failed"]
            return
        
        # Model is ready, generate response
        try:
            yield from local_model.generate(
                messages, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p
            )
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def _handle_api_model(self, messages: List[Dict[str, str]], max_tokens: int,
                         temperature: float, top_p: float, 
                         hf_token: Optional[gr.OAuthToken]) -> Generator[str, None, None]:
        """Handle API model response generation"""
        print("[MODE] api")
        
        if hf_token is None or not getattr(hf_token, "token", None):
            yield self.config["messages"]["login_required"]
            return
        
        try:
            yield from self.model_manager.api_model.generate(
                messages,
                hf_token=hf_token.token,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            yield f"Error generating response: {str(e)}"
