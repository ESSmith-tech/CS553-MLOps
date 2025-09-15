import queue, threading, time
from typing import Optional, Dict, Any, List, Generator
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Abstract interface for model implementations"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass

class LocalModel(ModelInterface):
    """Local model implementation using transformers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipe = None
        self._ready = False
        self._loading = False
    
    def load_model(self):
        """Load the local model"""
        if self._loading or self._ready:
            return
            
        self._loading = True
        try:
            from transformers import pipeline
            import torch
            
            print(f"[BACKGROUND] Loading local model: {self.model_name}")
            self.pipe = pipeline("text-generation", model=self.model_name)
            self._ready = True
            print("[BACKGROUND] Local model loaded successfully!")
        except Exception as e:
            print(f"[BACKGROUND] Error loading model: {e}")
            self._ready = False
        finally:
            self._loading = False
    
    def is_ready(self) -> bool:
        return self._ready
    
    def is_loading(self) -> bool:
        return self._loading
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512, 
                temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> Generator[str, None, None]:
        """Generate response from local model (single-turn, avoids self-conversation)"""
        if not self._ready:
            raise RuntimeError("Model not ready")

        # Use only system prompt and latest user message for local model
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
        user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), "")
        # Use the selected philosopher's name if present, else 'assistant'
        assistant_name = "assistant"
        for m in messages:
            if m['role'] not in ('system', 'user'):
                assistant_name = m['role']
                break
        prompt = f"system: {system_msg}\nuser: {user_msg}\n{assistant_name}:"

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False
        )

        response = outputs[0]["generated_text"]
        # Only return the first line (up to next newline or end)
        first_line = response.strip().split("\n")[0]
        yield first_line

class APIModel(ModelInterface):
    """API model implementation using HuggingFace Inference Client"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def is_ready(self) -> bool:
        return True  # API is always "ready" if we have a token
    
    def generate(self, messages: List[Dict[str, str]], hf_token: str, 
                max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, 
                **kwargs) -> Generator[str, None, None]:
        """Generate response from API model"""
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=hf_token, model=self.model_name)
        
        response = ""
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response

class ModelManager:
    """Manages model loading and message queuing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_model = LocalModel(config["model"]["local_model_name"])
        self.api_model = APIModel(config["model"]["api_model_name"])
        self.message_queue = queue.Queue()
        self.processing_queue = False
        self._model_thread: Optional[threading.Thread] = None
    
    def start_model_loading(self):
        """Start background model loading"""
        if self._model_thread is None or not self._model_thread.is_alive():
            self._model_thread = threading.Thread(
                target=self.local_model.load_model, 
                daemon=True
            )
            self._model_thread.start()
    
    def queue_message(self, message_data: Dict[str, Any]):
        """Queue a message for later processing"""
        self.message_queue.put(message_data)
    
    def has_queued_messages(self) -> bool:
        """Check if there are queued messages"""
        return not self.message_queue.empty()
    
    def process_queued_messages(self):
        """Process messages that were queued during model loading"""
        self.processing_queue = True
        
        processed_count = 0
        while not self.message_queue.empty():
            try:
                queued_data = self.message_queue.get_nowait()
                print(f"[QUEUE] Processing queued message...")
                processed_count += 1
            except queue.Empty:
                break
        
        self.processing_queue = False
        return processed_count
