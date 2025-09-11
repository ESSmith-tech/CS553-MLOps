import gradio as gr
from typing import Dict, Any

from src.chat_handler import ChatHandler
from src.ui_image_scraper import UIImageScraper

class UIFactory:
    theme = gr.themes.Default()
    UIImageScraper().download_images_to_local()

    """Factory for creating UI components"""
    @staticmethod
    def create_chatbot_interface(chat_handler: ChatHandler, config: Dict[str, Any]) -> gr.ChatInterface:
        """Create the main chatbot interface"""
        return gr.ChatInterface(
            fn=chat_handler.respond,
            additional_inputs=[
                gr.Textbox(
                    value=config["defaults"]["system_message"],
                    label="System message"
                ),
                gr.Slider(
                    minimum=config["parameters"]["max_tokens"]["min"], 
                    maximum=config["parameters"]["max_tokens"]["max"], 
                    value=config["defaults"]["max_tokens"], 
                    step=config["parameters"]["max_tokens"]["step"], 
                    label="Max new tokens"
                ),
                gr.Slider(
                    minimum=config["parameters"]["temperature"]["min"], 
                    maximum=config["parameters"]["temperature"]["max"], 
                    value=config["defaults"]["temperature"], 
                    step=config["parameters"]["temperature"]["step"], 
                    label="Temperature"
                ),
                gr.Slider(
                    minimum=config["parameters"]["top_p"]["min"], 
                    maximum=config["parameters"]["top_p"]["max"], 
                    value=config["defaults"]["top_p"], 
                    step=config["parameters"]["top_p"]["step"], 
                    label="Top-p (nucleus sampling)"
                ),
                gr.Checkbox(
                    label="Use Local Model", 
                    value=config["defaults"]["use_local_model"]
                )
            ],
            type="messages",
        )

    @staticmethod
    def create_main_interface(chatbot: gr.ChatInterface, config: Dict[str, Any], 
                            css: str) -> gr.Blocks:
        """Create the main application interface"""

        with gr.Blocks(css=css,theme=UIFactory.theme) as demo:
            with gr.Row():
                gr.LoginButton()

            chatbot.render()

        return demo
