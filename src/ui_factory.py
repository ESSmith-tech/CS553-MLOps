import gradio as gr
from typing import Dict, Any
import os

from src.chat_handler import ChatHandler
from src.ui_image_scraper import UIImageScraper

class UIFactory:
    theme = gr.themes.Default()

    """Factory for creating UI components"""
    @staticmethod
    def create_chatbot_interface(chat_handler: ChatHandler, config: Dict[str, Any]) -> gr.Blocks:
        """Create the main chatbot interface using Blocks context"""
        image_paths = UIImageScraper().download_images_to_local()
        gallery_items = [(img_path, os.path.basename(img_path).rsplit('.', 1)[0]) for img_path in image_paths]

        with gr.Blocks(theme=UIFactory.theme) as demo:
            # State to hold the selected philosopher key
            selected_philosopher_key = gr.State(value=gallery_items[0][1] if gallery_items else None)

            def on_gallery_select(evt: gr.SelectData):
                print(f"Gallery selected: {evt.value['caption']}")
                return str(evt.value['caption'])

            # Gallery at the top
            gallery = gr.Gallery(
                value=gallery_items,
                label="Philosopher Images",
                object_fit="contain",
                elem_id="image-gallery",
                columns=3,
                height="auto",
                selected_index=0,
                show_label=True
            )
            gallery.select(on_gallery_select, outputs=selected_philosopher_key)

            # Create all the control components first (but don't render them yet)
            max_tokens_slider = gr.Slider(
                minimum=config["parameters"]["max_tokens"]["min"], 
                maximum=config["parameters"]["max_tokens"]["max"], 
                value=config["defaults"]["max_tokens"], 
                step=config["parameters"]["max_tokens"]["step"], 
                label="Max new tokens",
                render=False  # Don't render yet
            )
            temperature_slider = gr.Slider(
                minimum=config["parameters"]["temperature"]["min"], 
                maximum=config["parameters"]["temperature"]["max"], 
                value=config["defaults"]["temperature"], 
                step=config["parameters"]["temperature"]["step"], 
                label="Temperature",
                render=False  # Don't render yet
            )
            top_p_slider = gr.Slider(
                minimum=config["parameters"]["top_p"]["min"], 
                maximum=config["parameters"]["top_p"]["max"], 
                value=config["defaults"]["top_p"], 
                step=config["parameters"]["top_p"]["step"], 
                label="Top-p (nucleus sampling)",
                render=False  # Don't render yet
            )
            use_local_checkbox = gr.Checkbox(
                label="Use Local Model", 
                value=config["defaults"]["use_local_model"],
                render=False  # Don't render yet
            )

            # Create the chat interface with all additional inputs
            chat = gr.ChatInterface(
                fn=chat_handler.respond,
                additional_inputs=[
                    selected_philosopher_key,
                    max_tokens_slider,
                    temperature_slider,
                    top_p_slider,
                    use_local_checkbox
                ],
                type="messages",
            )

        return demo

    @staticmethod
    def create_main_interface(chatbot: gr.ChatInterface, config: Dict[str, Any], 
                            css: str) -> gr.Blocks:
        """Create the main application interface"""

        with gr.Blocks(css=css, theme=UIFactory.theme) as demo:
            with gr.Row():
                gr.LoginButton()

            chatbot.render()
        return demo
    