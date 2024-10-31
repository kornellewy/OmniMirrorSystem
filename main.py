from pathlib import Path

import gradio as gr
from chatbot.chatbot import Chatbot
from utils import read_yaml_file, save_openai_api_key, save_antropic_api_key


chatbot_config = read_yaml_file(Path(__file__).parent / "config" / "chat_config.yaml")
database_config = read_yaml_file(
    Path(__file__).parent / "config" / "database_config.yaml"
)
CHATBOT = Chatbot(chatbot_config, database_config)


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Chat"):
            gr.ChatInterface(CHATBOT.get_answer, type="messages")
        with gr.TabItem("Settings"):
            gr.Interface(
                fn=save_openai_api_key,
                inputs="textbox",
                description="Place openai api key here.",
                outputs="label",
            )
            gr.Interface(
                fn=save_antropic_api_key,
                inputs="textbox",
                description="Place antropic api key here.",
                outputs="label",
            )
        with gr.TabItem("Updates"):
            pass
        with gr.TabItem("Tutorials"):
            pass
        with gr.TabItem("Contacts"):
            pass

demo.launch()
