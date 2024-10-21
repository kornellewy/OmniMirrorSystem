from pathlib import Path

import gradio as gr

from chatbot.chatbot import Chatbot
from utils import read_yaml_file
from dotenv import load_dotenv

load_dotenv()

chatbot_config = read_yaml_file(Path(__file__).parent / "config" / "chat_config.yaml")
database_config = read_yaml_file(
    Path(__file__).parent / "config" / "database_config.yaml"
)
CHATBOT = Chatbot(chatbot_config, database_config)


# Simple chatbot logic
def chatbot_gradio_interface(message, history):
    result = CHATBOT.get_answer(message)
    CHATBOT.chat_history.append(("User", message))
    CHATBOT.chat_history.append(("Chatbot", result))
    return CHATBOT.chat_history, CHATBOT.chat_history


# Gradio interface
with gr.Blocks() as demo:
    chatbot_interface = gr.Chatbot()
    message_box = gr.Textbox(placeholder="Type your message here...")

    def user_message(message, history):
        return chatbot_gradio_interface(message, history)

    # Event listeners
    message_box.submit(
        user_message,
        inputs=[message_box, chatbot_interface],
        outputs=[chatbot_interface, chatbot_interface],
    )
    # clear_btn.click(lambda: None, None, chatbot_interface)

# Launch the Gradio app
demo.launch()
