from pathlib import Path

import gradio as gr
from chatbot.chatbot import Chatbot
from utils import read_yaml_file, save_openai_api_key
from langchain.schema import AIMessage, HumanMessage


chatbot_config = read_yaml_file(Path(__file__).parent / "config" / "chat_config.yaml")
database_config = read_yaml_file(
    Path(__file__).parent / "config" / "database_config.yaml"
)
CHATBOT = Chatbot(chatbot_config, database_config)

# gr.ChatInterface(CHATBOT.get_answer, type="messages").launch()

with gr.Blocks() as demo:

    # Define tabs
    with gr.Tabs():

        # First chatbot tab
        with gr.TabItem("Chat"):
            gr.ChatInterface(CHATBOT.get_answer, type="messages")

        # Second chatbot tab
        with gr.TabItem("Settings"):
            gr.Interface(
                fn=save_openai_api_key,
                inputs="textbox",
                description="Place openai api key here.",
                outputs="textbox",
            )
            # message_box_2 = gr.Textbox(
            #     fn=save_openai_api_key, placeholder="Place openai api key here."
            # )


demo.launch()

# # Simple chatbot logic
# def chatbot_gradio_interface(message, history):
#     result = CHATBOT.get_answer(message)
#     CHATBOT.chat_history.append(("human", message))
#     CHATBOT.chat_history.append(("ai", result))
#     return CHATBOT.chat_history, CHATBOT.chat_history


# # Gradio interface
# with gr.Blocks() as demo:
#     chatbot_interface = gr.Chatbot()
#     message_box = gr.Textbox(placeholder="Type your message here...")

#     def user_message(message, history):
#         return chatbot_gradio_interface(message, history)

#     # Event listeners
#     message_box.submit(
#         user_message,
#         inputs=[message_box, chatbot_interface],
#         outputs=[chatbot_interface, chatbot_interface],
#     )
#     # clear_btn.click(lambda: None, None, chatbot_interface)

# # Launch the Gradio app
# demo.launch()


# def predict(message, history):
#     # Convert Gradio's chat history to LangChain's format
#     history_langchain_format = []
#     for msg in history:
#         if msg["role"] == "user":
#             history_langchain_format.append(HumanMessage(content=msg["content"]))
#         elif msg["role"] == "assistant":
#             history_langchain_format.append(AIMessage(content=msg["content"]))

#     # Append the user's new message
#     history_langchain_format.append(HumanMessage(content=message))

#     # Generate the chatbot's response using the RAG chain
#     gpt_response = CHATBOT.get_answer(message)

#     # Update Gradio's chat history
#     history.append({"role": "user", "content": message})
#     history.append({"role": "assistant", "content": gpt_response})

#     return history, history
