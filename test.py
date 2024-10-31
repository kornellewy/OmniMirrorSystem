from pathlib import Path

import gradio as gr
from chatbot.chatbot import Chatbot
from utils import read_yaml_file, save_openai_api_key, save_antropic_api_key
from langchain.schema import AIMessage, HumanMessage


chatbot_config = read_yaml_file(Path(__file__).parent / "config" / "chat_config.yaml")
database_config = read_yaml_file(
    Path(__file__).parent / "config" / "database_config.yaml"
)
CHATBOT = Chatbot(chatbot_config, database_config)


def main():
    """
    Pomysl jest taki kazda siceszka i zrudlo np jira to source i potem to zapisujemy do bazy danych.
    Narazie robimy 1 dir i 1 zrudlo.
    """

    while True:
        x = input("dawaj: ")
        answer = CHATBOT.get_answer(x, 1)
        print("ai: ", answer)


if __name__ == "__main__":
    main()
