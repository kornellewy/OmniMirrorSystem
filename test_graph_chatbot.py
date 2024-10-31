from pathlib import Path

import gradio as gr
from chatbot.graph_chatbot import GraphChatbot
from utils import read_yaml_file


chatbot_config = read_yaml_file(Path(__file__).parent / "config" / "chat_config.yaml")
database_config = read_yaml_file(
    Path(__file__).parent / "config" / "database_config.yaml"
)
CHATBOT = GraphChatbot(chatbot_config, database_config)


def main():
    """
    Pomysl jest taki kazda siceszka i zrudlo np jira to source i potem to zapisujemy do bazy danych.
    Narazie robimy 1 dir i 1 zrudlo.
    """

    while True:
        x = input("dawaj: ")
        answer = CHATBOT.get_answer(x, 1)
        print("ai: ", answer)


def auto_test():
    questions = [
        "co mozesz mi powiedzieć o get_features_distance i jesli jest podac mi kod",
        # "co mozesz mi powiedzieć oremove_variation_and_move_data_to_final_dir i jesli jest podac mi kod",
        # "the most popular Austrian watercolorist ?",
    ]

    for question in questions:
        answer = CHATBOT.get_answer(question)
        print("answer: ", answer)


if __name__ == "__main__":
    auto_test()
