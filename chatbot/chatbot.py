import os
from pathlib import Path

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Chatbot:
    OPENAI_MODELS_NAMES = ["text-embedding-3-small", "gpt-3.5-turbo"]

    def __init__(self, chat_config: dict, database_config: dict) -> None:
        self.chat_config = chat_config
        self.database_config = database_config

        if self.database_config["text_embedding_model"] in self.OPENAI_MODELS_NAMES:
            embeddings = OpenAIEmbeddings(
                model=self.database_config["text_embedding_model"]
            )
            self.vector_db = Chroma(
                embedding_function=embeddings,
                persist_directory=str(Path(__file__).parent.parent / "vector_db"),
            )

        self.retriever = self.vector_db.as_retriever(
            search_type=self.database_config["search_type"],
            search_kwargs={"k": self.database_config["number_for_chank_to_retrive"]},
        )

        if self.chat_config["llm_model"] in self.OPENAI_MODELS_NAMES:
            self.llm = ChatOpenAI(model=self.chat_config["llm_model"])

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        self.chat_history = []

    def get_answer(self, user_input: str, x) -> str:
        result = self.rag_chain.invoke(
            {"input": user_input, "chat_history": self.chat_history}
        )
        self.chat_history.append(HumanMessage(content=f"User: {result}"))
        self.chat_history.append(SystemMessage(content=result["answer"]))
        return result["answer"]
