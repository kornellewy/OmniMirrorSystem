from pathlib import Path
from typing import List, Literal
from pprint import pprint
import logging

from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langgraph.errors import GraphRecursionError
from langchain.load import dumps, loads

from utils import format_dict


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        subquestions: list of sub-questions
        generation: LLM generation
        documents: list of documents
    """

    question: str
    subquestions: List[str]
    generation: str
    documents: List[str]


class GraphChatbot:
    """
    Langchain chatbot using:
    - grapgh flow
    - ragfusion reranking
    - stepback
    - custom retriver context - todo
    """

    OPENAI_MODELS_NAMES = ["text-embedding-3-small", "gpt-3.5-turbo"]
    ANTROPIC_MODELS_NAMES = ["claude-3-haiku-20240307"]

    def __init__(self, chat_config: dict, database_config: dict) -> None:
        self.chat_config = chat_config
        self.database_config = database_config

        # Setup logging
        log_path = Path(__file__).parent.parent / "app.log"  # Save in main directory
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )
        logging.info("Initializing GraphChatbot...")

        if self.database_config["text_embedding_model"] in self.OPENAI_MODELS_NAMES:
            embeddings = OpenAIEmbeddings(
                model=self.database_config["text_embedding_model"]
            )
            self.vector_db = Chroma(
                embedding_function=embeddings,
                persist_directory=str(Path(__file__).parent.parent / "vector_db"),
            )
        if self.chat_config["llm_model"] in self.OPENAI_MODELS_NAMES:
            self.llm = ChatOpenAI(
                model=self.chat_config["llm_model"],
                temperature=float(self.chat_config["temperature"]),
            )
        if self.chat_config["llm_model"] in self.ANTROPIC_MODELS_NAMES:
            self.llm = ChatAnthropic(
                model=self.chat_config["llm_model"],
                temperature=float(self.chat_config["temperature"]),
            )
        if self.database_config["use_multi_query"]:
            self.retriever = MultiQueryRetriever.from_llm(
                retriever=self.vector_db.as_retriever(), llm=self.llm
            )
        else:
            self.retriever = self.vector_db.as_retriever(
                search_type=self.database_config["search_type"],
            )
        self.web_search_tool = TavilySearchResults(k=5)

        self.init_question_multiplayer()
        self.init_router_components()
        self.init_grader_components()
        self.init_generator_components()
        self.init_hallucination_grader()
        self.init_answer_grader()
        self.init_question_rewriter()
        self.create_workflow()

    def init_question_multiplayer(self) -> None:
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?\nwhat are the duties of The Police?\nwhat are the responsibilities of The Police?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?\nwho is Jan Sindel?\nJan Sindel’s country of origin?",
            },
        ]
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        multiplayer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge.
                    Your task is to step back and paraphrase a
                    question to a more generic step-back questions,
                    which is easier to answer. 
                    Provide these alternative questions separated by newlines.
                    Generate multiple search queries related to: {question} \n
                    Output (3 queries).
                    Here are a few examples:
                    """,
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )

        self.question_multiplayer_tool = (
            multiplayer_prompt
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

    def init_router_components(self) -> None:
        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to code, pytorch, machine learing.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        self.question_router = route_prompt | structured_llm_router

    def init_grader_components(self) -> None:
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )
        self.retrieval_grader = grade_prompt | structured_llm_grader

    def init_generator_components(self) -> None:
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    """You are an assistant for question-answering tasks.
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Return code if u can. Return alsow info about file location if u can.
                    Question: {question} 
                    Context: {context} 
                    Answer:""",
                ),
            ]
        )
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

    def init_hallucination_grader(self) -> None:
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
                ),
            ]
        )
        self.hallucination_grader = hallucination_prompt | structured_llm_grader

    def init_answer_grader(self) -> None:
        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)
        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "User question: \n\n {question} \n\n LLM generation: {generation}",
                ),
            ]
        )
        self.answer_grader = answer_prompt | structured_llm_grader

    def init_question_rewriter(self) -> None:
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()

    def get_answer(self, question: str) -> str:
        logging.info("Received question: %s", question)
        inputs = {"question": question}
        answer = "There is no information in the database or on the web to answer this question."
        try:
            for output in self.app_workflow.stream(inputs):
                for key, value in output.items():
                    logging.info(f"Processing {key} node: {format_dict(value)}")
                    # print(value)
                    pass
                pprint("\n---\n")
            # Final generation
            answer = value["generation"]
            logging.info(f"Final generation completed, {answer}")
        except GraphRecursionError as e:
            logging.error("GraphRecursionError encountered: %s", e)
        return answer

    def create_workflow(self):
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("web_search", self.web_search)  # web search
        workflow.add_node(
            "question_multiplayer", self.question_multiplayer
        )  # question multiplayer
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        # Build graph
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "question_multiplayer",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("question_multiplayer", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        self.app_workflow = workflow.compile()

    def retrieve(self, state: GraphState) -> dict:
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        all_documents = []
        if state["subquestions"]:
            for subquestion in state["subquestions"]:
                all_documents.extend(self.retriever.invoke(subquestion))
        else:
            all_documents = self.retriever.invoke(state["question"])
        # reranking
        fused_scores = {}
        for rank, doc in enumerate(all_documents):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + 60)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )[: self.database_config["number_for_chank_to_retrive"]]
        ]
        reranked_results = [doc for doc, _ in reranked_results]
        return {"documents": reranked_results, "question": state["question"]}

    def question_multiplayer(self, state: GraphState) -> dict:
        """
        Generates subquestions based on the given state's question.

        Args:
            state (GraphState): The current state of the graph.

        Returns:
            dict: A dictionary containing the generated subquestions.
        """
        subquestions = self.question_multiplayer_tool.invoke(
            {"question": state["question"]}
        )
        return {"subquestions": subquestions}

    def generate(self, state: GraphState) -> dict:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state: GraphState) -> dict:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state: GraphState) -> dict:
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state: GraphState) -> dict:
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        print(web_results)

        return {"documents": web_results, "question": question}

    def route_question(self, state: GraphState) -> str:
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
