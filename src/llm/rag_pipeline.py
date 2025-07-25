from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import json
import re

from .classes import *
from .llm_base import *
from ..database import *


class RAGPipeline:
    def __init__(self, llm_client: LlmClient, db_manager: VectorDataBaseManager):
        self.llm = llm_client
        self.db = db_manager

        self.graph = StateGraph(GraphState)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("classify_query", self.classify_query)
        self.graph.add_node("retrieve_context", self.retrieve_context)
        self.graph.add_node("generate_answer", self.generate_answer)
        self.graph.add_node("validate_answer", self.validate_answer)

        self.graph.set_entry_point("classify_query")
        self.graph.add_edge("classify_query", "retrieve_context")
        self.graph.add_edge("retrieve_context", "generate_answer")
        self.graph.add_edge("generate_answer", "validate_answer")

        self.graph.add_conditional_edges(
            "validate_answer",
            self.decide_to_finish,
            {
                "continue": "generate_answer",
                "end": END
            }
        )

        self.compiled_graph = self.graph.compile()

    async def classify_query(self, state: GraphState):
        classifier = self.llm.with_structured_output(ClassificationResult)

        prompt = ChatPromptTemplate.from_template(
            """Классифицируй пользовательский запрос в соответствующую коллекцию базы данных.
            Доступные коллекции: {collections}

            Запрос: {query}

            Верни JSON с полями collection_name и confidence (0-1)."""
        )

        chain = prompt | classifier
        result = await chain.ainvoke({
            "query": state["query"],
            "collections": self.db.list_collections()
        })
        print(result)
        return {**state, "collection_name": result.collection_name}

    async def retrieve_context(self, state: GraphState):
        retriever = self.db.get_retriever(
            state["collection_name"],
            search_kwargs={"k": 5}
        )
        docs = await retriever.ainvoke(state["query"])
        context = [doc.page_content for doc in docs]
        print(docs)
        return {**state, "context": context, "validation_attempts": 0}

    async def generate_answer(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            """Ответь на вопрос, используя предоставленный контекст, если контекста недостаточно, то вежливо закончи диалог
            Контекст: {context}

            Вопрос: {query}"""
        )

        chain = (
                RunnablePassthrough.assign(context=lambda x: "\n\n".join(x["context"]))
                | prompt
                | self.llm.llm
                | StrOutputParser()
        )

        answer = await chain.ainvoke(state)
        return {**state, "answer": answer}

    @staticmethod
    async def validate_answer(state: GraphState):
        answer = state.get("answer", "")
        is_valid = len(answer.strip()) > 10 and not answer.lower().startswith(("извините", "к сожалению"))

        validation_result = {
            "valid": is_valid,
            "reason": "Ответ слишком короткий или неинформативный" if not is_valid else ""
        }

        return {**state, "validation": validation_result}

    @staticmethod
    def decide_to_finish(state: GraphState):
        current_attempts = state.get("validation_attempts", 0)
        next_attempts = current_attempts + 1

        state["validation_attempts"] = next_attempts

        print(f"Validation attempt {next_attempts} of 2")

        if next_attempts >= 2 or state["validation"]["valid"]:
            print("Finishing the pipeline")
            return "end"

        print("Continuing to generate a new answer")
        return "continue"

    async def run(self, query: str):
        initial_state = {"query": query}
        return await self.compiled_graph.ainvoke(initial_state)
