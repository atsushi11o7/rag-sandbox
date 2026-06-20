"""Corrective RAG のグラフ定義。

検索した文書の関連性を LLM で判定し、不十分なら クエリを書き換えて再検索する。

流れ:
    retrieve
      → grade_documents（文書ごとに LLM-as-Judge で関連性を判定）
          → relevant       → generate → END
          → not_relevant   → rewrite_query → retrieve（ループ）
    max_retries 回連続で not_relevant なら強制的に generate へ進む。
"""

import os
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.rag.generation import RAGGenerator

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは IT・プログラミング技術の文書を扱う評価者です。"
            "必ず 'relevant' か 'not_relevant' の1語だけで答えてください。",
        ),
        (
            "user",
            "クエリ: {query}\n\n文書:\n{context}\n\n"
            "この文書はクエリに答えるために関連していますか？",
        ),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは IT・プログラミングに関する検索エンジンのクエリを改善する専門家です。"
            "必ず日本語のみで答えてください。",
        ),
        (
            "user",
            "次の検索クエリをより良いクエリに書き換えてください。1行だけ出力してください。\n"
            "クエリ: {query}",
        ),
    ]
)


class RAGState(TypedDict):
    """Corrective RAG グラフ全体で共有するステート。

    original_query: ユーザーが最初に入力したクエリ。生成時に使用し、書き換えない。
    search_query:   実際の検索に使うクエリ。rewrite_query ノードで更新される。
    """

    original_query: str
    search_query: str
    documents: list[Document]
    answer: str
    grade: str  # "relevant" | "not_relevant"
    retries: int


def build_corrective_rag(
    retriever: BaseRetriever,
    generator: RAGGenerator,
    max_retries: int = 2,
):
    """Corrective RAG グラフを構築して返す。

    引数に既存の Retriever と RAGGenerator を渡すだけで動く。
    LLM（grade / rewrite 用）は generator と同じモデルを使う。

    Args:
        retriever: 文書検索に使う Retriever（Dense・Hybrid・RerankedRetriever など）。
        generator: 回答生成に使う RAGGenerator。
        max_retries: 再検索の最大回数。超えたら強制的に generate へ進む。

    Returns:
        invoke() で実行できる LangGraph コンパイル済みグラフ。
        入力: {"original_query": "...", "search_query": "...", "retries": 0}
        出力: RAGState（answer / documents などを含む）。

    Example:
        >>> app = build_corrective_rag(retriever, generator)
        >>> result = app.invoke({
        ...     "original_query": "uvの利点は？",
        ...     "search_query": "uvの利点は？",
        ...     "retries": 0,
        ... })
        >>> print(result["answer"])
    """
    host = generator.ollama_host or os.environ.get("OLLAMA_HOST")
    llm = (
        ChatOllama(model=generator.model, base_url=host)
        if host
        else ChatOllama(model=generator.model)
    )

    grade_chain = _GRADE_PROMPT | llm | StrOutputParser()
    rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()

    # --- ノード定義 ---

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["search_query"])
        return {"documents": docs}

    def grade_documents(state: RAGState) -> dict:
        # 文書ごとに判定し、関連ありの文書だけ残す
        relevant_docs = []
        for doc in state["documents"]:
            raw = (
                grade_chain.invoke({"query": state["search_query"], "context": doc.page_content})
                .strip()
                .lower()
            )
            if raw.startswith("relevant"):
                relevant_docs.append(doc)
        grade = "relevant" if relevant_docs else "not_relevant"
        return {"documents": relevant_docs, "grade": grade}

    def generate(state: RAGState) -> dict:
        # 元のクエリに対して回答する（書き換え後のクエリは使わない）
        answer = generator.generate(state["original_query"], state["documents"])
        return {"answer": answer}

    def rewrite_query(state: RAGState) -> dict:
        new_query = rewrite_chain.invoke({"query": state["search_query"]}).strip()
        return {"search_query": new_query, "retries": state.get("retries", 0) + 1}

    # --- 条件分岐 ---

    def should_continue(state: RAGState) -> Literal["generate", "rewrite_query"]:
        if state["grade"] == "relevant" or state.get("retries", 0) >= max_retries:
            return "generate"
        return "rewrite_query"

    # --- グラフ組み立て ---

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite_query", rewrite_query)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", should_continue)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
