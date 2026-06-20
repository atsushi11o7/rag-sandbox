"""マルチクエリ Retriever（パラフレーズ・クエリ分解）。

素のクエリを LLM で複数のクエリに変換し、それぞれで検索した結果を RRF で統合する。

- ParaphraseRetriever : 同じ意図を持つ言い換えを生成して再現率を上げる
- DecomposeRetriever  : 複合的な質問をサブクエリに分解して各側面をカバーする

両クラスとも流れは同じ:
    1. LLM でクエリを N 個に変換
    2. 各クエリで base_retriever を検索
    3. 結果を RRF で統合して返す
"""

import os
import re

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import PrivateAttr, model_validator

_PARAPHRASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "必ず日本語のみで答えてください。"),
        (
            "user",
            "次の質問を {n} 通りに言い換えてください。1行に1つ、番号なしで出力してください。\n質問: {query}",
        ),
    ]
)

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "必ず日本語のみで答えてください。"),
        (
            "user",
            "次の質問を {n} 個のシンプルなサブクエリに分解してください。"
            "1行に1つ、番号なしで出力してください。\n質問: {query}",
        ),
    ]
)

# 行頭の番号（"1. " "1、"）や箇条書き記号（"- " "• " "・ "）を除去するパターン
_BULLET_RE = re.compile(r"^[\d]+[.、．]\s*|^[-•・]\s*")


def _parse_queries(text: str) -> list[str]:
    """LLM の出力テキストを1行1クエリのリストに変換する。"""
    queries = []
    for line in text.splitlines():
        cleaned = _BULLET_RE.sub("", line).strip()
        if cleaned:
            queries.append(cleaned)
    return queries


def _rrf_merge(results_list: list[list[Document]], k: int = 60) -> list[Document]:
    """複数の検索結果リストを RRF で統合する。

    Args:
        results_list: 各クエリの検索結果リストのリスト。
        k: RRF の平滑化定数（論文デフォルト値 60）。

    Returns:
        RRF スコア降順でソートされた Document リスト。
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id")
            if not key:
                raise ValueError(
                    f"Document has neither chunk_id nor doc_id in metadata: {doc.metadata}"
                )
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


class ParaphraseRetriever(BaseRetriever):
    """クエリを言い換えて複数検索し RRF で統合する Retriever。

    オリジナルのクエリ + LLM が生成した言い換えクエリで検索し、
    RRF で結果を統合することで再現率を向上させる。

    Attributes:
        base_retriever: 各クエリの検索に使う Retriever。
        model: Ollama のモデル名。
        ollama_host: Ollama サーバーの URL。None のとき環境変数 OLLAMA_HOST を参照。
        n_queries: 生成する言い換えの数（オリジナルを含めると n_queries+1 件で検索）。
        k: RRF の平滑化定数。
        top_k: 返す上位件数。

    Example:
        >>> dense = build_dense_retriever(store, top_k=10)
        >>> retriever = ParaphraseRetriever(base_retriever=dense, n_queries=3)
        >>> results = retriever.invoke("uvの利点は？")
    """

    base_retriever: BaseRetriever
    model: str = "qwen2.5:7b"
    ollama_host: str | None = None
    n_queries: int = 3
    k: int = 60
    top_k: int = 10

    _chain: Runnable = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _init_chain(self) -> "ParaphraseRetriever":
        host = self.ollama_host or os.environ.get("OLLAMA_HOST")
        llm = ChatOllama(model=self.model, base_url=host) if host else ChatOllama(model=self.model)
        self._chain = _PARAPHRASE_PROMPT | llm | StrOutputParser()
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """言い換えクエリを生成して検索し、RRF で統合した結果を返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            RRF スコア降順で上位 top_k 件の Document リスト。
        """
        text = self._chain.invoke({"query": query, "n": self.n_queries})
        queries = [query] + _parse_queries(text)
        results_list = [self.base_retriever.invoke(q) for q in queries]
        return _rrf_merge(results_list, self.k)[: self.top_k]


class DecomposeRetriever(BaseRetriever):
    """クエリをサブクエリに分解して複数検索し RRF で統合する Retriever。

    複合的な質問を複数のシンプルなサブクエリに分解し、各側面を個別に検索する。
    単一クエリでは拾えない複数トピックをカバーできる。

    Attributes:
        base_retriever: 各サブクエリの検索に使う Retriever。
        model: Ollama のモデル名。
        ollama_host: Ollama サーバーの URL。None のとき環境変数 OLLAMA_HOST を参照。
        n_queries: 分解するサブクエリの数。
        k: RRF の平滑化定数。
        top_k: 返す上位件数。

    Example:
        >>> dense = build_dense_retriever(store, top_k=10)
        >>> retriever = DecomposeRetriever(base_retriever=dense, n_queries=3)
        >>> results = retriever.invoke("uvとpipの速度と再現性の違いは？")
    """

    base_retriever: BaseRetriever
    model: str = "qwen2.5:7b"
    ollama_host: str | None = None
    n_queries: int = 3
    k: int = 60
    top_k: int = 10

    _chain: Runnable = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _init_chain(self) -> "DecomposeRetriever":
        host = self.ollama_host or os.environ.get("OLLAMA_HOST")
        llm = ChatOllama(model=self.model, base_url=host) if host else ChatOllama(model=self.model)
        self._chain = _DECOMPOSE_PROMPT | llm | StrOutputParser()
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """サブクエリを生成して検索し、RRF で統合した結果を返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            RRF スコア降順で上位 top_k 件の Document リスト。
        """
        text = self._chain.invoke({"query": query, "n": self.n_queries})
        queries = _parse_queries(text) or [query]
        results_list = [self.base_retriever.invoke(q) for q in queries]
        return _rrf_merge(results_list, self.k)[: self.top_k]
