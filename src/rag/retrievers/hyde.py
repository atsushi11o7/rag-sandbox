"""HyDE（Hypothetical Document Embeddings）Retriever。

クエリに対して LLM で仮回答を生成し、その仮回答文でベクトル検索を行う手法。
抽象的なクエリや短いクエリで、直接クエリを埋め込むより精度が上がることがある。

流れ:
    1. LLM（Ollama）でクエリに対する仮回答を生成
    2. 仮回答文を base_retriever で検索
    3. include_original=True のとき、元クエリ検索とRRFで統合して返す
"""

import os

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import PrivateAttr, model_validator

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "必ず日本語のみで答えてください。"),
        ("user", "次の質問に答える日本語の文章を簡潔に書いてください。\n質問: {query}"),
    ]
)


def _rrf_merge(results_list: list[list[Document]], k: int = 60) -> list[Document]:
    """複数の検索結果リストを RRF で統合する。"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id", "")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


class HydeRetriever(BaseRetriever):
    """仮回答を生成してから base_retriever で検索する HyDE Retriever。

    LCEL チェーン（prompt | llm | parser）で仮回答を生成する。
    include_original=True にすると、元クエリ検索と仮回答検索を RRF で統合する。

    Attributes:
        base_retriever: 仮回答文を使って検索する Retriever（Dense 推奨）。
        model: Ollama のモデル名。
        ollama_host: Ollama サーバーのホスト URL。None のとき環境変数 OLLAMA_HOST を参照。
        include_original: True のとき元クエリでも検索して RRF で統合する。
            仮回答で固有名詞や条件が薄まるリスクを低減できる。
        k: RRF の平滑化定数。include_original=True のときのみ有効。
        top_k: RRF 統合後に返す件数。include_original=True のときのみ有効。

    Example:
        >>> dense = build_dense_retriever(store, top_k=10)
        >>> retriever = HydeRetriever(base_retriever=dense, model="qwen2.5:7b")
        >>> results = retriever.invoke("RAGの精度を上げるには")
        >>> # 元クエリと仮回答を RRF 統合する場合
        >>> retriever_with_orig = HydeRetriever(
        ...     base_retriever=dense, include_original=True, top_k=10
        ... )
    """

    base_retriever: BaseRetriever
    model: str = "qwen2.5:7b"
    ollama_host: str | None = None
    include_original: bool = False
    k: int = 60
    top_k: int = 10

    _chain: Runnable = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _init_chain(self) -> "HydeRetriever":
        host = self.ollama_host or os.environ.get("OLLAMA_HOST")
        llm = ChatOllama(model=self.model, base_url=host) if host else ChatOllama(model=self.model)
        self._chain = _PROMPT | llm | StrOutputParser()
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """仮回答を生成して検索した結果を返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            include_original=False のとき仮回答文で検索した結果。
            include_original=True のとき元クエリ検索と RRF 統合した結果（top_k 件）。
        """
        hypothesis = self._chain.invoke({"query": query})
        hyde_results = self.base_retriever.invoke(hypothesis)
        if not self.include_original:
            return hyde_results
        raw_results = self.base_retriever.invoke(query)
        return _rrf_merge([raw_results, hyde_results], self.k)[: self.top_k]
