"""Cross-Encoder によるリランカー。

Reranker は二段階検索の第二段階として機能する。
第一段階（FAISS / BM25）で粗く候補を絞り、Cross-Encoder で精密にスコアを付け直す。

langchain_community の HuggingFaceCrossEncoder は sunset 予定のため、
sentence_transformers の CrossEncoder を直接使い、
langchain_core の BaseDocumentCompressor を継承して LangChain パイプラインに組み込む。

使い方:
    compressor = CrossEncoderReranker()
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
"""

from copy import copy

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import PrivateAttr, model_validator
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(BaseDocumentCompressor):
    """sentence_transformers の CrossEncoder でドキュメントを再ランク付けする compressor。

    rerank_score を metadata に追加して返す。

    Attributes:
        model_name: 使用する Cross-Encoder のモデル名。
        top_n: リランク後に返す上位件数。

    Example:
        >>> compressor = CrossEncoderReranker(top_n=5)
        >>> retriever = ContextualCompressionRetriever(
        ...     base_compressor=compressor,
        ...     base_retriever=base_retriever,
        ... )
    """

    model_name: str = "cl-nagoya/ruri-reranker-large"
    top_n: int = 5
    _model: CrossEncoder = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _init_model(self) -> "CrossEncoderReranker":
        self._model = CrossEncoder(self.model_name)
        return self

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        """クエリとの関連度でドキュメントを再ランク付けして上位 top_n 件を返す。

        rerank_score を各 Document の metadata に追加する。

        Args:
            documents: 再ランク対象の Document リスト。
            query: 検索クエリ文字列。
            callbacks: LangChain コールバック（省略可）。

        Returns:
            関連度スコア降順で上位 top_n 件の Document リスト。
            各 Document の metadata に rerank_score が追加される。
        """
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in ranked[: self.top_n]:
            new_doc = copy(doc)
            new_doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            results.append(new_doc)
        return results
