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


from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(BaseDocumentCompressor):
    """sentence_transformers の CrossEncoder でドキュメントを再ランク付けする compressor。

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
    _model: CrossEncoder | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            object.__setattr__(self, "_model", CrossEncoder(self.model_name))
        return self._model

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        """クエリとの関連度でドキュメントを再ランク付けして上位 top_n 件を返す。

        Args:
            documents: 再ランク対象の Document リスト。
            query: 検索クエリ文字列。
            callbacks: LangChain コールバック（省略可）。

        Returns:
            関連度スコア降順で上位 top_n 件の Document リスト。
        """
        if not documents:
            return []
        model = self._get_model()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]
