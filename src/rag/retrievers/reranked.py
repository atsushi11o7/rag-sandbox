"""リランク付き Retriever。

任意の Retriever の候補を CrossEncoderReranker で再ランク付けするラッパー。
ContextualCompressionRetriever は langchain_classic（レガシー）にしかないため
BaseRetriever で独自実装する。

二段階の流れ:
    1. base_retriever で candidate_k 件を粗く取得
    2. CrossEncoderReranker で精密にスコア付けして上位 top_n 件を返す
"""

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag.rerank import CrossEncoderReranker


class RerankedRetriever(BaseRetriever):
    """base_retriever の候補を CrossEncoderReranker で再ランク付けする Retriever。

    Attributes:
        base_retriever: 第一段階の粗い検索を担う Retriever。
        reranker: 第二段階のスコア付けを担う CrossEncoderReranker。
        candidate_k: base_retriever から取得する候補件数。

    Example:
        >>> base = build_dense_retriever(store, top_k=50)
        >>> reranker = CrossEncoderReranker(top_n=5)
        >>> retriever = RerankedRetriever(base_retriever=base, reranker=reranker)
        >>> results = retriever.invoke("検索クエリ")
    """

    base_retriever: BaseRetriever
    reranker: CrossEncoderReranker
    candidate_k: int = 50

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """候補を取得して再ランク付けした結果を返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            CrossEncoderReranker による再ランク付け後の Document リスト。
        """
        candidates = self.base_retriever.invoke(query)
        return self.reranker.compress_documents(candidates[: self.candidate_k], query)
