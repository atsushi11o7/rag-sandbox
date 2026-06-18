"""ハイブリッド検索 Retriever（Reciprocal Rank Fusion）。

複数の Retriever の結果を Reciprocal Rank Fusion（RRF）で統合する。
EnsembleRetriever は langchain_classic（レガシー）にしかないため BaseRetriever で独自実装する。

RRF スコア = Σ 1 / (k + rank)
k=60 はランキングの上位・下位の差を平滑化するための定数（論文デフォルト値）。
"""

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class HybridRetriever(BaseRetriever):
    """複数 Retriever の結果を RRF で融合するハイブリッド検索 Retriever。

    Attributes:
        retrievers: 融合対象の Retriever リスト（Dense + BM25 など）。
        k: RRF の平滑化定数。
        candidate_k: 各 Retriever から取得する候補件数。

    Example:
        >>> dense = build_dense_retriever(store, top_k=50)
        >>> bm25 = JapaneseBM25Retriever.from_documents(chunks, k=50)
        >>> retriever = HybridRetriever(retrievers=[dense, bm25], candidate_k=50)
        >>> results = retriever.invoke("検索クエリ")
    """

    retrievers: list[BaseRetriever]
    k: int = 60
    candidate_k: int = 50

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """各 Retriever の結果を RRF で統合して返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            RRF スコア降順でソートされた Document リスト。
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for retriever in self.retrievers:
            results = retriever.invoke(query)
            for rank, doc in enumerate(results[: self.candidate_k]):
                key = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id", doc.page_content[:50])
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.k + rank + 1)
                doc_map[key] = doc

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]
