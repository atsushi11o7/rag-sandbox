"""ベクトル検索 Retriever（Dense）。

FAISS.as_retriever() で LangChain の VectorStoreRetriever を生成する。
インデックスの構築・保存・読み込みは src.rag.store が担い、
このモジュールは検索の入口だけを提供する。
"""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever


def build_dense_retriever(
    store: FAISS,
    top_k: int = 10,
    mmr: bool = False,
    lambda_mult: float = 0.5,
) -> BaseRetriever:
    """FAISS ストアから Dense Retriever を生成する。

    Args:
        store: build_faiss または load_faiss で得た FAISS オブジェクト。
        top_k: 返す件数。
        mmr: True のとき MMR（Maximal Marginal Relevance）で多様性を考慮した検索を行う。
        lambda_mult: MMR の関連性と多様性のバランス（0.0=多様性重視, 1.0=関連性重視）。mmr=True のときのみ有効。

    Returns:
        VectorStoreRetriever インスタンス。

    Example:
        >>> emb = PrefixedEmbeddings()
        >>> store = load_faiss("data/index/faiss", emb)
        >>> retriever = build_dense_retriever(store, top_k=10)
        >>> retriever_mmr = build_dense_retriever(store, top_k=10, mmr=True, lambda_mult=0.5)
    """
    if mmr:
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "lambda_mult": lambda_mult},
        )
    return store.as_retriever(search_kwargs={"k": top_k})
