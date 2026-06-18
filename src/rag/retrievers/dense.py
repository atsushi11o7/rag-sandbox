"""ベクトル検索 Retriever（Dense）。

FAISS.as_retriever() で LangChain の VectorStoreRetriever を生成する。
インデックスの構築・保存・読み込みは src.rag.store が担い、
このモジュールは検索の入口だけを提供する。
"""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever


def build_dense_retriever(store: FAISS, top_k: int = 10) -> BaseRetriever:
    """FAISS ストアから Dense Retriever を生成する。

    Args:
        store: build_faiss または load_faiss で得た FAISS オブジェクト。
        top_k: 返す件数。

    Returns:
        VectorStoreRetriever インスタンス。

    Example:
        >>> emb = PrefixedEmbeddings()
        >>> store = load_faiss("data/index/faiss", emb)
        >>> retriever = build_dense_retriever(store, top_k=10)
        >>> docs = retriever.invoke("LangChainとは")
    """
    return store.as_retriever(search_kwargs={"k": top_k})
