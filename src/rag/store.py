"""FAISS ベクトルストアの構築・保存・読み込み。

langchain_community の FAISS ラッパーを使用する。
raw の faiss と異なり、次元数の管理や metadata の保存が自動で行われる。
構築した FAISS オブジェクトは .as_retriever() で直接 Retriever として利用できる。

ディスク保存形式は save_local / load_local（FAISS バイナリ + pickle）。
load_local には allow_dangerous_deserialization=True が必須（LangChain のセキュリティ仕様）。
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_faiss(
    docs: list[Document],
    embeddings: Embeddings,
    save_dir: str,
) -> FAISS:
    """Document リストから FAISS インデックスを構築してディスクに保存する。

    Args:
        docs: インデックス対象の Document リスト（chunk_id / doc_id を metadata に持つ）。
        embeddings: ベクトル化に使用する Embeddings 実装。
        save_dir: インデックスの保存先ディレクトリ（data/index/ 以下を推奨）。

    Returns:
        構築済みの FAISS オブジェクト。

    Example:
        >>> emb = PrefixedEmbeddings()
        >>> store = build_faiss(chunks, emb, "data/index/faiss")
        >>> retriever = store.as_retriever(search_kwargs={"k": 5})
    """
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(save_dir)
    return store


def load_faiss(
    save_dir: str,
    embeddings: Embeddings,
) -> FAISS:
    """保存済みの FAISS インデックスを読み込む。

    Args:
        save_dir: build_faiss で指定した保存先ディレクトリ。
        embeddings: 構築時と同じ Embeddings 実装。

    Returns:
        読み込み済みの FAISS オブジェクト。

    Example:
        >>> emb = PrefixedEmbeddings()
        >>> store = load_faiss("data/index/faiss", emb)
        >>> retriever = store.as_retriever(search_kwargs={"k": 5})
    """
    return FAISS.load_local(
        save_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
