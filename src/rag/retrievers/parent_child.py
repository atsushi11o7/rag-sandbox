"""Parent-Child Retriever のセットアップヘルパー。

小さい子チャンク（child）でベクトル検索し、ヒットした親チャンク（parent）を返す手法。
子チャンクは意味が凝縮されているため検索精度が高く、
LLM には文脈が多い親チャンクを渡すことで回答品質を両立する。

流れ:
    1. ドキュメントを親チャンク（大）→ 子チャンク（小）の2段階で分割
    2. 子チャンクを FAISS にインデックス
    3. 親チャンクを InMemoryStore に保存（parent_id で紐付け）
    4. 検索時は子チャンクで探して対応する親チャンクを返す

LangChain Classic の ParentDocumentRetriever が上記をすべて担う。
"""

import faiss
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import PrefixedEmbeddings

_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]


def build_parent_child_retriever(
    docs: list[Document],
    embeddings: PrefixedEmbeddings | None = None,
    parent_size: int = 1000,
    child_size: int = 200,
    child_k: int = 50,
) -> ParentDocumentRetriever:
    """Parent-Child Retriever を構築して返す。

    ドキュメントの分割・インデックス・docstore への登録をまとめて行う。
    返り値は BaseRetriever を継承しているため、RerankedRetriever などのラッパーと
    組み合わせて使える。

    Args:
        docs: コーパスの Document リスト（load_md_corpus の出力）。
        embeddings: 埋め込みモデル。None のとき PrefixedEmbeddings() をデフォルトで使用。
        parent_size: 親チャンクの最大文字数。LLM に渡す文脈の単位。
        child_size: 子チャンクの最大文字数。ベクトル検索の単位。
        child_k: 子チャンクの検索件数。Reranker と組み合わせる場合は多めに設定する。

    Returns:
        ParentDocumentRetriever。invoke() で検索すると親チャンクが返る。

    Example:
        >>> docs = load_md_corpus("data/corpus")
        >>> retriever = build_parent_child_retriever(docs)
        >>> results = retriever.invoke("uvの利点は？")
        >>> len(results[0].page_content) > 500  # 親チャンクは大きい
        True
    """
    emb = embeddings or PrefixedEmbeddings()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=0, separators=_DEFAULT_SEPARATORS
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=0, separators=_DEFAULT_SEPARATORS
    )

    # faiss で空インデックスを直接作成
    dim = len(emb.embed_query("dimension check"))
    index = faiss.IndexFlatL2(dim)
    vectorstore = FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={"k": child_k},
    )
    retriever.add_documents(docs)
    return retriever
