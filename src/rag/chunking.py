"""文書のチャンク分割。

LangChain の RecursiveCharacterTextSplitter を使用する。
固定文字数で機械的に切る CharacterTextSplitter と異なり、
区切り文字の優先順位に従って自然な境界で分割する。

デフォルトの separators は英語向け（\n\n → \n → スペース → 文字単位）のため、
日本語コーパスでは句点・読点を含むリストを明示的に設定する:
  ["\n\n", "\n", "。", "、", " ", ""]

入力の Document が持つ metadata（doc_id など）はチャンクにそのまま引き継がれる。
各チャンクには chunk_id（"doc_id#チャンク番号" 形式）を追加する。
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> list[Document]:
    """Document リストをチャンク単位に分割する。

    Args:
        docs: 分割対象の Document リスト。
        chunk_size: 1チャンクの最大文字数。
        chunk_overlap: 隣接チャンク間で重複させる文字数。
        separators: 区切り文字の優先順位リスト。
            None のときは日本語向けデフォルト（段落→改行→句点→読点→空白→文字単位）を使用。

    Returns:
        チャンク単位の Document リスト。各チャンクの metadata には
        元の doc_id に加え chunk_id（"doc_id#i" 形式）が含まれる。

    Example:
        >>> docs = load_md_corpus("data/corpus")
        >>> chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
        >>> chunks[0].metadata
        {"doc_id": "b44ac565651bdc68db1a", "chunk_id": "b44ac565651bdc68db1a#0"}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators if separators is not None else _DEFAULT_SEPARATORS,
    )
    chunks = []
    for doc in docs:
        doc_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}#{i}"
        chunks.extend(doc_chunks)
    return chunks
