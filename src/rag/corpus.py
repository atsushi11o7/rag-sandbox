"""コーパスの読み込み。

data/corpus/ 以下の .md ファイルを LangChain の Document 型として読み込む。
Document は page_content（本文）と metadata（doc_id など付加情報）を持つデータ型で、
LangChain の TextSplitter・VectorStore・Retriever など全コンポーネントが共通して扱う。
"""

import json
from pathlib import Path

from langchain_core.documents import Document


def load_md_corpus(corpus_dir: str) -> list[Document]:
    """ディレクトリ内の .md ファイルを LangChain Document として読み込む。

    ファイル名（拡張子なし）を doc_id として metadata に格納する。
    同名の .json サイドカー（fetch_qiita.py が生成）があれば title / created_at /
    tags / likes_count / url をメタデータに追加する。
    ファイルはアルファベット順にソートして返す。

    Args:
        corpus_dir: .md ファイルが格納されたディレクトリのパス。

    Returns:
        Document のリスト。各 Document の metadata には doc_id が含まれる。

    Example:
        >>> docs = load_md_corpus("data/corpus")
        >>> docs[0].metadata
        {"doc_id": "b44ac565651bdc68db1a", "tags": ["Python", "RAG"], ...}
    """
    paths = sorted(Path(corpus_dir).glob("*.md"))
    docs = []
    for p in paths:
        meta: dict = {"doc_id": p.stem}
        sidecar = p.with_suffix(".json")
        if sidecar.exists():
            meta.update(json.loads(sidecar.read_text(encoding="utf-8")))
        docs.append(Document(page_content=p.read_text(encoding="utf-8"), metadata=meta))
    return docs
