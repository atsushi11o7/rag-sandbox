"""タグによるメタデータフィルタ Retriever。

base_retriever の検索結果を Document.metadata["tags"] で絞り込む。
コーパスが fetch_qiita.py + load_md_corpus() で読み込まれていれば
各チャンクに tags が付いており、そのまま使える。

流れ:
    1. base_retriever で候補を取得
    2. メタデータの tags に指定タグが 1 つでも含まれるものだけ残す
    3. 残った Document を返す（順序はそのまま）
"""

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class TagFilteredRetriever(BaseRetriever):
    """タグで検索結果を絞り込む Retriever。

    base_retriever が返した Document のうち、metadata["tags"] に
    指定したタグが 1 つ以上含まれるものだけを返す。
    タグが付いていない Document（サイドカー .json なし）は除外される。

    Attributes:
        base_retriever: 一次検索に使う Retriever。
        tags: フィルタ条件のタグリスト（いずれか 1 つが一致すれば通過）。

    Example:
        >>> dense = build_dense_retriever(store, top_k=20)
        >>> retriever = TagFilteredRetriever(base_retriever=dense, tags=["Python"])
        >>> results = retriever.invoke("パッケージ管理の方法は？")
        >>> all("Python" in doc.metadata.get("tags", []) for doc in results)
        True
    """

    base_retriever: BaseRetriever
    tags: list[str]

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """タグ条件を満たす Document だけを返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            タグが一致した Document のリスト。
        """
        tag_set = set(self.tags)
        candidates = self.base_retriever.invoke(query)
        return [doc for doc in candidates if tag_set & set(doc.metadata.get("tags", []))]
