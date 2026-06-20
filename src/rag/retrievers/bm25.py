"""キーワード検索 Retriever（BM25 + 日本語形態素解析）。

langchain_community の BM25Retriever は日本語（fugashi）に対応していないため、
BaseRetriever を継承して独自実装する。
分かち書きは fugashi（MeCab ラッパー）、スコアリングは rank_bm25 の BM25Okapi を使用。
"""

from copy import copy

from fugashi import Tagger
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr, model_validator
from rank_bm25 import BM25Okapi


class JapaneseBM25Retriever(BaseRetriever):
    """fugashi で分かち書きし BM25Okapi でスコアリングする日本語キーワード検索 Retriever。

    Attributes:
        docs: インデックス対象の Document リスト。
        k: 返す件数。

    Example:
        >>> chunks = split_documents(docs)
        >>> retriever = JapaneseBM25Retriever.from_documents(chunks, k=10)
        >>> results = retriever.invoke("日本語の検索クエリ")
    """

    docs: list[Document]
    k: int = 10

    _tagger: Tagger = PrivateAttr()
    _bm25: BM25Okapi = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _build_index(self) -> "JapaneseBM25Retriever":
        self._tagger = Tagger()
        tokenized = [self._tokenize(doc.page_content) for doc in self.docs]
        self._bm25 = BM25Okapi(tokenized)
        return self

    def _tokenize(self, text: str) -> list[str]:
        return [word.surface for word in self._tagger(text)]

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 10) -> "JapaneseBM25Retriever":
        """Document リストからインデックスを構築して Retriever を返す。

        Args:
            docs: インデックス対象の Document リスト。
            k: 返す件数。

        Returns:
            インデックス構築済みの JapaneseBM25Retriever。
        """
        return cls(docs=docs, k=k)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """クエリに関連するドキュメントを BM25 スコア順で返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            BM25 スコア降順で上位 k 件の Document リスト。スコアが 0 の文書は除外する。
            各 Document の metadata に bm25_score が追加される。
        """
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            [
                (doc, float(score))
                for doc, score in zip(self.docs, scores, strict=True)
                if score > 0
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        results = []
        for doc, score in ranked[: self.k]:
            new_doc = copy(doc)
            new_doc.metadata = {**doc.metadata, "bm25_score": score}
            results.append(new_doc)
        return results
