"""テキスト埋め込みモデルのラッパー。

LangChain の Embeddings インターフェース（embed_documents / embed_query）を実装する。
このインターフェースを満たすことで FAISS などの VectorStore とそのまま繋がる。

ruri-v3 / bge-m3 系モデルはクエリと文書でプレフィックスを使い分けることで精度が上がるよう
学習されている。HuggingFaceEmbeddings はこの切り替えに対応していないため、
Embeddings を継承した薄いラッパー（PrefixedEmbeddings）を提供する。
"""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class PrefixedEmbeddings(Embeddings):
    """クエリ・文書でプレフィックスを切り替える Embeddings 実装。

    ruri-v3-310m をデフォルトとし、同じプレフィックス仕様を持つモデル
    （bge-m3 など）にも対応する。

    Attributes:
        model_name: 使用する SentenceTransformer のモデル名。
        query_prefix: クエリに付与するプレフィックス。
        doc_prefix: 文書に付与するプレフィックス。
        batch_size: エンコード時のバッチサイズ。

    Example:
        >>> emb = PrefixedEmbeddings()
        >>> vecs = emb.embed_documents(["LangChainとは何か"])
        >>> query_vec = emb.embed_query("LangChainの使い方")
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-310m",
        query_prefix: str = "検索クエリ: ",
        doc_prefix: str = "検索文書: ",
        batch_size: int = 64,
    ) -> None:
        """
        Args:
            model_name: 使用する SentenceTransformer のモデル名。
            query_prefix: クエリに付与するプレフィックス。
            doc_prefix: 文書に付与するプレフィックス。
            batch_size: エンコード時のバッチサイズ。
        """
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文書リストをベクトル化する（インデックス構築時に呼ばれる）。

        Args:
            texts: エンコード対象のテキストリスト。

        Returns:
            L2 正規化済みベクトルのリスト。shape は (len(texts), dim)。
        """
        vecs = self._model.encode(
            [self._doc_prefix + t for t in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        """クエリ1件をベクトル化する（検索時に呼ばれる）。

        Args:
            text: 検索クエリ文字列。

        Returns:
            L2 正規化済みベクトル。長さ dim のリスト。
        """
        vec = self._model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32").tolist()
