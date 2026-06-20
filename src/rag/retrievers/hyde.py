"""HyDE（Hypothetical Document Embeddings）Retriever。

クエリに対して LLM で仮回答を生成し、その仮回答文でベクトル検索を行う手法。
抽象的なクエリや短いクエリで、直接クエリを埋め込むより精度が上がることがある。

流れ:
    1. LLM（Ollama）でクエリに対する仮回答を生成
    2. 仮回答文を base_retriever で検索
    3. 結果を返す
"""

import os

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaLLM
from pydantic import PrivateAttr, model_validator

_PROMPT = "必ず日本語のみで答えてください。\n次の質問に答える日本語の文章を簡潔に書いてください。\n質問: {query}\n回答:"


class HydeRetriever(BaseRetriever):
    """仮回答を生成してから base_retriever で検索する HyDE Retriever。

    Attributes:
        base_retriever: 仮回答文を使って検索する Retriever（Dense 推奨）。
        model: Ollama のモデル名。
        ollama_host: Ollama サーバーのホスト URL。None のとき環境変数 OLLAMA_HOST を参照。

    Example:
        >>> dense = build_dense_retriever(store, top_k=10)
        >>> retriever = HydeRetriever(base_retriever=dense, model="qwen2.5:7b")
        >>> results = retriever.invoke("RAGの精度を上げるには")
    """

    base_retriever: BaseRetriever
    model: str = "qwen2.5:7b"
    ollama_host: str | None = None

    _llm: OllamaLLM = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _init_llm(self) -> "HydeRetriever":
        host = self.ollama_host or os.environ.get("OLLAMA_HOST")
        self._llm = (
            OllamaLLM(model=self.model, base_url=host) if host else OllamaLLM(model=self.model)
        )
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """仮回答を生成して検索した結果を返す。

        Args:
            query: 検索クエリ文字列。
            run_manager: LangChain コールバックマネージャー。

        Returns:
            仮回答文で検索した結果の Document リスト。
        """
        hypothesis = self._llm.invoke(_PROMPT.format(query=query))
        return self.base_retriever.invoke(hypothesis)
