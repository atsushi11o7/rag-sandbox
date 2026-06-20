"""RAG の回答生成モジュール。

検索済み Document リストとクエリを受け取り、LLM で回答を生成する。
LCEL（LangChain Expression Language）で prompt | llm | parser のチェーンを組む。
"""

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, PrivateAttr, model_validator

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "必ず日本語のみで答えてください。"),
        ("user", "以下の文書を参考に、質問に答えてください。\n\n文書:\n{context}\n\n質問: {query}"),
    ]
)

_CONTEXT_SEP = "\n---\n"


class RAGGenerator(BaseModel):
    """検索結果を元に LLM で回答を生成するクラス。

    LCEL チェーン（prompt | llm | parser）を内部に持ち、複数クエリを効率よく処理する。

    Attributes:
        model: Ollama のモデル名。
        ollama_host: Ollama サーバーの URL。None のとき環境変数 OLLAMA_HOST を参照。

    Example:
        >>> generator = RAGGenerator()
        >>> answer = generator.generate("RAGとは？", docs)
    """

    model: str = "qwen2.5:7b"
    ollama_host: str | None = None

    _chain: Runnable = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _init_chain(self) -> "RAGGenerator":
        host = self.ollama_host or os.environ.get("OLLAMA_HOST")
        llm = ChatOllama(model=self.model, base_url=host) if host else ChatOllama(model=self.model)
        self._chain = _PROMPT | llm | StrOutputParser()
        return self

    def generate(self, query: str, docs: list[Document]) -> str:
        """検索結果を元に回答を生成する。

        Args:
            query: ユーザーの質問文字列。
            docs: 検索で取得した Document リスト。

        Returns:
            LLM が生成した回答文字列。
        """
        context = _CONTEXT_SEP.join(doc.page_content for doc in docs)
        return self._chain.invoke({"context": context, "query": query})
