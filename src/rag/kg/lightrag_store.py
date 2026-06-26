"""LightRAG を使ったナレッジグラフ構築・検索モジュール。

LightRAG はドキュメント挿入時に LLM でエンティティ・関係を抽出してナレッジグラフを構築し、
5 つのクエリモード（naive / local / global / hybrid / mix）で検索・回答生成を行う。

このモジュールは：
- LightRAG インスタンスの初期化
- コーパスの挿入（グラフ構築）
- クエリの実行

を担う。グラフストレージは NetworkXStorage（追加サービス不要）を使用する。
埋め込みは Ollama の mxbai-embed-large（1024次元）を使用する。
LightRAG の ollama_embed はデフォルトで 1024次元を前提としているため、この組み合わせが素直に動く。
（lightrag.llm.ollama.ollama_embed のデフォルトモデルは bge-m3:latest / 1024次元。
mxbai-embed-large も同じ 1024次元なのでそのまま使える）
"""

import os
from pathlib import Path

from langchain_core.documents import Document
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

_WORKING_DIR = "data/lightrag"
_EMBED_MODEL = "mxbai-embed-large"
_EMBED_DIM = 1024


async def build_lightrag(
    working_dir: str = _WORKING_DIR,
    llm_model: str = "qwen2.5:7b",
    embed_model: str = _EMBED_MODEL,
    ollama_host: str | None = None,
) -> LightRAG:
    """LightRAG インスタンスを初期化して返す。

    グラフは NetworkXStorage（インメモリ + GraphML ファイル）に保存される。
    working_dir が存在しない場合は自動作成する。
    initialize_storages() を内部で呼ぶため async 関数になっている。

    Args:
        working_dir: グラフ・キャッシュの保存ディレクトリ。
        llm_model: Ollama の LLM モデル名。
        embed_model: Ollama の埋め込みモデル名。
        ollama_host: Ollama サーバーの URL。None のとき環境変数 OLLAMA_HOST を参照。

    Returns:
        初期化済みの LightRAG インスタンス。
    """
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    host = ollama_host or os.environ.get("OLLAMA_HOST")

    llm_kwargs: dict = {"host": host} if host else {}
    embed_kwargs: dict = {"host": host} if host else {}

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs=llm_kwargs,
        llm_model_max_async=1,
        embedding_func=EmbeddingFunc(
            embedding_dim=_EMBED_DIM,
            max_token_size=512,  # mxbai-embed-large の最大入力長
            func=lambda texts: ollama_embed(texts, embed_model=embed_model, **embed_kwargs),
        ),
        graph_storage="NetworkXStorage",
    )
    await rag.initialize_storages()
    return rag


async def insert_documents(rag: LightRAG, docs: list[Document]) -> None:
    """Document リストを LightRAG に挿入してナレッジグラフを構築する。

    各 Document の page_content を挿入する。タイトルが metadata にある場合は
    冒頭に付与してエンティティ抽出の精度を上げる。

    Args:
        rag: build_lightrag で作成した LightRAG インスタンス。
        docs: 挿入対象の Document リスト。
    """
    texts = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        body = doc.page_content
        texts.append(f"# {title}\n\n{body}" if title else body)

    for i, text in enumerate(texts, 1):
        print(f"  [{i}/{len(texts)}] inserting...")
        await rag.ainsert(text)


async def query(
    rag: LightRAG,
    question: str,
    mode: str = "hybrid",
) -> str:
    """ナレッジグラフに対してクエリを実行して回答を返す。

    Args:
        rag: build_lightrag で作成した LightRAG インスタンス。
        question: 質問文字列。
        mode: クエリモード。"local" / "global" / "hybrid" / "mix" のいずれか。

    Returns:
        LightRAG が生成した回答文字列。
    """
    return await rag.aquery(question, param=QueryParam(mode=mode))
