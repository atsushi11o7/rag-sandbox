"""LightRAG による GraphRAG デモ。

同じクエリを naive / local / global / hybrid / mix の 5 モードで実行して結果を比較する。
初回実行時はコーパスを挿入してグラフを構築する（data/lightrag/ に保存）。
2 回目以降は保存済みのグラフを再利用するため挿入をスキップできる。

使い方:
    python experiments/run_lightrag.py --query "uvの利点は？"
    python experiments/run_lightrag.py --query "uvの利点は？" --skip-insert
    python experiments/run_lightrag.py --query "uvの利点は？" --mode hybrid
"""

import argparse
import asyncio

from src.rag.corpus import load_md_corpus
from src.rag.kg.lightrag_store import build_lightrag, insert_documents, query

CORPUS_DIR = "data/corpus"
WORKING_DIR = "data/lightrag"
MODES = ["naive", "local", "global", "hybrid", "mix"]


async def main(args: argparse.Namespace) -> None:
    print("Initializing LightRAG...")
    rag = await build_lightrag(working_dir=WORKING_DIR, llm_model=args.model)

    if not args.skip_insert:
        print(f"Loading corpus from {CORPUS_DIR}...")
        docs = load_md_corpus(CORPUS_DIR)
        if args.limit:
            docs = docs[: args.limit]
        print(f"{len(docs)} documents loaded. Inserting into knowledge graph...")
        print("(This may take a while — LLM extracts entities from each document)\n")
        await insert_documents(rag, docs)
        print("Insertion complete.\n")

    modes = [args.mode] if args.mode else MODES

    print(f"Query: {args.query!r}\n")
    print("=" * 60)

    for mode in modes:
        print(f"\n[{mode.upper()}]")
        print("-" * 40)
        result = await query(rag, args.query, mode=mode)
        print(result)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightRAG GraphRAG demo")
    parser.add_argument(
        "--query",
        default="uvの利点と使い方を教えてください",
        help="検索クエリ",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="クエリモードを1つ指定（省略時は全5モードを実行）",
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true",
        help="コーパス挿入をスキップして既存グラフを再利用する",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama モデル名",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="挿入するドキュメント数の上限",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
