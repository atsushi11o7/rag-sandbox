"""Parent-Child Retriever + Reranker のデモ。

小さい子チャンクでベクトル検索し、対応する親チャンク（大きい文脈）を返したあと
CrossEncoder でリランクして LLM に渡す。

流れ:
    1. コーパス読み込み
    2. ParentChildRetriever 構築（child で検索 → parent を返す）
    3. RerankedRetriever でリランク
    4. LLM で回答を生成して出力

使い方:
    python experiments/run_parent_child.py                    # デフォルトクエリ
    python experiments/run_parent_child.py "uvの利点は？"     # クエリを指定
"""

import argparse

from src.rag.corpus import load_md_corpus
from src.rag.generation import RAGGenerator
from src.rag.rerank import CrossEncoderReranker
from src.rag.retrievers import RerankedRetriever, build_parent_child_retriever

CORPUS_DIR = "data/corpus"
PARENT_SIZE = 1000  # 親チャンク：LLM に渡す文脈の単位
CHILD_SIZE = 200  # 子チャンク：ベクトル検索の単位
TOP_K = 5  # リランク後に LLM へ渡す件数
DEFAULT_QUERY = "RAGの精度を上げるにはどうすればいいですか？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parent-Child + Rerank デモ")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="検索クエリ")
    args = parser.parse_args()

    print("Loading corpus and building Parent-Child index...")
    docs = load_md_corpus(CORPUS_DIR)

    base = build_parent_child_retriever(docs, parent_size=PARENT_SIZE, child_size=CHILD_SIZE)
    retriever = RerankedRetriever(
        base_retriever=base,
        reranker=CrossEncoderReranker(top_n=TOP_K),
    )
    generator = RAGGenerator()

    print(f"parent_size={PARENT_SIZE}  child_size={CHILD_SIZE}  top_k={TOP_K}")
    print(f"\nQuery: {args.query}")
    print("\nRetrieving...")
    retrieved = retriever.invoke(args.query)

    print(f"Retrieved {len(retrieved)} documents")
    for i, doc in enumerate(retrieved, 1):
        score = doc.metadata.get("rerank_score", "-")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        tags = doc.metadata.get("tags", [])
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] score={score_str}  len={len(doc.page_content)}{tag_str}")
        print(f"       {preview}...")

    print("\nGenerating answer...")
    answer = generator.generate(args.query, retrieved)
    print(f"\n=== Answer ===\n{answer}")


if __name__ == "__main__":
    main()
