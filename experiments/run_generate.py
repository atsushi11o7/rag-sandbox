"""コーパスを検索し、LLM で回答を生成するエンドツーエンドのデモ。

流れ:
    1. コーパス読み込み → チャンク分割
    2. FAISS インデックス構築（インメモリ）
    3. hybrid+rerank で関連文書を取得（--tag 指定時はタグでさらに絞り込む）
    4. LLM で回答を生成して出力

使い方:
    python experiments/run_generate.py                          # デフォルトクエリ
    python experiments/run_generate.py "RAGとは何ですか？"      # クエリを指定
    python experiments/run_generate.py "uvの利点は？" --tag Python  # タグで絞り込み
    python experiments/run_generate.py "質問" --tag Python --tag uv # 複数タグ（OR）
"""

import argparse

from langchain_community.vectorstores import FAISS

from src.rag.chunking import split_documents
from src.rag.corpus import load_md_corpus
from src.rag.embeddings import PrefixedEmbeddings
from src.rag.generation import RAGGenerator
from src.rag.rerank import CrossEncoderReranker
from src.rag.retrievers import (
    HybridRetriever,
    JapaneseBM25Retriever,
    RerankedRetriever,
    TagFilteredRetriever,
    build_dense_retriever,
)

CORPUS_DIR = "data/corpus"
CANDIDATE_K = 50  # 一次検索（Dense / BM25）で取得する候補件数
TOP_K = 5  # リランク後に LLM へ渡す件数
DEFAULT_QUERY = "RAGの精度を上げるにはどうすればいいですか？"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 生成デモ")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="検索クエリ")
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="タグでフィルタ（複数指定可、OR 条件）",
    )
    args = parser.parse_args()

    print("Loading corpus and building index...")
    docs = load_md_corpus(CORPUS_DIR)
    chunks = split_documents(docs)

    embeddings = PrefixedEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)

    dense = build_dense_retriever(store, top_k=CANDIDATE_K)
    bm25 = JapaneseBM25Retriever.from_documents(chunks, k=CANDIDATE_K)
    hybrid = HybridRetriever(retrievers=[dense, bm25], candidate_k=CANDIDATE_K, top_k=CANDIDATE_K)

    if args.tags:
        print(f"Tag filter: {args.tags}")
        filtered = TagFilteredRetriever(base_retriever=hybrid, tags=args.tags)
    else:
        filtered = hybrid

    retriever = RerankedRetriever(
        base_retriever=filtered,
        reranker=CrossEncoderReranker(top_n=TOP_K),
    )

    generator = RAGGenerator()

    print(f"\nQuery: {args.query}")
    print("\nRetrieving...")
    retrieved = retriever.invoke(args.query)

    print(f"Retrieved {len(retrieved)} documents")
    for i, doc in enumerate(retrieved, 1):
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("doc_id", "unknown"))
        score = doc.metadata.get("rerank_score", "-")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        tags = doc.metadata.get("tags", [])
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] {chunk_id}  score={score_str}{tag_str}")
        print(f"       {preview}...")

    print("\nGenerating answer...")
    answer = generator.generate(args.query, retrieved)
    print(f"\n=== Answer ===\n{answer}")


if __name__ == "__main__":
    main()
