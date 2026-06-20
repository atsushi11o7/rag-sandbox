"""HyDE（Hypothetical Document Embeddings）のデモ。

クエリに対して LLM で仮回答を生成し、その仮回答文でベクトル検索を行う。
通常の Dense 検索と並べて、HyDE の効果を確認できる。

使い方:
    python experiments/run_hyde.py                          # デフォルトクエリ
    python experiments/run_hyde.py "RAGとは何ですか？"      # クエリを指定
"""

import argparse

from langchain_community.vectorstores import FAISS

from src.rag.chunking import split_documents
from src.rag.corpus import load_md_corpus
from src.rag.embeddings import PrefixedEmbeddings
from src.rag.generation import RAGGenerator
from src.rag.retrievers import HydeRetriever, build_dense_retriever

CORPUS_DIR = "data/corpus"
TOP_K = 5
DEFAULT_QUERY = "RAGの精度を上げるにはどうすればいいですか？"


def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE デモ")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="検索クエリ")
    args = parser.parse_args()

    print("Loading corpus and building index...")
    docs = load_md_corpus(CORPUS_DIR)
    chunks = split_documents(docs)

    store = FAISS.from_documents(chunks, PrefixedEmbeddings())
    dense = build_dense_retriever(store, top_k=TOP_K)
    hyde = HydeRetriever(base_retriever=dense)
    generator = RAGGenerator()

    print(f"\nQuery: {args.query}")

    # Dense との比較
    print("\n--- Dense (without HyDE) ---")
    dense_results = dense.invoke(args.query)
    for i, doc in enumerate(dense_results, 1):
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("doc_id", "unknown"))
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] {chunk_id}")
        print(f"       {preview}...")

    print("\n--- HyDE ---")
    print("Generating hypothesis...")
    hyde_results = hyde.invoke(args.query)
    for i, doc in enumerate(hyde_results, 1):
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("doc_id", "unknown"))
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] {chunk_id}")
        print(f"       {preview}...")

    print("\nGenerating answer with HyDE results...")
    answer = generator.generate(args.query, hyde_results)
    print(f"\n=== Answer ===\n{answer}")


if __name__ == "__main__":
    main()
