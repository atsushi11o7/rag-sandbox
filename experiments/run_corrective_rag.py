"""Corrective RAG のデモ。

検索した文書の関連性を LLM で判定し、不十分なら クエリを書き換えて再検索する。
LangGraph でノードとエッジを組み、能動的な検索ループを実現する。

使い方:
    python experiments/run_corrective_rag.py                    # デフォルトクエリ
    python experiments/run_corrective_rag.py "uvの利点は？"     # クエリを指定
"""

import argparse

from langchain_community.vectorstores import FAISS

from src.rag.chunking import split_documents
from src.rag.corpus import load_md_corpus
from src.rag.embeddings import PrefixedEmbeddings
from src.rag.generation import RAGGenerator
from src.rag.graph import build_corrective_rag
from src.rag.rerank import CrossEncoderReranker
from src.rag.retrievers import (
    HybridRetriever,
    JapaneseBM25Retriever,
    RerankedRetriever,
    build_dense_retriever,
)

CORPUS_DIR = "data/corpus"
CANDIDATE_K = 50
TOP_K = 5
DEFAULT_QUERY = "RAGの精度を上げるにはどうすればいいですか？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrective RAG デモ")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="検索クエリ")
    args = parser.parse_args()

    print("Loading corpus and building index...")
    docs = load_md_corpus(CORPUS_DIR)
    chunks = split_documents(docs)

    store = FAISS.from_documents(chunks, PrefixedEmbeddings())
    dense = build_dense_retriever(store, top_k=CANDIDATE_K)
    bm25 = JapaneseBM25Retriever.from_documents(chunks, k=CANDIDATE_K)
    hybrid = HybridRetriever(retrievers=[dense, bm25], candidate_k=CANDIDATE_K, top_k=CANDIDATE_K)
    retriever = RerankedRetriever(
        base_retriever=hybrid,
        reranker=CrossEncoderReranker(top_n=TOP_K),
    )
    generator = RAGGenerator()
    app = build_corrective_rag(retriever, generator, max_retries=2)

    print(f"\nQuery: {args.query}")
    print("\nRunning Corrective RAG...")

    full_state: dict = {"original_query": args.query, "search_query": args.query}
    current_query = args.query
    done = False
    for step in app.stream(
        {"original_query": args.query, "search_query": args.query, "retries": 0}
    ):
        node_name, state = next(iter(step.items()))
        full_state.update(state)
        if node_name == "retrieve":
            print(f"\n[retrieve] query={current_query!r}")
            print(f"           {len(state['documents'])} docs retrieved")
        elif node_name == "grade_documents":
            n_relevant = len(state.get("documents", []))
            print(f"[grade]    {state['grade']}  ({n_relevant} relevant docs)")
        elif node_name == "rewrite_query":
            current_query = state["search_query"]
            print(f"[rewrite]  {current_query!r}  (retries={state['retries']})")
        elif node_name == "generate":
            done = True

    if not done:
        print("(no answer generated)")
        return

    print(f"\nRetrieved {len(full_state['documents'])} documents")
    for i, doc in enumerate(full_state["documents"], 1):
        chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("doc_id", "unknown"))
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] {chunk_id}")
        print(f"       {preview}...")

    print(f"\n=== Answer ===\n{full_state['answer']}")


if __name__ == "__main__":
    main()
