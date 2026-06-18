"""複数 Retriever を共通指標で横並び評価する。

流れ:
    1. コーパス読み込み → チャンク分割
    2. FAISS インデックス構築（インメモリ）
    3. 各 Retriever で全クエリを検索
    4. nDCG / MRR / Recall を平均して出力
"""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag.chunking import split_documents
from src.rag.corpus import load_md_corpus
from src.rag.embeddings import PrefixedEmbeddings
from src.rag.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.rag.retrievers import HybridRetriever, JapaneseBM25Retriever, build_dense_retriever

CORPUS_DIR = "data/corpus"
QRELS_PATH = "data/qrels.jsonl"
TOP_K = 10


def load_qrels(path: str) -> list[dict]:
    """{"query": ..., "positive_doc_ids": [...]} 形式の jsonl を読む。

    Args:
        path: qrels.jsonl のファイルパス。

    Returns:
        クエリと正解 doc_id リストの辞書リスト。
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def to_doc_ids(docs: list[Document]) -> list[str]:
    """チャンク Document を doc_id 単位に重複排除しながら順序を保つ。

    chunk_id が "doc_id#i" 形式のとき "#" より前を doc_id として扱う。

    Args:
        docs: Retriever が返した Document リスト。

    Returns:
        重複排除済みの doc_id リスト（出現順）。
    """
    seen: list[str] = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id", "")
        doc_id = chunk_id.split("#")[0]
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
    return seen


def evaluate(retriever: BaseRetriever, qrels: list[dict]) -> dict[str, float]:
    """全クエリの nDCG / MRR / Recall を平均して返す。

    Args:
        retriever: 評価対象の Retriever。
        qrels: load_qrels で読み込んだ評価データ。

    Returns:
        {"nDCG": float, "MRR": float, "Recall": float} の辞書。
    """
    ndcg = mrr = recall = 0.0
    for q in qrels:
        relevant = set(q["positive_doc_ids"])
        ranked = to_doc_ids(retriever.invoke(q["query"]))
        ndcg += ndcg_at_k(ranked, relevant, TOP_K)
        mrr += mrr_at_k(ranked, relevant, TOP_K)
        recall += recall_at_k(ranked, relevant, TOP_K)
    n = len(qrels)
    return {"nDCG": ndcg / n, "MRR": mrr / n, "Recall": recall / n}


def main() -> None:
    print("Loading corpus and building index...")
    docs = load_md_corpus(CORPUS_DIR)
    chunks = split_documents(docs)

    embeddings = PrefixedEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)

    dense = build_dense_retriever(store, top_k=TOP_K)
    bm25 = JapaneseBM25Retriever.from_documents(chunks, k=TOP_K)
    hybrid = HybridRetriever(retrievers=[dense, bm25])

    qrels = load_qrels(QRELS_PATH)

    retrievers: dict[str, BaseRetriever] = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
    }

    print(f"\n{'retriever':<10} {'nDCG@k':>8} {'MRR@k':>8} {'Recall@k':>9}")
    print("-" * 38)
    for name, retriever in retrievers.items():
        m = evaluate(retriever, qrels)
        print(f"{name:<10} {m['nDCG']:>8.4f} {m['MRR']:>8.4f} {m['Recall']:>9.4f}")


if __name__ == "__main__":
    main()
