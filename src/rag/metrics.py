"""検索評価指標（binary relevance）。

Retriever の性能を定量評価するための指標を提供する。
すべての指標は binary relevance を前提とし、上位 k 件の検索結果を対象とする。
LangChain に同等の実装がないため独自実装。

対応指標:
    - Recall@k  : 上位 k 件に含まれる正解の割合
    - MRR@k     : 最初に正解が現れた順位の逆数
    - nDCG@k    : 対数減衰を考慮した正規化累積利得
"""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """上位 k 件に含まれる正解ドキュメントの割合を返す。

    Args:
        ranked: 検索結果の doc_id リスト（スコア降順）。
        relevant: 正解 doc_id の集合。
        k: 評価対象とする上位件数。

    Returns:
        Recall@k の値（0.0〜1.0）。relevant が空の場合は 0.0。
    """
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """上位 k 件中で最初に正解が現れた順位の逆数を返す。

    Args:
        ranked: 検索結果の doc_id リスト（スコア降順）。
        relevant: 正解 doc_id の集合。
        k: 評価対象とする上位件数。

    Returns:
        MRR@k の値（0.0〜1.0）。k 件以内に正解がなければ 0.0。
    """
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """binary relevance での nDCG@k を返す。

    対数減衰 1/log2(i+2) を適用した DCG を、理想的な順序での IDCG で正規化する。

    Args:
        ranked: 検索結果の doc_id リスト（スコア降順）。
        relevant: 正解 doc_id の集合。
        k: 評価対象とする上位件数。

    Returns:
        nDCG@k の値（0.0〜1.0）。IDCG が 0 の場合（relevant が空）は 0.0。
    """
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0
