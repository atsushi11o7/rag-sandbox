# rag-sandbox

RAG（Retrieval-Augmented Generation）の技術を実装・実験するサンドボックス。  
Dense 検索・BM25・ハイブリッド検索・Reranker・HyDE・親子チャンク・Corrective RAG など、  
各手法を共通インターフェースで実装し、差し替えや組み合わせを手軽に試せる構成にしている。

コーパスには自分の Qiita 記事を使用。完全ローカル・OSS 構成（FAISS / sentence-transformers / Ollama）で動作する。

## 実装済みの手法

| カテゴリ | 手法 | モジュール |
|---|---|---|
| 検索 | Dense（FAISS） | `src/rag/retrievers/dense.py` |
| 検索 | BM25 + 日本語形態素解析 | `src/rag/retrievers/bm25.py` |
| 検索 | ハイブリッド（RRF） | `src/rag/retrievers/hybrid.py` |
| 検索 | Reranker（Cross-Encoder） | `src/rag/rerank.py` |
| 検索 | メタデータフィルタリング（タグ） | `src/rag/retrievers/filtered.py` |
| クエリ変換 | HyDE（仮回答生成） | `src/rag/retrievers/hyde.py` |
| クエリ変換 | マルチクエリ（言い換え / 分解） | `src/rag/retrievers/multi_query.py` |
| チャンク | Parent-Child Retriever | `src/rag/retrievers/parent_child.py` |
| 生成 | MMR（多様性付き検索） | `src/rag/retrievers/dense.py` |
| 生成 | LongContextReorder（並び順最適化） | `src/rag/generation.py` |
| Agentic | Corrective RAG（LangGraph） | `src/rag/graph/corrective_rag.py` |
| GraphRAG | LightRAG（ナレッジグラフ構築・5モード検索） | `src/rag/kg/lightrag_store.py` |

## ディレクトリ構成

```
rag-sandbox/
├── src/rag/                     # 共通ライブラリ
│   ├── corpus.py                # コーパス読み込み（.md + .json サイドカー）
│   ├── chunking.py              # チャンク分割（RecursiveCharacterTextSplitter）
│   ├── embeddings.py            # PrefixedEmbeddings（ruri-v3 / bge-m3 対応）
│   ├── store.py                 # FAISS インデックスの構築・保存・読み込み
│   ├── rerank.py                # CrossEncoderReranker（ruri-reranker-large）
│   ├── generation.py            # RAGGenerator（LCEL チェーン + LongContextReorder）
│   ├── metrics.py               # Recall@k / MRR@k / nDCG@k
│   ├── retrievers/              # 各手法。LangChain の BaseRetriever を継承
│   │   ├── dense.py             # Dense Retriever（MMR オプション付き）
│   │   ├── bm25.py              # 日本語 BM25 Retriever（fugashi + rank-bm25）
│   │   ├── hybrid.py            # HybridRetriever（RRF 統合）
│   │   ├── reranked.py          # RerankedRetriever（CrossEncoderReranker のラッパー）
│   │   ├── filtered.py          # TagFilteredRetriever（事後タグフィルタ）
│   │   ├── hyde.py              # HydeRetriever（仮回答 → Dense）
│   │   ├── multi_query.py       # ParaphraseRetriever / DecomposeRetriever
│   │   └── parent_child.py      # build_parent_child_retriever（small-to-big）
│   ├── graph/
│   │   └── corrective_rag.py    # Corrective RAG グラフ（LangGraph）
│   └── kg/
│       └── lightrag_store.py    # LightRAG: グラフ構築・挿入・クエリ（5モード）
│
├── scripts/
│   └── fetch_qiita.py           # Qiita 公開 API から記事を取得して data/corpus/ に保存
│
├── experiments/                 # 実行エントリポイント
│   ├── run_eval.py              # 複数 Retriever を横並び評価（nDCG / MRR / Recall）
│   ├── run_generate.py          # hybrid+rerank で検索 → LLM 生成（--tag フィルタ対応）
│   ├── run_hyde.py              # HyDE と通常 Dense の比較
│   ├── run_parent_child.py      # Parent-Child + Rerank + 生成
│   ├── run_corrective_rag.py    # Corrective RAG（LangGraph ストリーミング表示）
│   └── run_lightrag.py          # LightRAG GraphRAG（5モード横並び比較）
│
├── notebooks/                   # 探索・可視化（コミット対象外）
├── data/                        # コーパス・インデックス（コミット対象外）
│   ├── corpus/                  # <記事ID>.md / <記事ID>.json
│   ├── index/                   # FAISS インデックス
│   ├── lightrag/                # LightRAG グラフ・VDB（コミット対象外）
│   └── qrels.jsonl              # 評価用正解ラベル（クエリ → 正解 doc_id）
└── results/                     # 評価結果出力（コミット対象外）
```

## 環境

- **実行環境**: devcontainer（CUDA 12.6 + Python 3.12）
- **パッケージ管理**: uv
- **LLM**: Ollama（devcontainer 外で起動。`OLLAMA_HOST` 環境変数で接続先を設定）
- **埋め込みモデル**: `cl-nagoya/ruri-v3-310m`
- **Reranker**: `cl-nagoya/ruri-reranker-large`

## 主要ライブラリ

| ライブラリ | 用途 |
|---|---|
| `langchain-core` | 抽象基底クラス（`BaseRetriever`、`Document` など） |
| `langchain-community` | FAISS、`LongContextReorder` など |
| `langchain-ollama` | Ollama LLM 接続 |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langchain-classic` | `ParentDocumentRetriever`（LangChain 1.x 本体には含まれない） |
| `langgraph` | Corrective RAG のグラフ構築 |
| `sentence-transformers` | 埋め込み・Reranker |
| `faiss-cpu` | ベクトルインデックス |
| `fugashi` + `unidic-lite` | 日本語形態素解析（BM25 用） |
| `rank-bm25` | BM25 スコアリング |
| `ollama` | Ollama Python クライアント |
| `lightrag-hku` | LightRAG GraphRAG（ナレッジグラフ構築・検索） |
| `networkx` | グラフデータ構造（LightRAG の NetworkXStorage） |
