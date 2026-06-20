# rag-sandbox

RAG 技術を試すサンドボックス。各手法（ベクトル検索 / キーワード検索 / ハイブリッド / Reranker / HyDE / 能動的検索など）を手軽に試しつつ、固まった処理は再利用できる形に育てていく。完全ローカル / OSS を基本とし、日本語コーパスを前提とする。

## 環境・依存管理

- 実行環境は devcontainer（CUDA 12.6 + Python 3.12）。`requires-python = ">=3.12,<3.13"`。
- パッケージ管理は **uv**。`pip install` は使わず、依存の増減は次の手順で行う:
  - 追加: `uv add <pkg>`（dev 依存は `uv add --dev <pkg>`）
  - 削除: `uv remove <pkg>`
  - これらは `pyproject.toml` と `uv.lock` を自動更新する。手で `uv.lock` を編集しない。
- `uv.lock` は **コミット対象**。再現性のため依存を変えたら必ず lock も一緒にコミットする。
- torch は CUDA 12.6 wheel を `pytorch-cu126` index から取得する設定（`[tool.uv.sources]`）。バージョン変更時はこの制約を壊さない。
- 依存解決は **linux + Python 3.12 に限定**（`[tool.uv] environments`）。CUDA wheel の都合なので、他プラットフォーム向けに環境を広げない。

### 主要ライブラリと役割

- Embedding / Reranker: `transformers` / `sentence-transformers` / `FlagEmbedding`（ruri-v3 / bge-m3 / 日本語 Reranker など）
- 日本語キーワード検索: `fugashi` + `unidic-lite`（形態素解析）→ `rank-bm25`
- ベクトル検索: `faiss-cpu`
- 生成: `ollama`（生成および HyDE の仮回答生成に利用。Ollama はコンテナ外/別サービスなので、接続先は env で設定可能にする）
- 能動的検索: `langgraph` / `langchain`（Agentic RAG / Self-RAG 系）

## ディレクトリ構成

`PYTHONPATH=/workspace` が通っているので、パッケージは `from src.rag... import` で参照する。

- `src/rag/` — 共通処理。`corpus` / `chunking` / `embeddings` / `store` / `rerank` / `metrics`。
  - `src/rag/retrievers/` — 各検索手法。LangChain の `BaseRetriever` を継承し、`invoke()` で共通的に呼び出せる。新手法は1ファイル足すだけで評価に乗る。Reranker / HyDE は他 Retriever をラップする型。
- `scripts/` — 一回実行系のスクリプト。`fetch_qiita.py`（Qiita 公開 API → `data/corpus/`）。
- `experiments/` — 実験のエントリポイント。`run_eval.py`（複数 Retriever を横並び評価）。
- `notebooks/` — 探索・可視化用の `.ipynb`。試行錯誤はここで自由に。
- `data/` `outputs/` `results/` — 入出力。すべて gitignore（コミットしない）。コーパスは `data/corpus/`、FAISS 等の成果物は `data/index/`。

橋渡しルール: Notebook で探索 → 安定したら `src/rag/` の関数/クラスへ抽出 → `experiments/` のスクリプトから呼ぶ。Notebook に長いロジックを残し続けない。

評価データ `data/qrels.jsonl` は1行 `{"query": "...", "positive_doc_ids": ["doc_id", ...]}`。doc_id は `data/corpus/` のファイル名（拡張子なし）。

## コーディング・Lint

- Lint/format は **ruff**（`line-length=100`, `target=py312`, ルール: `E,F,I,B,UP`、`E501` は無視）。
- コミット前に `ruff check --fix .` と `ruff format .` を通す。
- データ列名やマジックナンバーなど、後で読んで非自明な箇所のみコメントを残す。

## Git・コミット

- コミットメッセージ: **タイトル1行のみ・英語の命令形・〜72文字・本文なし・`Co-Authored-By` フッターなし**。例: `Add dense retriever baseline`, `Fix BM25 tokenization`。
- 1コミット = 1論理変更。`git add -A` ではなく対象ファイルを明示的に stage する。
- ブランチ運用: `feature/<topic>` を切って作業し、PR 経由でレビュー後 `main` へマージする。`main` への直接コミットはしない。
- `.env` や `data/`・モデルキャッシュ・FAISS インデックスなど gitignore 対象は絶対にコミットしない。

## GitHub 操作（Issue / PR）

- Issue / PR の作成・更新は `gh` から行える（リモート `origin` = `atsushi11o7/rag-sandbox`、`gh` 認証済み）。
  - Issue: `gh issue create --title "..." --body "..."`
  - PR: ブランチを push 後 `gh pr create --base main --title "..." --body "..."`
- トークンは絶対にコミットしない（環境変数 / `gh auth login` で管理）。

## 秘密情報の扱い

- `.env` は秘密情報。**明示的な許可がない限り閲覧・編集しない**。トークンや API キーの値はユーザー自身が設定する。
- 新しい環境変数を足すときは雛形の `.env.example` 側を更新する。
