"""Qiita 公開 API でユーザーの記事を取得し data/corpus/ に Markdown で保存する。"""

import json
import os
import time
import urllib.request
from pathlib import Path

USER = os.environ.get("QIITA_USER", "atsushi11o7")
OUT_DIR = Path("data/corpus")
API = "https://qiita.com/api/v2/users/{user}/items?per_page=100&page={page}"


def fetch_items(user: str) -> list[dict]:
    """公開記事をページングで全件取得する。

    Args:
        user: Qiita のユーザー ID。

    Returns:
        記事情報の辞書リスト。各要素に id / title / body などが含まれる。
    """
    items: list[dict] = []
    page = 1
    while True:
        url = API.format(user=user, page=page)
        with urllib.request.urlopen(url) as response:
            batch = json.load(response)
        if not batch:
            break
        items.extend(batch)
        page += 1
        time.sleep(1)
    return items


def main() -> None:
    """記事を取得して data/corpus/<記事ID>.md として保存する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = fetch_items(USER)
    for item in items:
        path = OUT_DIR / f"{item['id']}.md"
        path.write_text(f"# {item['title']}\n\n{item['body']}", encoding="utf-8")
    print(f"saved {len(items)} articles to {OUT_DIR}")


if __name__ == "__main__":
    main()
