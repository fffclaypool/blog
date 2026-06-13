# fffclaypool tech notes

GitHub Pages + Jekyll の技術ブログ。公開先: <https://fffclaypool.github.io/blog/>

## 記事の書き方

`_posts/` に `YYYY-MM-DD-slug.md` を追加するだけ。先頭に front matter:

```yaml
---
layout: post
title: "記事タイトル"
description: "一行説明（任意）"
date: 2026-06-13
---
```

`git push` すると GitHub Pages が自動でビルド・公開する。

## 数式の書き方（重要）

数式は **必ず `$$...$$`（二重ドル）で囲む**。インラインもブロックも同じ。

- インライン: `文中の $$F_1 = \frac{2PR}{P+R}$$ は…`
- ブロック: `$$` を前後の空行つきで独立行に置く

理由: GitHub Pages の Markdown 変換器 kramdown は `$$...$$` を MathJax の
`\(...\)` / `\[...\]` に変換する。単独の `$...$` は数式扱いされない。

### 縦棒に注意

絶対値・集合濃度の縦棒は `|` ではなく **`\lvert ... \rvert`** を使う。
`|` は GFM の表区切りと衝突して数式が壊れるため。

- ✗ `$$|R \cap \mathrm{Rel}|$$`
- ○ `$$\lvert R \cap \mathrm{Rel} \rvert$$`

## ローカルプレビュー（任意）

GitHub 上でビルドされるのでローカル環境は不要。確認したい場合のみ:

```sh
bundle install
bundle exec jekyll serve
```

（Ruby 3.x 推奨。`Gemfile` に `gem "github-pages", group: :jekyll_plugins` を置く）
