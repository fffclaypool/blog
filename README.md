# fffclaypool tech notes

A technical blog built with GitHub Pages + Jekyll. Live at: <https://fffclaypool.github.io/blog/>

## Writing a post

Just add `YYYY-MM-DD-slug.md` under `_posts/`, with front matter at the top:

```yaml
---
layout: post
title: "Post title"
description: "One-line summary (optional)"
date: 2026-06-13
---
```

Running `git push` makes GitHub Pages build and publish automatically.

## Writing math (important)

**Always wrap math in `$$...$$` (double dollars)** — for both inline and block math.

- Inline: `… the score $$F_1 = \frac{2PR}{P+R}$$ in the text …`
- Block: put `$$` on its own line, with a blank line before and after.

Why: GitHub Pages' Markdown converter, kramdown, turns `$$...$$` into MathJax's
`\(...\)` / `\[...\]`. A single `$...$` is **not** treated as math.

### Watch out for vertical bars

For absolute values / set cardinality, use **`\lvert ... \rvert`** instead of `|`.
A raw `|` collides with GFM's table separator and breaks the math.

- ✗ `$$|R \cap \mathrm{Rel}|$$`
- ○ `$$\lvert R \cap \mathrm{Rel} \rvert$$`

## Local preview (optional)

The site is built on GitHub, so no local setup is required. Only if you want to preview:

```sh
bundle install
bundle exec jekyll serve
```

(Ruby 3.x recommended; put `gem "github-pages", group: :jekyll_plugins` in the `Gemfile`.)
