---
layout: default
title: "記事一覧"
permalink: /ja/
lang: ja
---

# 記事一覧

<p class="lang-switch"><a href="{{ '/en/' | relative_url }}">English »</a></p>

<ul class="post-list">
{% for post in site.posts %}{% if post.lang == 'ja' %}
  <li>
    <span class="post-list-date">{{ post.date | date: "%Y-%m-%d" }}</span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
{% endif %}{% endfor %}
</ul>
