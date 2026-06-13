---
layout: default
title: "Posts"
permalink: /en/
lang: en
---

# Posts

<p class="lang-switch"><a href="{{ '/ja/' | relative_url }}">日本語 »</a></p>

<ul class="post-list">
{% for post in site.posts %}{% if post.lang == 'en' %}
  <li>
    <span class="post-list-date">{{ post.date | date: "%Y-%m-%d" }}</span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
{% endif %}{% endfor %}
</ul>
