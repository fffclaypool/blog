---
layout: default
title: ""
---

# Posts / 記事

<ul class="post-list">
{% for post in site.posts %}
  <li>
    <span class="post-list-date">{{ post.date | date: "%Y-%m-%d" }}</span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    {% if post.lang %}<span class="lang-badge">{{ post.lang | upcase }}</span>{% endif %}
  </li>
{% endfor %}
</ul>
