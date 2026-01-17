# **this_is_a_m_6**

> fill chapter introduction here

### **This chapter can be separated into the following sections:**

{% for p in page.parent.children %}
{% if (p.is_section or p.is_page) and p.url and p.url != page.url %}
1. [{{ p.title }}]({{ '/' ~ p.url.lstrip('/') }})
{% elif p.is_section and p.children and p.children[0].url %}
1. [{{ p.title }}]({{ '/' ~ p.children[0].url.lstrip('/') }})
{% endif %}
{% endfor %}
