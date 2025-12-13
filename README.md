# MkDocs Nav Organizer (Page Tree)

当前使用 `page_tree_app/`（Confluence 风格页面树）来编辑 `mkdocs.yml` 的 `nav`，并可选同步移动 `doc/` 或 `docs/` 下的页面文件。

## Run

`pip install -r requirements.txt`

`python page_tree_app/app.py`

打开：`http://127.0.0.1:5000`

逻辑结构（Page Tree）：标题、父子关系、排序，像 Confluence 一样以“页面”为中心；移动页面不必等同于立刻重命名/重排磁盘目录。

物理结构（Files/Dirs）：docs/ 下的 .md 文件与资源；可以“相对同步”，但要有明确的同步规则/开关。
