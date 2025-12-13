# Page Tree App (Confluence-like MkDocs Nav)

目标：用“页面树（逻辑结构）”来编辑 MkDocs `nav`，并且**可选**把对应的 `.md` 文件/资源在磁盘上同步移动。

## 运行

1. 安装依赖（复用仓库根目录的 `requirements.txt`）：

   `pip install -r requirements.txt`

2. 启动：

   `python page_tree_app/app.py`

3. 打开：`http://127.0.0.1:5000`

## 约定

- 默认读取/写入：
  - `mkdocs.yml`：仓库根目录 `mkdocs.yml`（可用 `MKDOCS_PATH` 覆盖）
  - `docs_root`：优先读 `mkdocs.yml` 的 `docs_dir`，否则自动探测 `docs/` 或 `doc/`
  - 状态文件：仓库根目录 `.page_tree_state.json`（可用 `STATE_PATH` 覆盖）
- 备份文件：
  - 保存 `mkdocs.yml` 前会在 `backups/mkdocs/` 下创建备份
  - 仅保留最近 5 个 `mkdocs.yml.bak-*`
- “同步模式”
  - `Sync Nav Only`：只写 `mkdocs.yml` 的 `nav`（不移动磁盘文件）
  - `Sync + Move Files`：按页面树把页面 `.md` 文件移动到对应目录，并同步移动同名资源目录（`page.md` -> `page/`）
  - 同步时会使用当前 UI 的树（无需先点 `Save Tree`，但 `Save Tree` 可以单独保存草稿）
