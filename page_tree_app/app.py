from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable
from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

yaml = YAML()
yaml.preserve_quotes = True

ALLOWED_DOC_EXTS = {".md", ".markdown", ".mdx"}


def _json_error(message: str, status: int = 400, **extra: Any):
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return jsonify(payload), status


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir = REPO_ROOT / "backups" / "mkdocs"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.name}.bak-{_now_stamp()}"
    copy2(path, backup_path)

    # Keep only the most recent backups.
    keep = 5
    backups = sorted(
        backup_dir.glob(f"{path.name}.bak-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in backups[keep:]:
        try:
            old.unlink()
        except OSError:
            pass

    return backup_path


def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "section"


def _normalize_rel(path: str) -> str:
    cleaned = (path or "").strip().replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned).lstrip("./")
    return cleaned


def _safe_rel_path(rel: str) -> str:
    rel = _normalize_rel(rel)
    if not rel or rel.startswith("/") or rel.startswith("~"):
        raise ValueError("Invalid relative path.")
    parts = Path(rel).parts
    if any(part in ("..", ".") for part in parts):
        raise ValueError("Invalid relative path.")
    return rel


def _mkdocs_path() -> Path:
    env = os.environ.get("MKDOCS_PATH") or os.environ.get("MKDOCS_FILE")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (REPO_ROOT / p)
    return REPO_ROOT / "mkdocs.yml"


def _state_path() -> Path:
    env = os.environ.get("STATE_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (REPO_ROOT / p)
    return REPO_ROOT / ".page_tree_state.json"


def _load_mkdocs_config(mkdocs: Path) -> CommentedMap:
    if not mkdocs.exists():
        return CommentedMap()
    with mkdocs.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle) or CommentedMap()
    if not isinstance(data, CommentedMap):
        raise TypeError("mkdocs.yml must contain a mapping at the top level.")
    return data


def _save_mkdocs_config(mkdocs: Path, config: CommentedMap) -> Path | None:
    backup = _backup_file(mkdocs)
    mkdocs.parent.mkdir(parents=True, exist_ok=True)
    with mkdocs.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle)
    return backup


def _docs_root(mkdocs: Path) -> Path:
    env = os.environ.get("DOCS_ROOT")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (REPO_ROOT / p)
    try:
        config = _load_mkdocs_config(mkdocs)
        docs_dir = config.get("docs_dir")
        if isinstance(docs_dir, str) and docs_dir.strip():
            p = Path(docs_dir.strip())
            return p if p.is_absolute() else (mkdocs.parent / p)
    except Exception:
        pass
    for candidate in ("docs", "doc"):
        guess = REPO_ROOT / candidate
        if guess.exists():
            return guess.resolve()
    return REPO_ROOT


MKDOCS_PATH = _mkdocs_path()
STATE_PATH = _state_path()
DOCS_ROOT = _docs_root(MKDOCS_PATH)


def _new_id() -> str:
    return uuid4().hex


@dataclass
class Node:
    id: str
    type: str  # "folder" | "page"
    title: str
    segment: str | None = None  # for folder, stable physical dir name
    file: str | None = None  # for page, relative path under DOCS_ROOT
    children: list["Node"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "segment": self.segment,
            "file": self.file,
            "children": [c.to_dict() for c in (self.children or [])],
        }


def _node_from_dict(data: dict[str, Any]) -> Node:
    node_type = str(data.get("type") or "").strip()
    if node_type not in {"folder", "page"}:
        raise ValueError("Invalid node type.")
    title = str(data.get("title") or "").strip()
    if not title:
        raise ValueError("Title is required.")
    node_id = str(data.get("id") or "").strip() or _new_id()
    segment = data.get("segment")
    file = data.get("file")
    if segment not in (None, "") and not isinstance(segment, str):
        raise ValueError("Invalid segment.")
    if file not in (None, "") and not isinstance(file, str):
        raise ValueError("Invalid file.")

    children_raw = data.get("children") or []
    if children_raw and not isinstance(children_raw, list):
        raise ValueError("Invalid children.")
    children = [_node_from_dict(item) for item in (children_raw or [])]

    if node_type == "folder":
        seg = str(segment).strip() if isinstance(segment, str) and segment.strip() else _slugify(title)
        return Node(id=node_id, type="folder", title=title, segment=seg, file=None, children=children)

    file_str = str(file).strip() if isinstance(file, str) and file.strip() else None
    if file_str:
        file_str = _safe_rel_path(file_str)
    if children:
        raise ValueError("Page node cannot have children.")
    return Node(id=node_id, type="page", title=title, segment=None, file=file_str, children=[])


def _flatten_pages(nodes: Iterable[Node]) -> list[Node]:
    pages: list[Node] = []
    for node in nodes:
        if node.type == "page":
            pages.append(node)
        else:
            pages.extend(_flatten_pages(node.children or []))
    return pages


def _infer_segment_from_children(folder: Node, parent_prefix: str) -> str | None:
    candidates: list[str] = []

    def walk(items: Iterable[Node]):
        for n in items:
            if n.type == "page" and n.file:
                rel = _normalize_rel(n.file)
                remainder = rel
                if parent_prefix:
                    prefix = f"{parent_prefix}/"
                    if remainder.startswith(prefix):
                        remainder = remainder[len(prefix) :]
                parts = remainder.split("/")
                if parts and parts[0]:
                    candidates.append(parts[0])
            if n.type == "folder":
                walk(n.children or [])

    walk(folder.children or [])
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def _infer_segments_in_place(nodes: list[Node], parent_prefix: str = "") -> None:
    """Best-effort: keep folder `segment` aligned with existing disk paths on import.

    Only used for importing/bootstrapping. Regular editing keeps `segment` stable.
    """

    for node in nodes:
        if node.type != "folder":
            continue
        inferred = _infer_segment_from_children(node, parent_prefix)
        if inferred:
            node.segment = inferred
            next_prefix = f"{parent_prefix}/{inferred}" if parent_prefix else inferred
        else:
            node.segment = node.segment or _slugify(node.title)
            next_prefix = parent_prefix
        _infer_segments_in_place(node.children or [], next_prefix)


def _import_mkdocs_nav(nav_items: Any, parent_prefix: str = "") -> list[Node]:
    nodes: list[Node] = []
    if not isinstance(nav_items, (list, CommentedSeq)):
        return nodes
    for entry in nav_items:
        if isinstance(entry, str):
            rel = _safe_rel_path(entry)
            nodes.append(Node(id=_new_id(), type="page", title=entry, file=rel, children=[]))
            continue

        if isinstance(entry, (dict, CommentedMap)):
            items = list(entry.items())
            if not items:
                continue
            title, value = items[0]
            title_str = str(title).strip() or "untitled"

            if isinstance(value, (list, CommentedSeq)):
                folder = Node(
                    id=_new_id(),
                    type="folder",
                    title=title_str,
                    segment=_slugify(title_str),
                    children=_import_mkdocs_nav(value, parent_prefix),
                )
                nodes.append(folder)
                continue

            if value is None:
                folder = Node(id=_new_id(), type="folder", title=title_str, segment=_slugify(title_str), children=[])
                nodes.append(folder)
                continue

            rel = _safe_rel_path(str(value))
            nodes.append(Node(id=_new_id(), type="page", title=title_str, file=rel, children=[]))
            continue

        nodes.append(Node(id=_new_id(), type="page", title=str(entry), file=None, children=[]))

    return nodes


def _tree_to_mkdocs_nav(nodes: Iterable[Node]) -> CommentedSeq:
    nav = CommentedSeq()
    for node in nodes:
        title = (node.title or "").strip() or "untitled"
        if node.type == "folder":
            nav.append(CommentedMap({title: _tree_to_mkdocs_nav(node.children or [])}))
            continue
        if not node.file:
            nav.append(CommentedMap({title: None}))
            continue
        title_out = title
        # If the title has degraded into a path (common after auto-generated nav),
        # restore a sensible display name by using the basename.
        if title_out == node.file or "/" in title_out or "\\" in title_out:
            title_out = Path(node.file).name
        nav.append(CommentedMap({title_out: node.file}))
    return nav


def _build_source_tree(current: Path, base: Path) -> list[dict[str, Any]]:
    if not current.exists():
        return []
    try:
        children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return []

    entries: list[dict[str, Any]] = []
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append(
                {
                    "name": child.name,
                    "path": child.relative_to(base).as_posix(),
                    "kind": "dir",
                    "children": _build_source_tree(child, base),
                }
            )
            continue
        if child.suffix.lower() not in ALLOWED_DOC_EXTS:
            continue
        entries.append(
            {
                "name": child.name,
                "path": child.relative_to(base).as_posix(),
                "kind": "file",
                "children": [],
            }
        )
    return entries


def _load_state() -> list[Node]:
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8") or "{}")
        tree = raw.get("tree") if isinstance(raw, dict) else None
        if isinstance(tree, list):
            return [_node_from_dict(item) for item in tree]
    config = _load_mkdocs_config(MKDOCS_PATH)
    imported = _import_mkdocs_nav(config.get("nav", CommentedSeq()))
    _infer_segments_in_place(imported)
    return imported


def _save_state(nodes: list[Node]) -> None:
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tree": [n.to_dict() for n in nodes],
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_unique_targets(moves: list[tuple[Path, Path]]) -> None:
    seen: set[Path] = set()
    dupes: set[Path] = set()
    for _, dst in moves:
        if dst in seen:
            dupes.add(dst)
        seen.add(dst)
    if dupes:
        raise ValueError(f"Duplicate move targets: {', '.join(sorted(str(p) for p in dupes))}")


def _cleanup_empty_dirs(start_dirs: Iterable[Path], stop_at: Path) -> None:
    to_check: set[Path] = set()
    for d in start_dirs:
        current = d
        while stop_at in current.parents and current != stop_at:
            to_check.add(current)
            current = current.parent
    for directory in sorted(to_check, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue


def _compute_folder_dir(ancestors: list[Node]) -> str:
    segments = [a.segment for a in ancestors if a.type == "folder" and a.segment]
    return "/".join(segments)


def _plan_file_sync(
    nodes: list[Node], *, create_missing: bool
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[dict[str, Any]]]:
    page_moves: list[tuple[Path, Path]] = []
    asset_moves: list[tuple[Path, Path]] = []
    warnings: list[dict[str, Any]] = []

    def walk(items: list[Node], ancestors: list[Node]):
        folder_dir = _compute_folder_dir(ancestors)
        for node in items:
            if node.type == "folder":
                walk(node.children or [], ancestors + [node])
                continue
            basename = Path(node.file).name if node.file else f"{_slugify(node.title)}.md"
            desired_rel = f"{folder_dir}/{basename}" if folder_dir else basename
            desired_rel = _safe_rel_path(desired_rel)
            current_rel = _safe_rel_path(node.file) if node.file else desired_rel
            src = DOCS_ROOT / current_rel
            dst = DOCS_ROOT / desired_rel
            if not src.exists():
                warnings.append({"type": "missing_file", "file": current_rel, "title": node.title})
                if create_missing:
                    _create_missing_page(desired_rel, node.title)
                    node.file = desired_rel
                continue

            if src != dst:
                page_moves.append((src, dst))
                # Confluence-like "relative sync": also move a sibling asset folder named after the stem.
                src_assets = src.with_suffix("")  # page.md -> page
                dst_assets = dst.with_suffix("")
                if src_assets.exists() and src_assets.is_dir():
                    asset_moves.append((src_assets, dst_assets))
            node.file = desired_rel

    walk(nodes, [])
    _ensure_unique_targets(page_moves + asset_moves)
    return page_moves, asset_moves, warnings


def _apply_moves(moves: list[tuple[Path, Path]]) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    sources = []
    for src, dst in moves:
        sources.append(src.parent)
        if not src.exists():
            raise FileNotFoundError(str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(str(dst))
        src.rename(dst)
        applied.append({"from": src.relative_to(DOCS_ROOT).as_posix(), "to": dst.relative_to(DOCS_ROOT).as_posix()})
    _cleanup_empty_dirs(sources, DOCS_ROOT)
    return applied


def _create_missing_page(file_rel: str, title: str) -> None:
    file_rel = _safe_rel_path(file_rel)
    path = DOCS_ROOT / file_rel
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title.strip() or 'Untitled'}\n", encoding="utf-8")


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/meta", methods=["GET"])
def api_meta():
    return jsonify(
        {
            "mkdocs_path": str(MKDOCS_PATH),
            "docs_root": str(DOCS_ROOT),
            "state_path": str(STATE_PATH),
        }
    )


@app.route("/api/state", methods=["GET"])
def api_get_state():
    tree = _load_state()
    return jsonify({"tree": [n.to_dict() for n in tree]})


@app.route("/api/state", methods=["POST"])
def api_save_state():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    tree_raw = payload.get("tree")
    if not isinstance(tree_raw, list):
        return _json_error("Expected `tree` to be a JSON list.", 400)
    try:
        nodes = [_node_from_dict(item) for item in tree_raw]
    except Exception as exc:
        return _json_error(f"Invalid tree: {exc}", 400)
    _save_state(nodes)
    return jsonify({"status": "ok"})


@app.route("/api/import", methods=["POST"])
def api_import():
    try:
        config = _load_mkdocs_config(MKDOCS_PATH)
    except Exception as exc:
        return _json_error(str(exc), 500)
    nodes = _import_mkdocs_nav(config.get("nav", CommentedSeq()))
    _infer_segments_in_place(nodes)
    _save_state(nodes)
    return jsonify({"status": "ok", "tree": [n.to_dict() for n in nodes]})


@app.route("/api/source", methods=["GET"])
def api_source():
    return jsonify(_build_source_tree(DOCS_ROOT, DOCS_ROOT))


@app.route("/api/sync", methods=["POST"])
def api_sync():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)

    mode = payload.get("mode") or "nav_only"
    if mode not in {"nav_only", "sync_files"}:
        return _json_error("Invalid mode.", 400)
    create_missing = bool(payload.get("create_missing", True))

    try:
        if isinstance(payload.get("tree"), list):
            nodes = [_node_from_dict(item) for item in payload["tree"]]
            _save_state(nodes)
        else:
            nodes = _load_state()
    except Exception as exc:
        return _json_error(f"Failed to load state: {exc}", 500)

    warnings: list[dict[str, Any]] = []
    moves_applied: list[dict[str, str]] = []

    if mode == "sync_files":
        page_moves, asset_moves, plan_warnings = _plan_file_sync(nodes, create_missing=create_missing)
        warnings.extend(plan_warnings)
        try:
            moves_applied.extend(_apply_moves(page_moves))
            moves_applied.extend(_apply_moves(asset_moves))
        except (FileNotFoundError, FileExistsError) as exc:
            return _json_error(f"File move failed: {exc}", 400)
        except Exception as exc:
            return _json_error(f"File move failed: {exc}", 500)

        _save_state(nodes)

    # Always write mkdocs.yml nav from current state (nav_only uses existing page.file)
    try:
        config = _load_mkdocs_config(MKDOCS_PATH)
    except Exception as exc:
        return _json_error(f"Failed to read mkdocs.yml: {exc}", 500)
    config["nav"] = _tree_to_mkdocs_nav(nodes)
    try:
        backup = _save_mkdocs_config(MKDOCS_PATH, config)
    except Exception as exc:
        return _json_error(f"Failed to write mkdocs.yml: {exc}", 500)

    return jsonify(
        {
            "status": "ok",
            "mode": mode,
            "moves": moves_applied,
            "warnings": warnings,
            "backup": str(backup) if backup else None,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
