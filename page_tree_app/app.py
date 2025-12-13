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
PROJECT_ROOT = BASE_DIR.parent

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
    backup_dir = BASE_DIR / "backups" / "mkdocs"
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
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return PROJECT_ROOT / "mkdocs.yml"


def _state_path() -> Path:
    env = os.environ.get("STATE_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    # Keep the draft inside the app folder so users can drop `page_tree_app/` into any project.
    return BASE_DIR / ".page_tree_state.json"


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
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    try:
        config = _load_mkdocs_config(mkdocs)
        docs_dir = config.get("docs_dir")
        if isinstance(docs_dir, str) and docs_dir.strip():
            p = Path(docs_dir.strip())
            return p if p.is_absolute() else (mkdocs.parent / p)
    except Exception:
        pass
    for candidate in ("docs", "doc"):
        guess = PROJECT_ROOT / candidate
        if guess.exists():
            return guess.resolve()
    return PROJECT_ROOT


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
    file_prev: str | None = None  # for page, previous relative path (rename support)
    children: list["Node"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "segment": self.segment,
            "file": self.file,
            "file_prev": self.file_prev,
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
    file_prev = data.get("file_prev")
    if segment not in (None, "") and not isinstance(segment, str):
        raise ValueError("Invalid segment.")
    if file not in (None, "") and not isinstance(file, str):
        raise ValueError("Invalid file.")
    if file_prev not in (None, "") and not isinstance(file_prev, str):
        raise ValueError("Invalid file_prev.")

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
    prev_str = str(file_prev).strip() if isinstance(file_prev, str) and file_prev.strip() else None
    if prev_str:
        prev_str = _safe_rel_path(prev_str)
    if children:
        raise ValueError("Page node cannot have children.")
    return Node(id=node_id, type="page", title=title, segment=None, file=file_str, file_prev=prev_str, children=[])


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


def _collect_desired_folder_dirs(nodes: list["Node"]) -> set[str]:
    desired: set[str] = set()

    def walk(items: list["Node"], ancestors: list["Node"]):
        for node in items:
            if node.type != "folder":
                continue
            segs = [a.segment for a in (ancestors + [node]) if a.type == "folder" and a.segment]
            rel = "/".join(segs)
            if rel:
                desired.add(rel)
            walk(node.children or [], ancestors + [node])

    walk(nodes, [])
    return desired


def _ensure_desired_folder_dirs_exist(desired_dirs: set[str]) -> None:
    for rel in sorted(desired_dirs, key=lambda s: len(s.split("/"))):
        try:
            (_safe_docs_path(rel)).mkdir(parents=True, exist_ok=True)
        except Exception:
            continue


def _cleanup_stray_empty_dirs(root: Path, *, keep_rel_dirs: set[str]) -> None:
    root = root.resolve()
    for directory in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            rel = directory.relative_to(root).as_posix()
        except Exception:
            continue
        if not rel or rel in keep_rel_dirs:
            continue
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
            # If a user renamed a file in the editor, prefer the previous path as the source for moving.
            current_rel = _safe_rel_path(node.file_prev) if node.file_prev else (_safe_rel_path(node.file) if node.file else desired_rel)
            src = DOCS_ROOT / current_rel
            dst = DOCS_ROOT / desired_rel
            if not src.exists():
                warnings.append({"type": "missing_file", "file": current_rel, "title": node.title})
                if create_missing:
                    _create_missing_page(desired_rel, node.title)
                    node.file = desired_rel
                    node.file_prev = None
                continue

            if src != dst:
                page_moves.append((src, dst))
                # Confluence-like "relative sync": also move a sibling asset folder named after the stem.
                src_assets = src.with_suffix("")  # page.md -> page
                dst_assets = dst.with_suffix("")
                if src_assets.exists() and src_assets.is_dir():
                    asset_moves.append((src_assets, dst_assets))
            node.file = desired_rel
            node.file_prev = None

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


def _safe_docs_path(rel: str) -> Path:
    rel = _safe_rel_path(rel)
    path = (DOCS_ROOT / rel).resolve()
    docs = DOCS_ROOT.resolve()
    if docs not in path.parents and path != docs:
        raise ValueError("Path escapes docs root.")
    return path


@app.route("/api/rename_file", methods=["POST"])
def api_rename_file():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    old_file = str(payload.get("old_file") or "").strip()
    new_basename = str(payload.get("new_basename") or "").strip()
    if not old_file:
        return _json_error("`old_file` is required.", 400)
    if not new_basename:
        return _json_error("`new_basename` is required.", 400)
    if "/" in new_basename or "\\" in new_basename:
        return _json_error("Filename must not include directories.", 400)

    old_rel = _safe_rel_path(old_file)
    try:
        old_path = _safe_docs_path(old_rel)
    except Exception as exc:
        return _json_error(str(exc), 400)
    if not old_path.exists() or not old_path.is_file():
        return _json_error("Old file not found.", 404)

    # Keep extension unless user specifies one.
    base = new_basename
    if "." not in Path(base).name:
        base = f"{base}{old_path.suffix}"
    if old_path.suffix.lower() == ".md" and not base.lower().endswith(".md"):
        base = f"{base}.md"

    new_path = (old_path.parent / base).resolve()
    docs = DOCS_ROOT.resolve()
    if docs not in new_path.parents and new_path != docs:
        return _json_error("Path escapes docs root.", 400)
    if new_path.exists():
        return _json_error("Target already exists.", 409)

    try:
        old_path.rename(new_path)
    except Exception as exc:
        return _json_error(f"Rename failed: {exc}", 500)

    return jsonify({"status": "ok", "file": new_path.relative_to(DOCS_ROOT).as_posix()})


@app.route("/api/rename_dir", methods=["POST"])
def api_rename_dir():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    dir_rel = str(payload.get("dir") or "").strip().strip("/")
    new_segment = str(payload.get("new_segment") or "").strip().strip("/")
    if not dir_rel:
        return _json_error("`dir` is required.", 400)
    if not new_segment:
        return _json_error("`new_segment` is required.", 400)
    if "/" in new_segment or "\\" in new_segment:
        return _json_error("Invalid segment.", 400)

    old_rel = _safe_rel_path(dir_rel)
    try:
        old_path = _safe_docs_path(old_rel)
    except Exception as exc:
        return _json_error(str(exc), 400)
    if not old_path.exists() or not old_path.is_dir():
        return _json_error("Directory not found.", 404)

    parent = old_path.parent
    new_path = (parent / new_segment).resolve()
    docs = DOCS_ROOT.resolve()
    if docs not in new_path.parents and new_path != docs:
        return _json_error("Path escapes docs root.", 400)
    if new_path.exists():
        return _json_error("Target already exists.", 409)

    try:
        old_path.rename(new_path)
    except Exception as exc:
        return _json_error(f"Rename failed: {exc}", 500)

    return jsonify(
        {
            "status": "ok",
            "segment": new_segment,
            "dir": new_path.relative_to(DOCS_ROOT).as_posix(),
        }
    )


@app.route("/api/delete_files", methods=["POST"])
def api_delete_files():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    files = payload.get("files")
    if not isinstance(files, list):
        return _json_error("Expected `files` to be a JSON list.", 400)
    delete_assets = bool(payload.get("delete_assets", False))

    deleted: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for item in files:
        if not isinstance(item, str) or not item.strip():
            continue
        rel = _safe_rel_path(item.strip())
        try:
            path = _safe_docs_path(rel)
        except Exception as exc:
            errors.append({"file": rel, "error": str(exc)})
            continue

        if path.exists() and path.is_file():
            try:
                path.unlink()
                deleted.append({"kind": "file", "path": rel})
            except Exception as exc:
                errors.append({"file": rel, "error": str(exc)})

        if delete_assets:
            asset_dir = path.with_suffix("")
            if asset_dir.exists() and asset_dir.is_dir():
                try:
                    for child in asset_dir.rglob("*"):
                        if child.is_file():
                            child.unlink()
                    for d in sorted([p for p in asset_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
                        try:
                            d.rmdir()
                        except OSError:
                            pass
                    asset_dir.rmdir()
                    deleted.append({"kind": "asset_dir", "path": asset_dir.relative_to(DOCS_ROOT).as_posix()})
                except Exception as exc:
                    errors.append({"file": rel, "error": f"asset_dir: {exc}"})

    return jsonify({"status": "ok", "deleted": deleted, "errors": errors})


def _unique_child_name(parent: Path, name: str) -> str:
    base = re.sub(r"[/\\\\]+", "-", name).strip()
    if not base:
        base = "untitled"
    candidate = base
    i = 2
    while (parent / candidate).exists():
        candidate = f"{base}-{i}"
        i += 1
    return candidate


@app.route("/api/create_section", methods=["POST"])
def api_create_section():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    title = str(payload.get("title") or "").strip()
    if not title:
        return _json_error("Title is required.", 400)
    parent_dir = str(payload.get("parent_dir") or "").strip().strip("/")
    parent_path = DOCS_ROOT if not parent_dir else _safe_docs_path(parent_dir)
    if parent_path.exists() and not parent_path.is_dir():
        return _json_error("Parent path is not a directory.", 400)
    parent_path.mkdir(parents=True, exist_ok=True)

    desired = str(payload.get("segment") or "").strip()
    if desired and ("/" in desired or "\\" in desired):
        return _json_error("Invalid segment.", 400)
    segment = _slugify(desired or title)
    segment = _unique_child_name(parent_path, segment)
    created = parent_path / segment
    created.mkdir(parents=True, exist_ok=False)
    rel = created.relative_to(DOCS_ROOT).as_posix()
    return jsonify({"status": "ok", "segment": segment, "dir": rel})


@app.route("/api/create_page", methods=["POST"])
def api_create_page():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    title = str(payload.get("title") or "").strip()
    if not title:
        return _json_error("Title is required.", 400)
    parent_dir = str(payload.get("parent_dir") or "").strip().strip("/")
    parent_path = DOCS_ROOT if not parent_dir else _safe_docs_path(parent_dir)
    if parent_path.exists() and not parent_path.is_dir():
        return _json_error("Parent path is not a directory.", 400)
    parent_path.mkdir(parents=True, exist_ok=True)

    desired = str(payload.get("basename") or "").strip()
    if desired and ("/" in desired or "\\" in desired):
        return _json_error("Invalid filename.", 400)
    base = desired or f"{_slugify(title)}.md"
    if not base.lower().endswith(".md"):
        base = f"{base}.md"
    base = _unique_child_name(parent_path, base)
    file_path = parent_path / base
    file_path.write_text(f"# {title.strip() or 'Untitled'}\n", encoding="utf-8")
    rel = file_path.relative_to(DOCS_ROOT).as_posix()
    return jsonify({"status": "ok", "file": rel})


@app.route("/api/delete_dirs", methods=["POST"])
def api_delete_dirs():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    dirs = payload.get("dirs")
    if not isinstance(dirs, list):
        return _json_error("Expected `dirs` to be a JSON list.", 400)

    deleted: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for item in dirs:
        if not isinstance(item, str) or not item.strip():
            continue
        rel = _safe_rel_path(item.strip().strip("/"))
        try:
            path = _safe_docs_path(rel)
        except Exception as exc:
            errors.append({"dir": rel, "error": str(exc)})
            continue

        if not path.exists():
            continue
        if not path.is_dir():
            errors.append({"dir": rel, "error": "Not a directory."})
            continue
        try:
            path.rmdir()
            deleted.append({"kind": "dir", "path": rel})
        except OSError as exc:
            errors.append({"dir": rel, "error": f"Not empty: {exc}"})
        except Exception as exc:
            errors.append({"dir": rel, "error": str(exc)})

    return jsonify({"status": "ok", "deleted": deleted, "errors": errors})


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

        desired_dirs = _collect_desired_folder_dirs(nodes)
        _ensure_desired_folder_dirs_exist(desired_dirs)
        _cleanup_stray_empty_dirs(DOCS_ROOT, keep_rel_dirs=desired_dirs)
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
