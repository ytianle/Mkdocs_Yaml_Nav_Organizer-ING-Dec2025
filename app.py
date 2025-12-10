# Run locally with: pip install -r requirements.txt && python app.py
from __future__ import annotations

import os
from datetime import datetime
from itertools import count
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable, Iterator

from flask import Flask, jsonify, render_template, request
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

app = Flask(__name__, template_folder="templates")

yaml = YAML()
yaml.preserve_quotes = True

MKDOCS_FILENAME = "mkdocs.yml"


def _resolve_mkdocs_path() -> Path:
    """Resolve the mkdocs.yml path, honoring an optional environment override."""
    env_path = os.environ.get("MKDOCS_PATH") or os.environ.get("MKDOCS_FILE")
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.is_absolute() else Path.cwd() / candidate
    return Path.cwd() / MKDOCS_FILENAME


MKDOCS_PATH = _resolve_mkdocs_path()
ALLOWED_DOC_EXTS = {".md", ".markdown", ".mdx"}


def _resolve_docs_root() -> Path:
    env_root = os.environ.get("DOCS_ROOT")
    if env_root:
        candidate = Path(env_root)
        return candidate if candidate.is_absolute() else Path.cwd() / candidate
    try:
        config = NavStorage(MKDOCS_PATH).load()
        docs_dir = config.get("docs_dir")
        if isinstance(docs_dir, str) and docs_dir.strip():
            path = Path(docs_dir.strip())
            return path if path.is_absolute() else MKDOCS_PATH.parent / path
    except Exception:
        pass
    for candidate in ("docs", "doc"):
        guess = Path.cwd() / candidate
        if guess.exists():
            return guess.resolve()
    return Path.cwd()


class NavStorage:
    """Small helper around the mkdocs.yml file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> CommentedMap:
        if not self.path.exists():
            return CommentedMap()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = yaml.load(handle) or CommentedMap()
        except Exception as exc:  # pragma: no cover - defensive around YAML parsing
            raise ValueError(f"Failed to read {self.path.name}: {exc}") from exc
        if not isinstance(data, CommentedMap):
            raise TypeError("mkdocs.yml must contain a mapping at the top level.")
        return data

    def save_nav(self, nav_items: CommentedSeq) -> Path | None:
        data = self.load()
        data["nav"] = nav_items
        backup_path = _backup_file(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.dump(data, handle)
        return backup_path


DOCS_ROOT = _resolve_docs_root()


def _backup_file(path: Path) -> Path | None:
    """Create a timestamped backup before overwriting mkdocs.yml."""
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{timestamp}")
    copy2(path, backup_path)
    return backup_path


def _load_mkdocs() -> CommentedMap:
    storage = NavStorage(MKDOCS_PATH)
    return storage.load()


def _nav_to_tree(nav_items: Iterable[Any], id_counter: Iterator[int]) -> list[dict[str, Any]]:
    tree: list[dict[str, Any]] = []
    for entry in nav_items or []:
        node_id = f"node-{next(id_counter)}"
        title = ""
        path = None
        children: list[dict[str, Any]] = []

        if isinstance(entry, str):
            title = entry
            path = entry
        elif isinstance(entry, (CommentedMap, dict)):
            items = list(entry.items())
            for key, value in items:
                title = str(key)
                if isinstance(value, (list, CommentedSeq)):
                    children = _nav_to_tree(value, id_counter)
                    path = None
                elif value is None:
                    path = None
                    children = []
                else:
                    # Represent single file mappings (e.g., "index: index/index.md")
                    # as a folder with one child for better drag/drop UX.
                    child_id = f"node-{next(id_counter)}"
                    filename = str(value)
                    children = [{
                        "id": child_id,
                        "title": filename,
                        "path": filename,
                        "children": [],
                    }]
                    path = None
                break
        else:
            title = str(entry)
            path = str(entry)

        tree.append({
            "id": node_id,
            "title": title,
            "path": path,
            "children": children,
        })
    return tree


def _tree_to_nav(nodes: Iterable[dict[str, Any]]) -> CommentedSeq:
    nav_seq: CommentedSeq = CommentedSeq()
    for node in nodes:
        title = str(node.get("title", "")).strip()
        path = node.get("path")
        children = node.get("children") or []

        if children:
            nav_seq.append(CommentedMap({title: _tree_to_nav(children)}))
        elif path not in (None, ""):
            if title == path:
                nav_seq.append(path)
            else:
                nav_seq.append(CommentedMap({title: path}))
        else:
            nav_seq.append(CommentedMap({title: None}))
    return nav_seq


def _validate_tree_payload(nodes: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if not isinstance(nodes, list):
        raise ValueError("Nav payload must be a JSON array.")
    validated: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Item {idx + 1} at depth {depth} must be an object.")
        title = node.get("title", "")
        if not isinstance(title, str):
            raise ValueError(f"Item {idx + 1} at depth {depth} has an invalid title.")
        cleaned_title = title.strip()

        path = node.get("path", None)
        if path not in (None, "") and not isinstance(path, str):
            raise ValueError(f"Item {idx + 1} at depth {depth} has an invalid path.")
        children = node.get("children") or []
        if children and path not in (None, ""):
            raise ValueError(f"Item {idx + 1} at depth {depth} cannot have both children and a path.")

        validated.append({
            "id": str(node.get("id") or ""),
            "title": cleaned_title,
            "path": None if children else (path if path not in ("", None) else None),
            "children": _validate_tree_payload(children, depth=depth + 1) if children else [],
        })
    return validated


def _validate_paths_exist(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure all nodes with paths point to existing files relative to DOCS_ROOT."""
    missing: list[dict[str, Any]] = []

    def _walk(items: list[dict[str, Any]]):
        for node in items:
            path = node.get("path")
            if path:
                exists, case_mismatch = _path_exists_case_sensitive(path)
                if not exists or case_mismatch:
                    missing.append({
                        "path": path,
                        "line": _find_line_number_in_mkdocs(path),
                        "case_mismatch": case_mismatch,
                    })
            _walk(node.get("children") or [])

    _walk(nodes)
    return missing


def _path_exists_case_sensitive(rel_path: str) -> tuple[bool, bool]:
    """Check existence with case sensitivity, even on case-insensitive filesystems.

    Returns (exists, case_mismatch) where case_mismatch=True if a path exists but
    differs only by case at some component.
    """
    if not rel_path:
        return False, False
    rel = Path(rel_path)
    parts = rel.parts
    current = DOCS_ROOT
    case_mismatch = False
    try:
        for part in parts:
            # gather exact names in current directory
            try:
                names = {p.name for p in current.iterdir()}
            except OSError:
                return False, False
            if part not in names:
                # Check case-insensitive match to flag case issues.
                lowered = {n.lower() for n in names}
                if part.lower() in lowered:
                    case_mismatch = True
                    # find the actual different-cased name to continue traversal
                    actual = next(n for n in names if n.lower() == part.lower())
                    current = current / actual
                    continue
                return False, case_mismatch
            current = current / part
        return current.exists(), case_mismatch
    except Exception:
        return False, case_mismatch


def _find_line_number_in_mkdocs(fragment: str) -> int | None:
    """Best-effort: find the first line in mkdocs.yml that mentions the fragment or its basename."""
    if not MKDOCS_PATH.exists():
        return None
    fragment_str = str(fragment)
    basename = Path(fragment_str).name
    try:
        with MKDOCS_PATH.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                if fragment_str in line or basename in line:
                    return idx
    except OSError:
        return None
    return None


def _build_source_tree(current: Path, base: Path) -> list[dict[str, Any]]:
    """Build a simple folder/file tree rooted at `base`, including only markdown-like files."""
    if not current.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return entries

    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append({
                "name": child.name,
                "path": child.relative_to(base).as_posix(),
                "kind": "dir",
                "children": _build_source_tree(child, base),
            })
        else:
            if child.suffix.lower() not in ALLOWED_DOC_EXTS:
                continue
            entries.append({
                "name": child.name,
                "path": child.relative_to(base).as_posix(),
                "kind": "file",
                "children": [],
            })
    return entries


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/nav", methods=["GET"])
def get_nav():
    try:
        data = _load_mkdocs()
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 500
    nav_items = data.get("nav", CommentedSeq())
    counter = count(1)
    tree = _nav_to_tree(nav_items, counter)
    missing = _validate_paths_exist(tree)
    return jsonify({
        "tree": tree,
        "missing": missing,
        "docs_root": str(DOCS_ROOT),
    })


@app.route("/nav", methods=["POST"])
def update_nav():
    payload = request.get_json(silent=True)
    if not isinstance(payload, list):
        return jsonify({"error": "Expected a JSON list at the root."}), 400

    try:
        validated_payload = _validate_tree_payload(payload)
        missing_paths = _validate_paths_exist(validated_payload)
        if missing_paths:
            return jsonify({
                "error": "Missing markdown files.",
                "missing": missing_paths,
                "docs_root": str(DOCS_ROOT),
            }), 400
        new_nav = _tree_to_nav(validated_payload)
    except ValueError as exc:
        return jsonify({"error": f"Invalid nav payload: {exc}"}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Invalid nav payload: {exc}"}), 400

    storage = NavStorage(MKDOCS_PATH)
    try:
        backup_path = storage.save_nav(new_nav)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to save nav: {exc}"}), 500
    response = {"status": "ok"}
    if backup_path:
        response["backup"] = str(backup_path)
    return jsonify(response)


@app.route("/health", methods=["GET"])
def healthcheck():
    exists = MKDOCS_PATH.exists()
    return jsonify({
        "status": "ok",
        "mkdocs_path": str(MKDOCS_PATH),
        "exists": exists,
    })


@app.route("/source", methods=["GET"])
def get_source_tree():
    tree = _build_source_tree(DOCS_ROOT, DOCS_ROOT)
    return jsonify(tree)


if __name__ == "__main__":
    app.run(debug=True)
