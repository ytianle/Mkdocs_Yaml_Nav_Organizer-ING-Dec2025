# Run locally with: pip install -r requirements.txt && python app.py
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from itertools import count
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable, Iterator
import re

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


def _normalize_rel_path(path: str) -> str:
    """Normalize a nav path to a clean, POSIX-like relative form."""
    cleaned = (path or "").strip().replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    cleaned = cleaned.lstrip("./")
    return cleaned


def _slugify_segment(title: str) -> str:
    """Convert a nav title into a filesystem-friendly directory name."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", (title or "").strip().lower())
    slug = slug.strip("-")
    return slug or "section"


def _collect_paths(nodes: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []

    def _walk(items: list[dict[str, Any]]):
        for node in items:
            path = node.get("path")
            if path:
                paths.append(_normalize_rel_path(path))
            _walk(node.get("children") or [])

    _walk(nodes or [])
    return paths


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


def _derive_folder_segment(node: dict[str, Any], parent_base: str) -> str | None:
    """Try to infer the folder name from existing child paths."""
    candidates: list[str] = []

    def _collect(items: list[dict[str, Any]]):
        for child in items:
            path = child.get("path")
            if path:
                normalized = _normalize_rel_path(path)
                remainder = normalized
                if parent_base:
                    prefix = f"{parent_base}/"
                    if normalized.startswith(prefix):
                        remainder = normalized[len(prefix):]
                parts = remainder.split("/")
                if parts and parts[0]:
                    candidates.append(parts[0])
            _collect(child.get("children") or [])

    _collect(node.get("children") or [])
    if not candidates:
        return None
    counter = Counter(candidates)
    return counter.most_common(1)[0][0]


def _rebase_tree(nodes: list[dict[str, Any]], parent_base: str = "") -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Rebuild a nav tree so that file paths follow the logical folder structure.

    Returns (new_tree, moves) where moves is a list of (old_path, new_path) for files
    that need to be moved on disk.
    """
    rebased: list[dict[str, Any]] = []
    moves: list[tuple[str, str]] = []

    def _join(base: str, name: str) -> str:
        return f"{base}/{name}" if base else name

    for node in nodes:
        title = (node.get("title") or "").strip() or "untitled"
        path = _normalize_rel_path(node.get("path") or "")
        children = node.get("children") or []
        node_id = node.get("id")

        if children:
            segment = _derive_folder_segment(node, parent_base) or _slugify_segment(title)
            folder_base = _join(parent_base, segment)
            rebased_children, child_moves = _rebase_tree(children, folder_base)
            rebased.append({
                "id": node_id,
                "title": title,
                "path": None,
                "children": rebased_children,
            })
            moves.extend(child_moves)
            continue

        filename = Path(path).name if path else f"{_slugify_segment(title)}.md"
        new_path = _join(parent_base, filename) if parent_base else filename
        if path and path != new_path:
            moves.append((path, new_path))
            # If the title was just mirroring the old path, keep it in sync with the new path
            if title.lower() == path.lower():
                title = new_path
        rebased.append({
            "id": node_id,
            "title": title,
            "path": new_path,
            "children": [],
        })

    return rebased, moves


def _apply_moves(moves: list[tuple[str, str]]) -> None:
    """Move files on disk according to the provided mapping."""
    for src_rel, dst_rel in moves:
        src_path = DOCS_ROOT / src_rel
        dst_path = DOCS_ROOT / dst_rel
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_rel}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            raise FileExistsError(f"Target already exists: {dst_rel}")
        src_path.rename(dst_path)

    # Clean up empty source directories after moves.
    source_dirs: set[Path] = set()
    for src_rel, _ in moves:
        source_dirs.add((DOCS_ROOT / src_rel).parent)
    _cleanup_empty_dirs(source_dirs)


def _cleanup_empty_dirs(paths: Iterable[Path]) -> None:
    """Remove empty directories under DOCS_ROOT, stopping at DOCS_ROOT."""
    to_check: set[Path] = set()
    for path in paths:
        current = path
        while DOCS_ROOT in current.parents and current != DOCS_ROOT:
            to_check.add(current)
            current = current.parent

    # Remove deepest first so parents can become empty.
    for directory in sorted(to_check, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            # Not empty or not removable; ignore.
            continue


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
    except ValueError as exc:
        return jsonify({"error": f"Invalid nav payload: {exc}"}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Invalid nav payload: {exc}"}), 400

    # Confirm the currently referenced files exist before we start moving things.
    missing_paths = _validate_paths_exist(validated_payload)
    if missing_paths:
        return jsonify({
            "error": "Missing markdown files.",
            "missing": missing_paths,
            "docs_root": str(DOCS_ROOT),
        }), 400

    # Rebase paths to reflect the new logical folder positions.
    try:
        rebased_tree, planned_moves = _rebase_tree(validated_payload)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Invalid nav payload: {exc}"}), 400

    planned_moves = [(src, dst) for src, dst in planned_moves if src != dst]
    final_paths = _collect_paths(rebased_tree)
    if len(final_paths) != len(set(final_paths)):
        return jsonify({"error": "Duplicate target paths after reordering."}), 400

    existing_paths = set(_collect_paths(validated_payload))
    collisions: list[dict[str, str]] = []
    for src, dst in planned_moves:
        if dst in existing_paths and dst != src:
            collisions.append({"source": src, "target": dst, "reason": "target already referenced in nav"})
            continue
        if (DOCS_ROOT / dst).exists() and dst not in existing_paths:
            collisions.append({"source": src, "target": dst, "reason": "target already exists on disk"})

    if collisions:
        return jsonify({
            "error": "Conflicting target paths.",
            "collisions": collisions,
        }), 400

    try:
        _apply_moves(planned_moves)
    except (FileNotFoundError, FileExistsError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to move files: {exc}"}), 500

    # After moving files, ensure the rebased nav points to valid locations.
    rebased_missing = _validate_paths_exist(rebased_tree)
    if rebased_missing:
        return jsonify({
            "error": "Missing markdown files after applying moves.",
            "missing": rebased_missing,
            "docs_root": str(DOCS_ROOT),
        }), 500

    new_nav = _tree_to_nav(rebased_tree)

    storage = NavStorage(MKDOCS_PATH)
    try:
        backup_path = storage.save_nav(new_nav)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to save nav: {exc}"}), 500
    response = {
        "status": "ok",
        "moves_applied": len(planned_moves),
    }
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
