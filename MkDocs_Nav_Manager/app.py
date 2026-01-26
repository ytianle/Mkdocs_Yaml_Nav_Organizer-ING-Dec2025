from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from string import Template
from typing import Any, Iterable
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request, Response, stream_with_context
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
# mkdocs.yml style: keep nav and nested blocks compact with 2-space indentation.
yaml.indent(mapping=2, sequence=2, offset=2)
# Avoid line wrapping that can split `Key: value` into two lines.
yaml.width = 4096

ALLOWED_DOC_EXTS = {".md", ".markdown", ".mdx"}

MKDOCS_LOCK = threading.RLock()
MKDOCS_PROC: subprocess.Popen[str] | None = None
MKDOCS_STATUS: dict[str, Any] = {
    "state": "stopped",  # stopped | starting | rendering | ready | error
    "running": False,
    "message": "",
}
MKDOCS_LOGS: list[str] = []
MKDOCS_LOG_LIMIT = 200
MKDOCS_STOPPING = False
MKDOCS_SUBSCRIBERS: list[queue.Queue[str]] = []
ICONSEARCH_INDEX_PATH = BASE_DIR / "static" / "iconsearch_index.json"


def _mkdocs_broadcast(payload: dict[str, Any]) -> None:
    data = f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
    with MKDOCS_LOCK:
        subscribers = list(MKDOCS_SUBSCRIBERS)
    for q in subscribers:
        try:
            q.put_nowait(data)
        except queue.Full:
            continue


def _mkdocs_set_status(state: str, *, running: bool | None = None, message: str | None = None) -> None:
    changed = False
    payload: dict[str, Any] | None = None
    with MKDOCS_LOCK:
        if MKDOCS_STATUS.get("state") != state:
            changed = True
        MKDOCS_STATUS["state"] = state
        if running is not None:
            if MKDOCS_STATUS.get("running") != running:
                changed = True
            MKDOCS_STATUS["running"] = running
        if message is not None:
            MKDOCS_STATUS["message"] = message
        if changed or message:
            payload = {"kind": "status", **MKDOCS_STATUS}
    if payload:
        _mkdocs_broadcast(payload)


def _mkdocs_add_log(line: str) -> None:
    text = (line or "").strip()
    if not text:
        return
    with MKDOCS_LOCK:
        MKDOCS_LOGS.append(text)
        if len(MKDOCS_LOGS) > MKDOCS_LOG_LIMIT:
            del MKDOCS_LOGS[: len(MKDOCS_LOGS) - MKDOCS_LOG_LIMIT]
    _mkdocs_broadcast({"kind": "log", "line": text})


def _mkdocs_parse_line(line: str) -> str | None:
    low = line.lower()
    if "error" in low or "traceback" in low:
        return "error"
    if "building documentation" in low or "rebuilding documentation" in low or "cleaning site directory" in low:
        return "rendering"
    if "documentation built" in low or "serving on" in low:
        return "ready"
    return None


def _mkdocs_read_logs(proc: subprocess.Popen[str]) -> None:
    global MKDOCS_PROC
    try:
        if proc.stdout:
            for line in proc.stdout:
                msg = line.strip()
                if msg:
                    _mkdocs_add_log(msg)
                    state = _mkdocs_parse_line(msg)
                    if state:
                        _mkdocs_set_status(state, running=True, message=msg)
                    else:
                        with MKDOCS_LOCK:
                            current = MKDOCS_STATUS.get("state", "rendering")
                        _mkdocs_set_status(current, running=True, message=msg)
    finally:
        code = proc.poll()
        with MKDOCS_LOCK:
            stopping = MKDOCS_STOPPING
            MKDOCS_STOPPING = False
        if code not in (None, 0) and not stopping:
            msg = f"mkdocs exited with code {code}"
            _mkdocs_add_log(msg)
            _mkdocs_set_status("error", running=False, message=msg)
        else:
            _mkdocs_add_log("mkdocs stopped")
            _mkdocs_set_status("stopped", running=False, message="mkdocs stopped")
        with MKDOCS_LOCK:
            if MKDOCS_PROC is proc:
                MKDOCS_PROC = None


def _mkdocs_start() -> dict[str, Any]:
    global MKDOCS_PROC
    with MKDOCS_LOCK:
        if MKDOCS_PROC and MKDOCS_PROC.poll() is None:
            return _mkdocs_monitor()

        try:
            proc = subprocess.Popen(
                ["mkdocs", "serve", "--livereload"],
                cwd=str(MKDOCS_PATH.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            msg = "mkdocs not found in PATH"
            _mkdocs_add_log(msg)
            _mkdocs_set_status("error", running=False, message=msg)
            return _mkdocs_monitor()
        except Exception as exc:
            msg = f"Failed to start mkdocs: {exc}"
            _mkdocs_add_log(msg)
            _mkdocs_set_status("error", running=False, message=msg)
            return _mkdocs_monitor()

        MKDOCS_PROC = proc
        _mkdocs_add_log("mkdocs starting")
        _mkdocs_set_status("starting", running=True, message="mkdocs starting")
        thread = threading.Thread(target=_mkdocs_read_logs, args=(proc,), daemon=True)
        thread.start()
        return _mkdocs_monitor()


def _mkdocs_stop() -> dict[str, Any]:
    global MKDOCS_PROC
    with MKDOCS_LOCK:
        proc = MKDOCS_PROC
        MKDOCS_PROC = None
        MKDOCS_STOPPING = True
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    _mkdocs_add_log("mkdocs stopped")
    _mkdocs_set_status("stopped", running=False, message="mkdocs stopped")
    return _mkdocs_monitor()


def _shutdown_handler(signum: int | None = None, frame: Any | None = None) -> None:
    _mkdocs_stop()
    if signum is not None:
        os._exit(0)


def _mkdocs_status() -> dict[str, Any]:
    with MKDOCS_LOCK:
        proc = MKDOCS_PROC
        status = dict(MKDOCS_STATUS)
    if proc and proc.poll() is not None:
        _mkdocs_set_status("stopped", running=False, message="mkdocs stopped")
        with MKDOCS_LOCK:
            status = dict(MKDOCS_STATUS)
    return status


def _mkdocs_monitor() -> dict[str, Any]:
    status = _mkdocs_status()
    with MKDOCS_LOCK:
        logs = list(MKDOCS_LOGS)
    status["logs"] = logs
    return status


def _mkdocs_snapshot() -> dict[str, Any]:
    snap = _mkdocs_monitor()
    snap["kind"] = "snapshot"
    return snap


def _is_allowed_local_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost"}


def _probe_url(url: str, *, timeout: float = 2.0) -> tuple[bool, int | None]:
    try:
        req = Request(url, method="GET", headers={"User-Agent": "mkdocs-nav-manager"})
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
        return True, status
    except HTTPError as exc:
        return False, exc.code
    except (URLError, Exception):
        return False, None


def _json_error(message: str, status: int = 400, **extra: Any):
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return jsonify(payload), status


def _icon_entry(
    *,
    entry_type: str,
    name: str,
    shortcode: str,
    path: str | None = None,
    unicode_value: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": entry_type,
        "name": name,
        "shortcode": shortcode,
    }
    if path:
        payload["path"] = path
    if unicode_value:
        payload["unicode"] = unicode_value
    if keywords:
        payload["keywords"] = keywords
    return payload


def _join_base(base: str, path: str) -> str:
    base = base.strip()
    path = path.strip()
    if not base:
        return path
    if base.endswith("/"):
        base = base[:-1]
    if path.startswith("/"):
        path = path[1:]
    return f"{base}/{path}"


def _load_iconsearch_index() -> list[dict[str, Any]]:
    if not ICONSEARCH_INDEX_PATH.exists():
        raise FileNotFoundError(f"{ICONSEARCH_INDEX_PATH}")
    raw = ICONSEARCH_INDEX_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("iconsearch_index.json must be a JSON list or object")

    entries: list[dict[str, Any]] = []
    icons = data.get("icons") if isinstance(data.get("icons"), dict) else {}
    emojis = data.get("emojis") if isinstance(data.get("emojis"), dict) else {}
    icons_base = icons.get("base") if isinstance(icons.get("base"), str) else ""
    emojis_base = emojis.get("base") if isinstance(emojis.get("base"), str) else ""
    icons_data = icons.get("data") if isinstance(icons.get("data"), dict) else {}
    emojis_data = emojis.get("data") if isinstance(emojis.get("data"), dict) else {}

    for name, rel_payload in icons_data.items():
        if not isinstance(name, str):
            continue
        path = ""
        keywords = None
        unicode_value = None
        if isinstance(rel_payload, dict):
            raw_path = rel_payload.get("path", rel_payload.get("svg"))
            if raw_path is not None:
                path = _join_base(icons_base, str(raw_path))
            if isinstance(rel_payload.get("keywords"), list):
                keywords = [str(k) for k in rel_payload["keywords"] if str(k).strip()]
            if isinstance(rel_payload.get("unicode"), str):
                unicode_value = rel_payload["unicode"]
        else:
            if rel_payload is not None:
                path = _join_base(icons_base, str(rel_payload))
        shortcode = f":{name.strip()}:"
        entries.append(
            _icon_entry(
                entry_type="icon",
                name=name,
                shortcode=shortcode,
                path=path or None,
                unicode_value=unicode_value,
                keywords=keywords,
            )
        )

    for name, rel_payload in emojis_data.items():
        if not isinstance(name, str):
            continue
        path = ""
        keywords = None
        unicode_value = None
        if isinstance(rel_payload, dict):
            raw_path = rel_payload.get("path", rel_payload.get("svg"))
            if raw_path is not None:
                path = _join_base(emojis_base, str(raw_path))
            if isinstance(rel_payload.get("keywords"), list):
                keywords = [str(k) for k in rel_payload["keywords"] if str(k).strip()]
            if isinstance(rel_payload.get("unicode"), str):
                unicode_value = rel_payload["unicode"]
        else:
            if rel_payload is not None:
                path = _join_base(emojis_base, str(rel_payload))
        shortcode = f":{name.strip()}:"
        entries.append(
            _icon_entry(
                entry_type="emoji",
                name=name,
                shortcode=shortcode,
                path=path or None,
                unicode_value=unicode_value,
                keywords=keywords,
            )
        )

    return entries


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _backup_file(path: Path) -> Path | None:
    if str(os.environ.get("MKDOCS_BACKUP", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return None
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


def _snakeify(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", str(text or "")).strip()
    cleaned = re.sub(r"c\+\+", " cpp ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"c#", " csharp ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("++", " plus plus ").replace("+", " plus ").replace("#", " sharp ")
    cleaned = re.sub(r"[^\w]+", "_", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower() or "item"


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


def _same_file_path(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _is_protected_section(title: str | None) -> bool:
    if not title:
        return False
    upper = str(title).strip().upper()
    return upper in {"HOME", "JOURNAL"}


def _has_protected_ancestor(ancestors: list["Node"]) -> bool:
    return any(a.type == "folder" and _is_protected_section(a.title) for a in ancestors)


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


def _ui_state_path() -> Path:
    env = os.environ.get("UI_STATE_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return BASE_DIR / ".page_tree_ui_state.json"


def _load_ui_state() -> dict[str, Any]:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        with UI_STATE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_ui_state(payload: dict[str, Any]) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UI_STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


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


def _dump_nav_block(nav_value: Any) -> str:
    """Dump a standalone `nav:` block with 2-space indentation."""
    tmp = CommentedMap()
    tmp["nav"] = nav_value if nav_value is not None else CommentedSeq()
    buf = []
    # ruamel needs a stream-like object; implement minimal write collector
    class _W:
        def write(self, s: str):
            if isinstance(s, (bytes, bytearray)):
                buf.append(s.decode("utf-8", errors="replace"))
            else:
                buf.append(str(s))

    yaml.dump(tmp, _W())
    text = "".join(buf)
    # Ensure trailing newline for file splicing.
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _write_mkdocs_nav_only(mkdocs: Path, nav_value: Any) -> Path | None:
    """Replace only the root-level `nav:` block in mkdocs.yml, leaving the rest untouched."""
    backup = _backup_file(mkdocs)
    mkdocs.parent.mkdir(parents=True, exist_ok=True)
    original = mkdocs.read_text(encoding="utf-8") if mkdocs.exists() else ""
    lines = original.splitlines(keepends=True)

    start = None
    for i, line in enumerate(lines):
        if line.startswith("nav:") and (len(line) == 4 or line[4].isspace()):
            start = i
            break

    nav_block = _dump_nav_block(nav_value)
    nav_lines = nav_block.splitlines(keepends=True)

    if start is None:
        # No existing nav: append at end with a separating newline if needed.
        if original and not original.endswith("\n"):
            original += "\n"
        if original and not original.endswith("\n\n"):
            original += "\n"
        mkdocs.write_text(original + nav_block, encoding="utf-8")
        return backup

    # Find the end of the nav block: next top-level key (no indent) that looks like "key:".
    end = len(lines)
    for j in range(start + 1, len(lines)):
        l = lines[j]
        if not l.strip():
            continue
        if not l.startswith((" ", "\t", "-")) and ":" in l:
            end = j
            break

    new_lines = lines[:start] + nav_lines + lines[end:]
    mkdocs.write_text("".join(new_lines), encoding="utf-8")
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
UI_STATE_PATH = _ui_state_path()
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
    is_overview: bool = False  # for page: section overview page (title-less, fixed slot)
    children: list["Node"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "segment": self.segment,
            "file": self.file,
            "file_prev": self.file_prev,
            "is_overview": self.is_overview,
            "children": [c.to_dict() for c in (self.children or [])],
        }


def _node_from_dict(data: dict[str, Any]) -> Node:
    node_type = str(data.get("type") or "").strip()
    if node_type not in {"folder", "page"}:
        raise ValueError("Invalid node type.")
    is_overview = bool(data.get("is_overview", False))
    title = str(data.get("title") or "").strip()
    if not title and not is_overview:
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
    return Node(
        id=node_id,
        type="page",
        title=title,
        segment=None,
        file=file_str,
        file_prev=prev_str,
        is_overview=is_overview,
        children=[],
    )


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
                # Only infer a directory segment when the remainder actually contains a directory.
                # A direct child page like "<parent>/index.md" provides no segment signal.
                if "/" in remainder:
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
        if _is_protected_section(node.title):
            node.segment = ""
            _infer_segments_in_place(node.children or [], parent_prefix)
            continue
        inferred = _infer_segment_from_children(node, parent_prefix)
        if inferred:
            node.segment = inferred
            next_prefix = f"{parent_prefix}/{inferred}" if parent_prefix else inferred
        else:
            node.segment = node.segment or _slugify(node.title)
            next_prefix = parent_prefix
        _infer_segments_in_place(node.children or [], next_prefix)


def _ensure_section_overviews_in_place(nodes: list[Node], parent_dir: str = "") -> None:
    """Best-effort: if a folder has an unlabeled first item or README/index direct child, mark it as overview."""

    for folder in nodes:
        if folder.type != "folder":
            continue

        folder_dir = f"{parent_dir}/{folder.segment}".strip("/") if folder.segment else parent_dir
        children = list(folder.children or [])

        # If there is already an overview node, keep it and force it to the front.
        ov_idx = next((i for i, c in enumerate(children) if c.type == "page" and c.is_overview), None)
        if ov_idx is not None:
            ov = children.pop(ov_idx)
            children.insert(0, ov)
        else:
            # Mark a direct child README/index as overview if present; otherwise keep empty slot.
            for i, child in enumerate(children):
                if child.type != "page" or not child.file:
                    continue
                rel = _safe_rel_path(child.file)
                if folder_dir:
                    if not rel.startswith(f"{folder_dir}/"):
                        continue
                    base = rel[len(folder_dir) + 1 :]
                else:
                    base = rel
                if "/" in base:
                    continue
                if Path(base).name.lower() in {"readme.md", "index.md"}:
                    child.is_overview = True
                    children.pop(i)
                    children.insert(0, child)
                    break

        folder.children = children
        _ensure_section_overviews_in_place(folder.children or [], folder_dir)


def _import_mkdocs_nav(nav_items: Any, parent_prefix: str = "", depth: int = 0) -> list[Node]:
    nodes: list[Node] = []
    if not isinstance(nav_items, (list, CommentedSeq)):
        return nodes
    first = True
    for entry in nav_items:
        if isinstance(entry, str):
            rel = _safe_rel_path(entry)
            if first:
                # Section overview: allow first item to be a plain path (no title).
                nodes.append(Node(id=_new_id(), type="page", title="", file=rel, is_overview=True, children=[]))
            else:
                nodes.append(Node(id=_new_id(), type="page", title=entry, file=rel, children=[]))
            first = False
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
                    children=_import_mkdocs_nav(value, parent_prefix, depth + 1),
                )
                nodes.append(folder)
                first = False
                continue

            if value is None:
                folder = Node(id=_new_id(), type="folder", title=title_str, segment=_slugify(title_str), children=[])
                nodes.append(folder)
                first = False
                continue

            rel = _safe_rel_path(str(value))
            # Back-compat: "Overview: path" is treated as the section overview.
            if title_str.lower() == "overview":
                nodes.append(Node(id=_new_id(), type="page", title="", file=rel, is_overview=True, children=[]))
            # Rule: at top-level, "Title: path" means a Section with a Display Page.
            # Inside a section list, "Title: path" means a normal Page.
            elif depth == 0:
                folder = Node(
                    id=_new_id(),
                    type="folder",
                    title=title_str,
                    segment=_slugify(title_str),
                    children=[Node(id=_new_id(), type="page", title="", file=rel, is_overview=True, children=[])],
                )
                nodes.append(folder)
            else:
                nodes.append(Node(id=_new_id(), type="page", title=title_str, file=rel, children=[]))
            first = False
            continue

        nodes.append(Node(id=_new_id(), type="page", title=str(entry), file=None, children=[]))
        first = False

    return nodes


def _tree_to_mkdocs_nav(nodes: Iterable[Node], depth: int = 0) -> CommentedSeq:
    nav = CommentedSeq()
    for node in nodes:
        title = (node.title or "").strip() or "untitled"
        if node.type == "folder":
            children = list(node.children or [])
            overview = next((c for c in children if c.type == "page" and c.is_overview and c.file), None)
            rest = [c for c in children if c is not overview]

            # If this is a top-level section with only a display page, serialize as "Title: path".
            # Exception: if the display page lives under `blog/`, keep the list form to avoid
            # collapsing `- Section: [blog/index.md]` into `- Section: blog/index.md`.
            if depth == 0 and overview and not rest:
                overview_rel = _safe_rel_path(overview.file)
                if overview_rel.startswith("blog/"):
                    seq = CommentedSeq()
                    seq.append(overview_rel)
                    nav.append(CommentedMap({title: seq}))
                    continue
                nav.append(CommentedMap({title: _safe_rel_path(overview.file)}))
                continue

            seq = CommentedSeq()
            if overview and overview.file:
                seq.append(_safe_rel_path(overview.file))
            seq.extend(_tree_to_mkdocs_nav(rest, depth + 1))
            nav.append(CommentedMap({title: seq}))
            continue
        if node.is_overview:
            if node.file:
                nav.append(_safe_rel_path(node.file))
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
            segs = [
                a.segment
                for a in (ancestors + [node])
                if a.type == "folder" and a.segment and not _is_protected_section(a.title)
            ]
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
    segments = [a.segment for a in ancestors if a.type == "folder" and a.segment and not _is_protected_section(a.title)]
    return "/".join(segments)


def _plan_file_sync(
    nodes: list[Node], *, create_missing: bool, preserve_page_dirs: bool
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    page_moves: list[tuple[Path, Path]] = []
    asset_moves: list[tuple[Path, Path]] = []
    warnings: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    planned_sources: set[Path] = set()
    planned_targets: set[Path] = set()
    used_rel: set[str] = set()

    def _unique_rel(desired_rel: str) -> str:
        desired_rel = _safe_rel_path(desired_rel)
        if desired_rel in used_rel:
            raise ValueError(f"Duplicate target: {desired_rel}")
        used_rel.add(desired_rel)
        return desired_rel

    entries: list[tuple[Node, list[Node], str, str, bool]] = []

    def collect(items: list[Node], ancestors: list[Node]):
        folder_dir = _compute_folder_dir(ancestors)
        protected = _has_protected_ancestor(ancestors)
        for node in items:
            if node.type == "folder":
                collect(node.children or [], ancestors + [node])
                continue
            basename = Path(node.file).name if node.file else f"{_slugify(node.title)}.md"
            desired_rel_default = f"{folder_dir}/{basename}" if folder_dir else basename
            desired_rel_default = _safe_rel_path(desired_rel_default)
            # If a user renamed a file in the editor, prefer the previous path as the source for moving.
            current_rel = _safe_rel_path(node.file_prev) if node.file_prev else (_safe_rel_path(node.file) if node.file else desired_rel_default)
            if protected:
                desired_rel = current_rel
            elif preserve_page_dirs and node.file:
                desired_rel = _safe_rel_path(node.file)
            else:
                desired_rel = desired_rel_default
            entries.append((node, list(ancestors), current_rel, desired_rel, protected))

    collect(nodes, [])

    # Precompute all sources that will be used for the plan (to avoid false collisions when swapping/moving).
    for node, _anc, current_rel, desired_rel, _protected in entries:
        src = DOCS_ROOT / current_rel
        if src.exists():
            planned_sources.add(src.resolve())

    # Preflight: detect collisions before mutating node paths or moving files.
    for node, _anc, current_rel, desired_rel, _protected in entries:
        try:
            desired_rel = _safe_rel_path(desired_rel)
        except Exception as exc:
            collisions.append({"type": "invalid_target", "target": desired_rel, "title": node.title, "error": str(exc)})
            continue
        if desired_rel in used_rel:
            collisions.append({"type": "duplicate_target", "target": desired_rel, "title": node.title})
            continue
        used_rel.add(desired_rel)
        src = DOCS_ROOT / current_rel
        dst = DOCS_ROOT / desired_rel
        if dst.exists() and not _same_file_path(src, dst) and dst.resolve() not in planned_sources:
            collisions.append({"type": "exists", "target": desired_rel, "title": node.title})

    if collisions:
        return [], [], warnings, collisions

    used_rel = set()

    def apply(entries_in: list[tuple[Node, list[Node], str, str, bool]]):
        for node, ancestors, current_rel, desired_rel, protected in entries_in:
            src = DOCS_ROOT / current_rel
            unique_rel = _unique_rel(desired_rel)
            dst = DOCS_ROOT / unique_rel
            if not src.exists():
                warnings.append({"type": "missing_file", "file": current_rel, "title": node.title})
                if create_missing and not protected:
                    _create_missing_page(unique_rel, node.title)
                    node.file = unique_rel
                    node.file_prev = None
                continue

            planned_targets.add(dst.resolve())

            if src != dst and not _same_file_path(src, dst):
                page_moves.append((src, dst))
            node.file = unique_rel
            node.file_prev = None

    apply(entries)

    _ensure_unique_targets(page_moves + asset_moves)
    return page_moves, asset_moves, warnings, collisions


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


def _preflight_moves(moves: list[tuple[Path, Path]]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for src, dst in moves:
        if not src.exists():
            errors.append({"type": "missing_source", "from": src.relative_to(DOCS_ROOT).as_posix()})
            continue
        if dst.exists() and not _same_file_path(src, dst):
            errors.append({"type": "target_exists", "to": dst.relative_to(DOCS_ROOT).as_posix()})
    return errors


def _collect_pages_under(nodes: Iterable[Node]) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        if n.type == "page":
            out.append(n)
        elif n.type == "folder":
            out.extend(_collect_pages_under(n.children or []))
    return out


def _apply_moves_with_temps(moves: list[tuple[Path, Path]]) -> list[dict[str, str]]:
    """Apply a set of renames safely even when moves form cycles."""
    if not moves:
        return []
    # Phase 1: move every source to a unique temp path in its source directory.
    tmp_moves: list[tuple[Path, Path]] = []
    final_moves: list[tuple[Path, Path]] = []
    touched_dirs: set[Path] = set()
    for src, dst in moves:
        touched_dirs.add(src.parent)
        touched_dirs.add(dst.parent)
        tmp = src.with_name(f".__tmp__{uuid4().hex}__{src.name}")
        if tmp.exists():
            raise FileExistsError(str(tmp))
        tmp_moves.append((src, tmp))
        final_moves.append((tmp, dst))

    for src, tmp in tmp_moves:
        if not src.exists():
            raise FileNotFoundError(str(src))
        tmp.parent.mkdir(parents=True, exist_ok=True)
        src.rename(tmp)

    applied: list[dict[str, str]] = []
    for tmp, dst in final_moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(str(dst))
        tmp.rename(dst)
        applied.append({"from": tmp.name, "to": dst.relative_to(DOCS_ROOT).as_posix()})

    _cleanup_empty_dirs(sorted(touched_dirs), DOCS_ROOT)
    return applied


def _plan_partial_sync(nodes: list[Node], moved_ids: set[str]) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[dict[str, Any]]]:
    """Plan moves only for pages under moved nodes (folders/pages)."""
    page_moves: list[tuple[Path, Path]] = []
    asset_moves: list[tuple[Path, Path]] = []
    collisions: list[dict[str, Any]] = []
    planned_sources: set[Path] = set()
    used_rel: set[str] = set()

    entries: list[tuple[Node, str, str, bool]] = []

    def walk(items: list[Node], ancestors: list[Node], under_moved: bool):
        folder_dir = _compute_folder_dir(ancestors)
        protected = _has_protected_ancestor(ancestors)
        for node in items:
            is_under = under_moved or (node.id in moved_ids)
            if node.type == "folder":
                walk(node.children or [], ancestors + [node], is_under)
                continue
            if not is_under:
                continue
            if not node.file:
                continue
            basename = Path(node.file).name
            current_rel = _safe_rel_path(node.file_prev) if node.file_prev else _safe_rel_path(node.file)
            if protected:
                desired_rel = current_rel
            else:
                desired_rel = f"{folder_dir}/{basename}" if folder_dir else basename
                desired_rel = _safe_rel_path(desired_rel)
            entries.append((node, current_rel, desired_rel, protected))

    walk(nodes, [], False)

    for node, current_rel, _desired_rel, _protected in entries:
        src = DOCS_ROOT / current_rel
        if src.exists():
            planned_sources.add(src.resolve())

    def _unique_rel(desired_rel: str) -> str:
        desired_rel = _safe_rel_path(desired_rel)
        if desired_rel in used_rel:
            raise ValueError(f"Duplicate target: {desired_rel}")
        used_rel.add(desired_rel)
        return desired_rel

    for node, current_rel, desired_rel, _protected in entries:
        try:
            desired_rel = _safe_rel_path(desired_rel)
        except Exception as exc:
            collisions.append({"type": "invalid_target", "target": desired_rel, "title": node.title, "error": str(exc)})
            continue
        if desired_rel in used_rel:
            collisions.append({"type": "duplicate_target", "target": desired_rel, "title": node.title})
            continue
        used_rel.add(desired_rel)
        src = DOCS_ROOT / current_rel
        dst = DOCS_ROOT / desired_rel
        if dst.exists() and not _same_file_path(src, dst) and dst.resolve() not in planned_sources:
            collisions.append({"type": "exists", "target": desired_rel, "title": node.title})

    if collisions:
        return [], [], collisions

    used_rel = set()

    for node, current_rel, desired_rel, _protected in entries:
        src = DOCS_ROOT / current_rel
        if not src.exists():
            continue
        unique_rel = _unique_rel(desired_rel)
        dst = DOCS_ROOT / unique_rel
        if _same_file_path(src, dst):
            node.file = unique_rel
            node.file_prev = None
            continue

        page_moves.append((src, dst))
        node.file = unique_rel
        node.file_prev = None

    if collisions:
        return [], [], collisions

    _ensure_unique_targets(page_moves + asset_moves)
    return page_moves, asset_moves, []


def _create_missing_page(file_rel: str, title: str) -> None:
    file_rel = _safe_rel_path(file_rel)
    path = DOCS_ROOT / file_rel
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.lower() == "readme.md":
        section_name = path.parent.name or "Section"
        path.write_text(_section_readme_template(section_name), encoding="utf-8")
    else:
        path.write_text(f"# {title.strip() or 'Untitled'}\n", encoding="utf-8")


def _section_readme_template(section_name: str) -> str:
    name = (section_name or "").strip() or "Section"
    tpl = Template(
        "---\n"
        "icon: material/library\n"
        "---\n\n"
        "# **$section**\n\n"
        "> fill chapter introduction here\n"
        "## **This chapter can be separated into the following sections:**\n\n"
        "{% set base_path = page.file.src_path | replace('README.md', '') %}\n\n"
        "{% for p in page.parent.children %}\n"
        "{% if (p.is_section or p.is_page) and p.file and p.file.src_path and p.file.src_path != page.file.src_path %}\n"
        "1. [{{ p.title }}]({{ p.file.src_path | replace(base_path, '', 1) }})\n"
        "{% elif p.is_section and p.children and p.children[0].file and p.children[0].file.src_path %}\n"
        "1. [{{ p.title }}]({{ p.children[0].file.src_path | replace(base_path, '', 1) }})\n"
        "{% endif %}\n"
        "{% endfor %}\n"
    )
    return tpl.substitute(section=name)


def _validate_imported_tree(nodes: list["Node"]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def walk(items: list["Node"], trail: list[str]):
        for node in items:
            if node.type == "folder":
                walk(node.children or [], trail + [node.title or node.segment or "(section)"])
                continue
            file_rel = (node.file or "").strip()
            if not file_rel:
                errors.append(
                    {
                        "type": "missing_file_field",
                        "title": node.title,
                        "path": " / ".join(trail + [node.title or "(page)"]),
                    }
                )
                continue
            try:
                safe_rel = _safe_rel_path(file_rel)
                if Path(safe_rel).suffix.lower() not in ALLOWED_DOC_EXTS:
                    errors.append(
                        {
                            "type": "invalid_extension",
                            "file": file_rel,
                            "title": node.title,
                            "path": " / ".join(trail + [node.title or "(page)"]),
                        }
                    )
                    continue
                abs_path = _safe_docs_path(safe_rel)
            except Exception as exc:
                errors.append(
                    {
                        "type": "illegal_path",
                        "file": file_rel,
                        "title": node.title,
                        "path": " / ".join(trail + [node.title or "(page)"]),
                        "error": str(exc),
                    }
                )
                continue

            if not abs_path.exists() or not abs_path.is_file():
                errors.append(
                    {
                        "type": "missing_file",
                        "file": file_rel,
                        "title": node.title,
                        "path": " / ".join(trail + [node.title or "(page)"]),
                    }
                )

    walk(nodes, [])
    return errors, warnings


def _safe_docs_path(rel: str) -> Path:
    rel = _safe_rel_path(rel)
    path = (DOCS_ROOT / rel).resolve()
    docs = DOCS_ROOT.resolve()
    if docs not in path.parents and path != docs:
        raise ValueError("Path escapes docs root.")
    return path


def _rename_path(old_path: Path, new_path: Path) -> None:
    if old_path == new_path:
        return
    if new_path.exists():
        try:
            if old_path.samefile(new_path):
                if old_path.name == new_path.name:
                    return
                tmp = old_path.parent / f".rename-{uuid4().hex}"
                while tmp.exists():
                    tmp = old_path.parent / f".rename-{uuid4().hex}"
                old_path.rename(tmp)
                tmp.rename(new_path)
                return
        except OSError:
            pass
        raise FileExistsError("Target already exists.")
    old_path.rename(new_path)


def _open_with_default_viewer(path: Path) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", str(path)])
        return
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen(["xdg-open", str(path)])


def _open_with_vscode(path: Path) -> bool:
    code = shutil.which("code")
    if code:
        subprocess.Popen([code, str(path)])
        return True
    if sys.platform.startswith("darwin"):
        try:
            subprocess.Popen(["open", "-a", "Visual Studio Code", str(path)])
            return True
        except FileNotFoundError:
            return False
    return False


def _open_in_editor_or_default(path: Path) -> str:
    if _open_with_vscode(path):
        return "vscode"
    _open_with_default_viewer(path)
    return "default"


@app.route("/api/rename_file", methods=["POST"])
def api_rename_file():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    old_file = str(payload.get("old_file") or "").strip()
    new_basename = str(payload.get("new_basename") or "").strip()
    if not old_file:
        return _json_error("`old_file` is required.", 400, old_file=old_file)
    if not new_basename:
        return _json_error("`new_basename` is required.", 400, old_file=old_file, new_basename=new_basename)
    if "/" in new_basename or "\\" in new_basename:
        return _json_error("Filename must not include directories.", 400, old_file=old_file, new_basename=new_basename)

    old_rel = _safe_rel_path(old_file)
    try:
        old_path = _safe_docs_path(old_rel)
    except Exception as exc:
        return _json_error(str(exc), 400, old_file=old_file, new_basename=new_basename)
    if not old_path.exists() or not old_path.is_file():
        return _json_error("Old file not found.", 404, old_file=old_file, new_basename=new_basename)

    # Keep extension unless user specifies one.
    base = new_basename
    if "." not in Path(base).name:
        base = f"{base}{old_path.suffix}"
    if old_path.suffix.lower() == ".md" and not base.lower().endswith(".md"):
        base = f"{base}.md"

    new_path = (old_path.parent / base).resolve()
    docs = DOCS_ROOT.resolve()
    if docs not in new_path.parents and new_path != docs:
        return _json_error("Path escapes docs root.", 400, old_file=old_file, new_basename=new_basename)
    try:
        _rename_path(old_path, new_path)
    except FileExistsError:
        return _json_error("Target already exists.", 409, old_file=old_file, new_basename=new_basename)
    except Exception as exc:
        return _json_error(f"Rename failed: {exc}", 500, old_file=old_file, new_basename=new_basename)

    return jsonify({"status": "ok", "file": new_path.relative_to(DOCS_ROOT).as_posix()})


@app.route("/api/rename_dir", methods=["POST"])
def api_rename_dir():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    dir_rel = str(payload.get("dir") or "").strip().strip("/")
    new_segment = str(payload.get("new_segment") or "").strip().strip("/")
    if not dir_rel:
        return _json_error("`dir` is required.", 400, dir=dir_rel)
    if not new_segment:
        return _json_error("`new_segment` is required.", 400, dir=dir_rel, new_segment=new_segment)
    if "/" in new_segment or "\\" in new_segment:
        return _json_error("Invalid segment.", 400, dir=dir_rel, new_segment=new_segment)

    old_rel = _safe_rel_path(dir_rel)
    try:
        old_path = _safe_docs_path(old_rel)
    except Exception as exc:
        return _json_error(str(exc), 400, dir=dir_rel, new_segment=new_segment)
    if not old_path.exists() or not old_path.is_dir():
        return _json_error("Directory not found.", 404, dir=dir_rel, new_segment=new_segment)

    parent = old_path.parent
    new_path = (parent / new_segment).resolve()
    docs = DOCS_ROOT.resolve()
    if docs not in new_path.parents and new_path != docs:
        return _json_error("Path escapes docs root.", 400, dir=dir_rel, new_segment=new_segment)
    try:
        _rename_path(old_path, new_path)
    except FileExistsError:
        return _json_error("Target already exists.", 409, dir=dir_rel, new_segment=new_segment)
    except Exception as exc:
        return _json_error(f"Rename failed: {exc}", 500, dir=dir_rel, new_segment=new_segment)

    return jsonify(
        {
            "status": "ok",
            "segment": new_segment,
            "dir": new_path.relative_to(DOCS_ROOT).as_posix(),
        }
    )


@app.route("/api/preflight_renames", methods=["POST"])
def api_preflight_renames():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    dirs = payload.get("dirs") if isinstance(payload.get("dirs"), list) else []
    files = payload.get("files") if isinstance(payload.get("files"), list) else []

    errors: list[dict[str, Any]] = []
    seen_targets: set[str] = set()

    for item in dirs:
        if not isinstance(item, dict):
            continue
        dir_rel = str(item.get("dir") or "").strip().strip("/")
        new_segment = str(item.get("new_segment") or "").strip().strip("/")
        if not dir_rel or not new_segment:
            errors.append({"type": "invalid_dir_plan", "dir": dir_rel, "new_segment": new_segment})
            continue
        if "/" in new_segment or "\\" in new_segment:
            errors.append({"type": "invalid_segment", "dir": dir_rel, "new_segment": new_segment})
            continue
        try:
            old_path = _safe_docs_path(_safe_rel_path(dir_rel))
        except Exception as exc:
            errors.append({"type": "invalid_dir_path", "dir": dir_rel, "new_segment": new_segment, "error": str(exc)})
            continue
        if not old_path.exists() or not old_path.is_dir():
            errors.append({"type": "missing_dir", "dir": dir_rel, "new_segment": new_segment})
            continue
        new_path = (old_path.parent / new_segment).resolve()
        rel_target = new_path.relative_to(DOCS_ROOT).as_posix()
        if rel_target in seen_targets:
            errors.append({"type": "duplicate_target", "target": rel_target, "dir": dir_rel, "new_segment": new_segment})
            continue
        seen_targets.add(rel_target)
        if new_path.exists() and not _same_file_path(old_path, new_path):
            errors.append({"type": "target_exists", "target": rel_target, "dir": dir_rel, "new_segment": new_segment})

    for item in files:
        if not isinstance(item, dict):
            continue
        old_file = str(item.get("old_file") or "").strip()
        new_basename = str(item.get("new_basename") or "").strip()
        dir_after = str(item.get("dir_after") or "").strip().strip("/")
        if not old_file or not new_basename:
            errors.append({"type": "invalid_file_plan", "old_file": old_file, "new_basename": new_basename, "dir_after": dir_after})
            continue
        if "/" in new_basename or "\\" in new_basename:
            errors.append({"type": "invalid_basename", "old_file": old_file, "new_basename": new_basename, "dir_after": dir_after})
            continue
        try:
            old_path = _safe_docs_path(_safe_rel_path(old_file))
        except Exception as exc:
            errors.append({"type": "invalid_file_path", "old_file": old_file, "new_basename": new_basename, "dir_after": dir_after, "error": str(exc)})
            continue
        if not old_path.exists() or not old_path.is_file():
            errors.append({"type": "missing_file", "old_file": old_file, "new_basename": new_basename, "dir_after": dir_after})
            continue
        base = new_basename
        if "." not in Path(base).name:
            base = f"{base}{old_path.suffix}"
        if old_path.suffix.lower() == ".md" and not base.lower().endswith(".md"):
            base = f"{base}.md"
        target_parent = old_path.parent
        if dir_after:
            try:
                target_parent = _safe_docs_path(_safe_rel_path(dir_after))
            except Exception:
                target_parent = old_path.parent
        new_path = (target_parent / base).resolve()
        rel_target = new_path.relative_to(DOCS_ROOT).as_posix()
        if rel_target in seen_targets:
            errors.append({"type": "duplicate_target", "target": rel_target, "old_file": old_file, "new_basename": new_basename, "dir_after": dir_after})
            continue
        seen_targets.add(rel_target)
        if new_path.exists() and not _same_file_path(old_path, new_path):
            errors.append({"type": "target_exists", "target": rel_target, "old_file": old_file, "new_basename": new_basename, "dir_after": dir_after})

    return jsonify({"status": "ok", "errors": errors})


@app.route("/api/create_page_with_folder", methods=["POST"])
def api_create_page_with_folder():
    """Create a single page in nav, but place the file inside a same-named folder on disk."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    title = str(payload.get("title") or "").strip()
    parent_dir = str(payload.get("parent_dir") or "").strip().strip("/")
    if not title:
        return _json_error("`title` is required.", 400)

    folder_seg = _snakeify(title)
    file_base = f"{folder_seg}.md"

    folder_rel = f"{parent_dir}/{folder_seg}" if parent_dir else folder_seg
    file_rel = f"{folder_rel}/{file_base}"
    try:
        folder_rel = _safe_rel_path(folder_rel)
        file_rel = _safe_rel_path(file_rel)
    except Exception as exc:
        return _json_error(str(exc), 400)

    folder_path = _safe_docs_path(folder_rel)
    file_path = _safe_docs_path(file_rel)
    if file_path.exists():
        return _json_error("Target already exists.", 409)

    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"# {title}\n", encoding="utf-8")
    except Exception as exc:
        return _json_error(f"Create page failed: {exc}", 500)

    return jsonify({"status": "ok", "file": file_rel, "dir": folder_rel})


def _page_map(nodes: list[Node]) -> dict[str, Node]:
    out: dict[str, Node] = {}
    for p in _flatten_pages(nodes):
        if p.id:
            out[p.id] = p
    return out


@app.route("/api/apply_history_step", methods=["POST"])
def api_apply_history_step():
    """Apply an undo/redo step with minimal filesystem changes."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)

    before_raw = payload.get("before_tree")
    after_raw = payload.get("after_tree")
    if not isinstance(before_raw, list) or not isinstance(after_raw, list):
        return _json_error("`before_tree` and `after_tree` must be lists.", 400)

    try:
        before_nodes = [_node_from_dict(item) for item in before_raw]
        after_nodes = [_node_from_dict(item) for item in after_raw]
    except Exception as exc:
        return _json_error(f"Invalid tree: {exc}", 400)

    before_pages = _page_map(before_nodes)
    after_pages = _page_map(after_nodes)

    moves: list[tuple[Path, Path]] = []
    asset_moves: list[tuple[Path, Path]] = []
    deletes: list[Path] = []
    asset_deletes: list[Path] = []
    creates: list[str] = []
    collisions: list[dict[str, Any]] = []

    planned_sources: set[Path] = set()
    planned_targets: set[Path] = set()

    # Compute deletes (pages removed in target tree).
    for pid, before_node in before_pages.items():
        if pid in after_pages:
            continue
        if not before_node.file:
            continue
        rel = _safe_rel_path(before_node.file)
        src = DOCS_ROOT / rel
        deletes.append(src)
        assets = src.with_suffix("")
        if assets.exists() and assets.is_dir():
            asset_deletes.append(assets)

    # Compute creates (pages added in target tree).
    for pid, after_node in after_pages.items():
        if pid in before_pages:
            continue
        if not after_node.file:
            continue
        rel = _safe_rel_path(after_node.file)
        creates.append(rel)

    # Compute moves for pages whose file changed.
    for pid, after_node in after_pages.items():
        if pid not in before_pages:
            continue
        before_node = before_pages[pid]
        before_file = (before_node.file or "").strip()
        after_file = (after_node.file or "").strip()
        if not before_file or not after_file:
            continue
        if _normalize_rel(before_file) == _normalize_rel(after_file):
            continue
        src_rel = _safe_rel_path(before_file)
        dst_rel = _safe_rel_path(after_file)
        src = DOCS_ROOT / src_rel
        dst = DOCS_ROOT / dst_rel
        if not src.exists():
            # Nothing to move; allow nav/state to update anyway.
            continue
        moves.append((src, dst))
        planned_sources.add(src.resolve())
        planned_targets.add(dst.resolve())
        src_assets = src.with_suffix("")
        dst_assets = dst.with_suffix("")
        if src_assets.exists() and src_assets.is_dir():
            asset_moves.append((src_assets, dst_assets))
            planned_sources.add(src_assets.resolve())
            planned_targets.add(dst_assets.resolve())

    # Validate collisions.
    for _, dst in moves + asset_moves:
        if dst.exists() and dst.resolve() not in planned_sources:
            collisions.append({"type": "exists", "target": dst.relative_to(DOCS_ROOT).as_posix()})

    if collisions:
        return _json_error("Sync blocked by file conflict.", 409, collisions=collisions)

    # Apply filesystem changes (only what changed).
    try:
        if deletes or asset_deletes:
            for p in asset_deletes:
                try:
                    # Directory must be empty to remove.
                    p.rmdir()
                except OSError:
                    pass
            for p in deletes:
                try:
                    p.unlink()
                except OSError:
                    pass
        moved_pages = _apply_moves_with_temps(moves) if moves else []
        moved_assets = _apply_moves_with_temps(asset_moves) if asset_moves else []
        for rel in creates:
            # Create only if missing (no overwrite).
            abs_path = DOCS_ROOT / rel
            if abs_path.exists():
                return _json_error("Sync blocked by file conflict.", 409, collisions=[{"type": "exists", "target": rel}])
            _create_missing_page(rel, Path(rel).stem)
    except FileExistsError as exc:
        return _json_error("Sync blocked by file conflict.", 409, collisions=[{"type": "exists", "target": str(exc)}])
    except Exception as exc:
        return _json_error(f"File sync failed: {exc}", 500)

    nav_value = _tree_to_mkdocs_nav(after_nodes)
    try:
        backup = _write_mkdocs_nav_only(MKDOCS_PATH, nav_value)
    except Exception as exc:
        return _json_error(f"Failed to write mkdocs.yml nav: {exc}", 500)

    _save_state(after_nodes)
    return jsonify(
        {
            "status": "ok",
            "backup": str(backup) if backup else None,
            "applied": {
                "moves": moved_pages,
                "assets": moved_assets,
                "created": creates,
                "deleted": [p.relative_to(DOCS_ROOT).as_posix() for p in deletes if p.exists() is False],
            },
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
    segment = _snakeify(desired or title)
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
    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        return _json_error("`content` must be a string.", 400)
    parent_dir = str(payload.get("parent_dir") or "").strip().strip("/")
    parent_path = DOCS_ROOT if not parent_dir else _safe_docs_path(parent_dir)
    if parent_path.exists() and not parent_path.is_dir():
        return _json_error("Parent path is not a directory.", 400)
    parent_path.mkdir(parents=True, exist_ok=True)

    desired = str(payload.get("basename") or "").strip()
    if desired and ("/" in desired or "\\" in desired):
        return _json_error("Invalid filename.", 400)
    base = desired or f"{_snakeify(title)}.md"
    if not base.lower().endswith(".md"):
        base = f"{base}.md"
    base = _unique_child_name(parent_path, base)
    file_path = parent_path / base
    if isinstance(content, str) and content.strip():
        body = content
    elif base.lower() == "readme.md":
        section_title = str(payload.get("section_title") or "").strip()
        section_name = section_title or Path(parent_dir).name or "Section"
        body = _section_readme_template(section_name)
    else:
        body = f"# {title.strip() or 'Untitled'}\n"
    file_path.write_text(body, encoding="utf-8")
    rel = file_path.relative_to(DOCS_ROOT).as_posix()
    return jsonify({"status": "ok", "file": rel})


@app.route("/api/open_file", methods=["POST"])
def api_open_file():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    file_rel = str(payload.get("file") or "").strip()
    if not file_rel:
        return _json_error("`file` is required.", 400)
    try:
        safe_rel = _safe_rel_path(file_rel)
        if Path(safe_rel).suffix.lower() not in ALLOWED_DOC_EXTS:
            return _json_error("Only markdown files can be opened.", 400)
        abs_path = _safe_docs_path(safe_rel)
    except Exception as exc:
        return _json_error(str(exc), 400)
    if not abs_path.exists() or not abs_path.is_file():
        return _json_error("File not found.", 404)
    try:
        mode = _open_in_editor_or_default(abs_path)
    except Exception as exc:
        return _json_error(f"Open failed: {exc}", 500)
    return jsonify({"status": "ok", "mode": mode})


@app.route("/api/read_file", methods=["POST"])
def api_read_file():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    file_rel = str(payload.get("file") or "").strip()
    if not file_rel:
        return _json_error("`file` is required.", 400)
    try:
        safe_rel = _safe_rel_path(file_rel)
        if Path(safe_rel).suffix.lower() not in ALLOWED_DOC_EXTS:
            return _json_error("Only markdown files can be read.", 400)
        abs_path = _safe_docs_path(safe_rel)
    except Exception as exc:
        return _json_error(str(exc), 400)
    if not abs_path.exists() or not abs_path.is_file():
        return _json_error("File not found.", 404)
    try:
        content = abs_path.read_text(encoding="utf-8")
    except Exception as exc:
        return _json_error(f"Read failed: {exc}", 500)
    return jsonify({"status": "ok", "file": safe_rel, "content": content})


@app.route("/api/write_file", methods=["POST"])
def api_write_file():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    file_rel = str(payload.get("file") or "").strip()
    if not file_rel:
        return _json_error("`file` is required.", 400)
    content = payload.get("content")
    if not isinstance(content, str):
        return _json_error("`content` must be a string.", 400)
    try:
        safe_rel = _safe_rel_path(file_rel)
        if Path(safe_rel).suffix.lower() not in ALLOWED_DOC_EXTS:
            return _json_error("Only markdown files can be written.", 400)
        abs_path = _safe_docs_path(safe_rel)
    except Exception as exc:
        return _json_error(str(exc), 400)
    if not abs_path.exists() or not abs_path.is_file():
        return _json_error("File not found.", 404)
    try:
        abs_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return _json_error(f"Write failed: {exc}", 500)
    return jsonify({"status": "ok", "file": safe_rel})


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


@app.route("/api/icons_index", methods=["GET"])
def api_icons_index():
    try:
        data = _load_iconsearch_index()
    except FileNotFoundError:
        return _json_error(
            f"Missing iconsearch_index.json at {ICONSEARCH_INDEX_PATH}.",
            404,
        )
    except Exception as exc:
        return _json_error(f"Failed to load iconsearch_index.json: {exc}", 500)
    return jsonify(data)


@app.route("/api/mkdocs/status", methods=["GET"])
def api_mkdocs_status():
    return jsonify(_mkdocs_status())


@app.route("/api/mkdocs/check", methods=["POST"])
def api_mkdocs_check():
    data = request.get_json(silent=True) or {}
    url = data.get("url") if isinstance(data, dict) else None
    if not isinstance(url, str) or not url.strip():
        return _json_error("`url` is required.", 400)
    url = url.strip()
    if not _is_allowed_local_url(url):
        return _json_error("URL not allowed.", 400)
    reachable, status = _probe_url(url)
    return jsonify({"reachable": reachable, "status": status})


@app.route("/api/mkdocs/monitor", methods=["GET"])
def api_mkdocs_monitor():
    return jsonify(_mkdocs_monitor())


@app.route("/api/mkdocs/stream", methods=["GET"])
def api_mkdocs_stream():
    q: queue.Queue[str] = queue.Queue(maxsize=200)
    with MKDOCS_LOCK:
        MKDOCS_SUBSCRIBERS.append(q)

    def _event_stream():
        try:
            yield f"data: {json.dumps(_mkdocs_snapshot(), ensure_ascii=True)}\n\n"
            while True:
                try:
                    chunk = q.get(timeout=15)
                    yield chunk
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with MKDOCS_LOCK:
                if q in MKDOCS_SUBSCRIBERS:
                    MKDOCS_SUBSCRIBERS.remove(q)

    return Response(stream_with_context(_event_stream()), mimetype="text/event-stream")


@app.route("/api/mkdocs/start", methods=["POST"])
def api_mkdocs_start():
    return jsonify(_mkdocs_start())


@app.route("/api/mkdocs/stop", methods=["POST"])
def api_mkdocs_stop():
    return jsonify(_mkdocs_stop())


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
    _ensure_section_overviews_in_place(nodes)
    errors, warnings = _validate_imported_tree(nodes)
    if errors:
        # Best-effort: show the offending mkdocs.yml lines to help users fix quickly.
        lines: list[dict[str, Any]] = []
        try:
            raw_lines = MKDOCS_PATH.read_text(encoding="utf-8").splitlines()
            needles = []
            for err in errors:
                if isinstance(err, dict) and isinstance(err.get("file"), str) and err["file"].strip():
                    needles.append(err["file"].strip())
            needles = list(dict.fromkeys(needles))
            for i, line in enumerate(raw_lines, start=1):
                for needle in needles:
                    if needle in line:
                        lines.append({"no": i, "text": line})
                        break
        except Exception:
            lines = []
        return _json_error(
            "Import blocked: mkdocs.yml contains illegal doc paths.",
            409,
            errors=errors,
            lines=lines,
        )
    _save_state(nodes)
    return jsonify({"status": "ok", "tree": [n.to_dict() for n in nodes], "warnings": warnings})


@app.route("/api/source", methods=["GET"])
def api_source():
    return jsonify(_build_source_tree(DOCS_ROOT, DOCS_ROOT))


@app.route("/api/ui_state", methods=["GET"])
def api_get_ui_state():
    return jsonify(_load_ui_state())


@app.route("/api/ui_state", methods=["POST"])
def api_save_ui_state():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    _save_ui_state(payload)
    return jsonify({"status": "ok"})


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
        else:
            nodes = _load_state()
    except Exception as exc:
        return _json_error(f"Failed to load state: {exc}", 500)

    warnings: list[dict[str, Any]] = []
    moves_applied: list[dict[str, str]] = []

    if mode == "sync_files":
        preserve_page_dirs = bool(payload.get("preserve_page_dirs", False))
        page_moves, asset_moves, plan_warnings, collisions = _plan_file_sync(
            nodes, create_missing=create_missing, preserve_page_dirs=preserve_page_dirs
        )
        warnings.extend(plan_warnings)
        if collisions:
            return _json_error("File move blocked: target already exists.", 409, collisions=collisions)
        preflight_errors = _preflight_moves(page_moves + asset_moves)
        if preflight_errors:
            return _json_error("File move blocked: preflight failed.", 409, errors=preflight_errors)
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
        # Do not persist state until after mkdocs.yml is written successfully.

    # Always write mkdocs.yml nav from current state, but only replace the nav block.
    nav_value = _tree_to_mkdocs_nav(nodes)
    try:
        backup = _write_mkdocs_nav_only(MKDOCS_PATH, nav_value)
    except Exception as exc:
        return _json_error(f"Failed to write mkdocs.yml nav: {exc}", 500)

    # Only persist state after mkdocs write succeeds.
    _save_state(nodes)

    return jsonify(
        {
            "status": "ok",
            "mode": mode,
            "moves": moves_applied,
            "warnings": warnings,
            "backup": str(backup) if backup else None,
        }
    )


@app.route("/api/sync_drag_files", methods=["POST"])
def api_sync_drag_files():
    """Incremental sync for drag operations: move only dragged pages/folders on disk, then write nav."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Expected a JSON object.", 400)
    tree_raw = payload.get("tree")
    moved_ids_raw = payload.get("moved_ids")
    if not isinstance(tree_raw, list):
        return _json_error("Expected `tree` to be a JSON list.", 400)
    if not isinstance(moved_ids_raw, list) or not all(isinstance(x, str) for x in moved_ids_raw):
        return _json_error("Expected `moved_ids` to be a JSON list of strings.", 400)

    try:
        nodes = [_node_from_dict(item) for item in tree_raw]
    except Exception as exc:
        return _json_error(f"Invalid tree: {exc}", 400)

    moved_ids = {x for x in moved_ids_raw if x}
    page_moves, asset_moves, collisions = _plan_partial_sync(nodes, moved_ids)
    if collisions:
        return _json_error("File move blocked: target already exists.", 409, collisions=collisions)
    preflight_errors = _preflight_moves(page_moves + asset_moves)
    if preflight_errors:
        return _json_error("File move blocked: preflight failed.", 409, errors=preflight_errors)

    moves_applied: list[dict[str, str]] = []
    try:
        moves_applied.extend(_apply_moves(page_moves))
        moves_applied.extend(_apply_moves(asset_moves))
    except (FileNotFoundError, FileExistsError) as exc:
        return _json_error(f"File move failed: {exc}", 400)
    except Exception as exc:
        return _json_error(f"File move failed: {exc}", 500)

    desired_dirs = _collect_desired_folder_dirs(nodes)
    _ensure_desired_folder_dirs_exist(desired_dirs)

    nav_value = _tree_to_mkdocs_nav(nodes)
    try:
        backup = _write_mkdocs_nav_only(MKDOCS_PATH, nav_value)
    except Exception as exc:
        return _json_error(f"Failed to write mkdocs.yml nav: {exc}", 500)

    _save_state(nodes)
    return jsonify({"status": "ok", "moves": moves_applied, "backup": str(backup) if backup else None})


if __name__ == "__main__":
    atexit.register(_shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    app.run(debug=True, port=5001)
