#!/usr/bin/env python3
"""
Lawra HTML ingestion — SSO Print-HTML, DOM-aware.

Built from real SSO structure analysis (Highway Code + Act).

Confirmed SSO Print-HTML structure
────────────────────────────────────
  div#tocView              All legislation content lives here.
  td.part[id="P1I-"]       Part container (e.g., Part I, Part II)
    td.partHdr             Part title inside the above
  div.prov1                One provision / paragraph
    td.prov1Hdr            Optional sub-heading inside the prov1 div
    td.prov1Txt            Provision body; number in <strong> at the start
  div.amendNote            Amendment note [S xxx/yyyy wef dd/mm/yyyy]
  blockquote.TocParagraph  Table of Contents entries (ignored automatically)

Chunking mode (auto-detected from average provision length, overridable via manifest)
  "section"  long provisions (Acts / Rules)   → one chunk per section
  "grouped"  short provisions (Highway Code)  → one chunk per Part

How to get your HTML files from SSO
────────────────────────────────────
  1. Open e.g. https://sso.agc.gov.sg/SL/RTA1961-R11 in your browser
  2. Wait for full load → Print panel → Select All → Print — HTML
  3. New clean tab opens with just the legislation text
  4. Ctrl+S → save as "Webpage, HTML Only" (.html) → html/

Requirements:
    pip install beautifulsoup4 lxml requests qdrant-client

Workflow:
    python lawra_ingest_html.py --html-dir ./html --init-manifest manifest.json
    # fill manifest: title, source_url, version_date, doc_type, label_prefix
    # doc_type: "Act" | "Rules" | "Code"   label_prefix: "s" | "r" | "para"

    python lawra_ingest_html.py --html-dir ./html --manifest manifest.json --inspect
    python lawra_ingest_html.py --html-dir ./html --manifest manifest.json --dry-run
    python lawra_ingest_html.py --html-dir ./html --manifest manifest.json \\
        --embed-model bge-m3 --collection lawra_rta
"""
from __future__ import annotations

import argparse, json, os, re, sys, uuid
from dataclasses import dataclass, asdict
from itertools import groupby
from pathlib import Path
from typing import Optional

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_URL  = os.environ.get("OLLAMA_URL",        "http://localhost:11434")
DEFAULT_QDRANT_URL  = os.environ.get("QDRANT_URL",        "http://localhost:6333")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL",       "bge-m3")
DEFAULT_COLLECTION  = os.environ.get("QDRANT_COLLECTION", "lawra")

TARGET_CHARS  = 2000
OVERLAP_CHARS = 200
MIN_FRAG_LEN  = 20

LAWRA_NS = uuid.UUID("6f1d2c4a-0000-4000-8000-000000000001")

# Strip inline amendment notes from provision text
_AMEND_RE = re.compile(r"\[S\s*\d+/\d+[^\]]*\]", re.I)

# Auto-detect mode: avg provision length above this → "section", below → "grouped"
_SECTION_LEN_THRESHOLD = 300


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class DocMeta:
    doc_id:       str
    title:        str
    source_url:   str  = ""
    version_date: str  = ""
    doc_type:     str  = ""
    label_prefix: str  = "s"

    @property
    def attribution(self) -> str:
        return (
            f"Source: Singapore Statutes Online (SSO), {self.title}"
            + (f", version as at {self.version_date}" if self.version_date else "")
            + ". Reproduced with the permission of AGC. UNOFFICIAL consolidation; "
              "authoritative text is the Government Gazette. "
              "Verify at SSO"
            + (f": {self.source_url}" if self.source_url else ".")
        )


@dataclass
class ProvisionRaw:
    number:  str
    text:    str
    heading: str = ""
    part:    str = ""


@dataclass
class Chunk:
    id:              str
    text:            str
    doc_id:          str
    title:           str
    doc_type:        str
    section_label:   str
    section_heading: str
    chunk_mode:      str
    chunk_index:     int
    provisions:      list
    source_url:      str
    version_date:    str
    attribution:     str


# ── SSO DOM parser ─────────────────────────────────────────────────────────────
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_sso_html(html_path: Path) -> tuple[list[ProvisionRaw], str]:
    """
    Parse SSO Print-HTML. Returns (provisions, mode).

    Key lessons from real-file analysis:
    - ALL content is inside div#tocView — never kill elements by id-regex
    - td.partHdr is inside td.part, NOT inside div.prov1
    - td.prov1Hdr IS inside div.prov1 (optional italic sub-heading)
    - Number is in <strong> at the start of td.prov1Txt
    - Never modify the DOM while iterating — extract by string operations
    """
    from bs4 import BeautifulSoup

    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    # ── Safe-only cleanup: scripts/styles + known harmless UI chrome ──────
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    # Known SSO chrome IDs (NOT tocView which is the content)
    for safe_id in ("helpModal", "searchPhraseTypePopover",
                    "searchLegisTypePopover"):
        el = soup.find(id=safe_id)
        if el:
            el.decompose()
    # SSO app-config div (just JSON data, no legislation)
    for el in soup.find_all("div", class_="global-vars"):
        el.decompose()
    for el in soup.find_all("div", class_="print-header"):
        el.decompose()

    # ── Locate content root ───────────────────────────────────────────────
    # SSO Print-HTML puts all legislation inside div#tocView
    content = soup.find(id="tocView") or soup.find("body") or soup

    # ── Walk partHdr + prov1 in document order ───────────────────────────
    # find_all with a lambda returns ALL matching descendants in doc order.
    # We want: td.partHdr  and  div.prov1 (but not div.prov1Rep wrappers)
    def _match(el) -> bool:
        if not hasattr(el, "name"):
            return False
        cls = el.get("class") or []
        if el.name == "td"  and "partHdr" in cls:
            return True
        if el.name == "div" and "prov1" in cls and "prov1Rep" not in cls:
            return True
        return False

    elements = content.find_all(_match)

    # ── Extract provisions ────────────────────────────────────────────────
    provisions: list[ProvisionRaw] = []
    current_part = ""

    for el in elements:
        # ── Part header ──
        if el.name == "td" and "partHdr" in (el.get("class") or []):
            current_part = _clean(el.get_text())
            continue

        # ── Provision div ──
        # Optional sub-heading inside the prov1 div
        hdr_el = el.find(class_="prov1Hdr")
        heading = _clean(hdr_el.get_text()) if hdr_el else ""

        # Provision body text
        txt_td = el.find("td", class_="prov1Txt")
        if not txt_td:
            continue

        # Number is in <strong> at the very start of prov1Txt
        strong = txt_td.find("strong")
        number = _clean(strong.get_text()) if strong else ""

        # Get full text via string extraction (no DOM mutation)
        full = _clean(txt_td.get_text(separator=" "))

        # Strip the number prefix that appears at the start
        body = full
        if number and body.startswith(number):
            body = body[len(number):].strip()

        # Strip amendment notes e.g. [S 3173/2019 wef 01/12/2019]
        body = _AMEND_RE.sub("", body).strip()
        body = re.sub(r"\s+", " ", body).strip()

        if len(body) < MIN_FRAG_LEN:
            continue

        provisions.append(ProvisionRaw(
            number  = number,
            text    = body,
            heading = heading,
            part    = current_part,
        ))

    # ── Auto-detect chunking mode ─────────────────────────────────────────
    if provisions:
        avg_len = sum(len(p.text) for p in provisions) / len(provisions)
        mode = "section" if avg_len > _SECTION_LEN_THRESHOLD else "grouped"
    else:
        mode = "section"

    return provisions, mode


# ── Chunking ──────────────────────────────────────────────────────────────────
def _split_long(text: str, target: int, overlap: int) -> list[str]:
    if len(text) <= target:
        return [text]
    chunks, buf = [], ""
    for para in re.split(r"\n\s*\n|\s{2,}", text):
        if len(buf) + len(para) + 1 <= target:
            buf = (buf + " " + para).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= target:
                buf = para
            else:
                for i in range(0, len(para), target - overlap):
                    chunks.append(para[i:i + target])
                buf = ""
    if buf:
        chunks.append(buf)
    if overlap and len(chunks) > 1:
        stitched = [chunks[0]]
        for i in range(1, len(chunks)):
            stitched.append((chunks[i-1][-overlap:] + " " + chunks[i]).strip())
        chunks = stitched
    return [c for c in chunks if c]


def _cid(doc_id: str, label: str, idx: int) -> str:
    return str(uuid.uuid5(LAWRA_NS, f"{doc_id}|{label}|{idx}"))


def build_chunks_section(meta: DocMeta,
                         provisions: list[ProvisionRaw]) -> list[Chunk]:
    """Acts / Rules: one chunk per provision, sub-chunked if long.

    The section context header is prepended to EVERY sub-chunk so retrieval
    of any fragment still carries its doc/section anchor — critical for long
    definition sections (e.g. s 2 Interpretation) that split 10+ ways.
    """
    chunks = []
    for p in provisions:
        label  = f"{meta.label_prefix} {p.number.rstrip('.')}" if p.number else "(body)"
        header = f"[{meta.title} — {label}"
        header += f": {p.heading}]\n" if p.heading else "]\n"
        for idx, piece in enumerate(_split_long(p.text, TARGET_CHARS, OVERLAP_CHARS)):
            piece = header + piece
            chunks.append(Chunk(
                id=_cid(meta.doc_id, label, idx),
                text=piece, doc_id=meta.doc_id, title=meta.title,
                doc_type=meta.doc_type, section_label=label,
                section_heading=p.heading, chunk_mode="section",
                chunk_index=idx, provisions=[p.number],
                source_url=meta.source_url, version_date=meta.version_date,
                attribution=meta.attribution,
            ))
    return chunks


def build_chunks_grouped(meta: DocMeta,
                         provisions: list[ProvisionRaw]) -> list[Chunk]:
    """Highway Code: group provisions by Part, one chunk per Part."""
    chunks = []
    for part, grp in groupby(provisions, key=lambda p: p.part):
        grp = list(grp)
        lines = []
        for p in grp:
            line = p.number + " " if p.number else ""
            if p.heading:
                line += f"[{p.heading}] "
            line += p.text
            lines.append(line.strip())
        body_text = "\n".join(lines)
        label  = part or "(preamble)"
        header = f"[{meta.title} — {label}]\n"
        full   = header + body_text
        provs  = [p.number for p in grp]
        for idx, piece in enumerate(_split_long(full, TARGET_CHARS, OVERLAP_CHARS)):
            chunks.append(Chunk(
                id=_cid(meta.doc_id, label, idx),
                text=piece, doc_id=meta.doc_id, title=meta.title,
                doc_type=meta.doc_type, section_label=label,
                section_heading=label, chunk_mode="grouped",
                chunk_index=idx, provisions=provs,
                source_url=meta.source_url, version_date=meta.version_date,
                attribution=meta.attribution,
            ))
    return chunks


def process_file(html_path: Path, meta: DocMeta) -> list[Chunk]:
    provisions, auto_mode = parse_sso_html(html_path)
    # doc_type hint overrides auto-detection
    if meta.doc_type.lower() in ("act", "rules"):
        mode = "section"
    elif meta.doc_type.lower() in ("code",):
        mode = "grouped"
    else:
        mode = auto_mode

    chunks = (build_chunks_section(meta, provisions) if mode == "section"
              else build_chunks_grouped(meta, provisions))
    print(f"  {html_path.name}: mode={mode!r}  "
          f"{len(provisions)} provisions → {len(chunks)} chunks")
    return chunks


# ── Manifest ──────────────────────────────────────────────────────────────────
def load_manifest(path: Path) -> dict[str, DocMeta]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        fname: DocMeta(
            doc_id       = m.get("doc_id") or Path(fname).stem,
            title        = m.get("title") or Path(fname).stem,
            source_url   = m.get("source_url", ""),
            version_date = m.get("version_date", ""),
            doc_type     = m.get("doc_type", ""),
            label_prefix = m.get("label_prefix", "s"),
        )
        for fname, m in data.items()
    }


def init_manifest(html_dir: Path, out_path: Path) -> None:
    files = sorted(p.name for p in html_dir.glob("*.html"))
    if not files:
        print(f"No .html files found in {html_dir}", file=sys.stderr); sys.exit(1)
    stub = {
        name: {
            "doc_id": Path(name).stem, "title": "",
            "source_url": "", "version_date": "",
            "doc_type": "",           # "Act" | "Rules" | "Code"
            "label_prefix": "s",      # "s" Act, "r" Rules, "para" Code
        } for name in files
    }
    out_path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    print(f"Wrote manifest stub ({len(files)} entries) → {out_path}")
    print("  doc_type: 'Act' | 'Rules' | 'Code'")
    print("  label_prefix: 's' (Act), 'r' (Rules), 'para' (Code/Highway)")


# ── Inspect ────────────────────────────────────────────────────────────────────
def inspect_file(html_path: Path) -> None:
    from bs4 import BeautifulSoup

    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    prov_count = len(soup.find_all("div",  class_="prov1"))
    part_count = len(soup.find_all("td",   class_="partHdr"))
    hdr_count  = len(soup.find_all(class_="prov1Hdr"))
    has_tocview = bool(soup.find(id="tocView"))

    provisions, mode = parse_sso_html(html_path)

    print(f"\n{'─'*66}")
    print(f"INSPECT: {html_path.name}")
    print(f"{'─'*66}")
    print(f"  div#tocView     : {'found ✓' if has_tocview else 'not found (fallback to body)'}")
    print(f"  div.prov1       : {prov_count}")
    print(f"  td.partHdr      : {part_count}")
    print(f"  prov1Hdr        : {hdr_count}")
    print(f"  Parsed provs    : {len(provisions)}")
    if provisions:
        avg = sum(len(p.text) for p in provisions) / len(provisions)
        print(f"  Avg prov length : {avg:.0f} chars  → mode={mode!r}")
    print()

    if mode == "grouped":
        print("  Parts and counts:")
        for part, grp in groupby(provisions, key=lambda p: p.part):
            print(f"    {len(list(grp)):3d}  {part!r}")
    else:
        print("  First 6 provisions:")
        for p in provisions[:6]:
            print(f"    [{p.number:>5}]  heading={p.heading[:30]!r}  "
                  f"text={p.text[:60]!r}")


# ── Ollama + Qdrant ────────────────────────────────────────────────────────────
def embed_one(text: str, model: str, url: str) -> list[float]:
    import requests
    r = requests.post(f"{url}/api/embeddings",
                      json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def upsert_to_qdrant(chunks: list[Chunk], model: str, collection: str,
                     ollama_url: str, qdrant_url: str, batch: int = 64) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    if not chunks:
        print("No chunks to upsert."); return
    client = QdrantClient(url=qdrant_url)
    print(f"Probing dimension (model={model!r})...")
    dim = len(embed_one(chunks[0].text, model, ollama_url))
    print(f"Dimension = {dim}")
    if collection not in {c.name for c in client.get_collections().collections}:
        client.create_collection(collection_name=collection,
                                  vectors_config=VectorParams(size=dim,
                                                              distance=Distance.COSINE))
        print(f"Created collection '{collection}'.")
    points: list = []
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        payload = asdict(ch); payload.pop("id", None)
        points.append(PointStruct(id=ch.id,
                                   vector=embed_one(ch.text, model, ollama_url),
                                   payload=payload))
        if len(points) >= batch or i == total:
            client.upsert(collection_name=collection, points=points); points = []
        if i % 25 == 0 or i == total:
            print(f"  {i}/{total}")
    print(f"Done — {total} chunks in '{collection}'.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Lawra HTML ingestion — SSO Print-HTML")
    ap.add_argument("--html-dir",      required=True, type=Path)
    ap.add_argument("--manifest",      type=Path)
    ap.add_argument("--init-manifest", type=Path)
    ap.add_argument("--inspect",       action="store_true")
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--out",   type=Path, default=Path("html_chunks_preview.json"))
    ap.add_argument("--embed-model",   default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--collection",    default=DEFAULT_COLLECTION)
    ap.add_argument("--ollama-url",    default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--qdrant-url",    default=DEFAULT_QDRANT_URL)
    args = ap.parse_args()

    if args.init_manifest:
        init_manifest(args.html_dir, args.init_manifest); return

    if args.inspect:
        for f in sorted(args.html_dir.glob("*.html")):
            inspect_file(f)
        return

    if not args.manifest or not args.manifest.exists():
        print("Provide --manifest (or --init-manifest).", file=sys.stderr); sys.exit(1)

    manifest = load_manifest(args.manifest)
    all_chunks: list[Chunk] = []
    for html in sorted(args.html_dir.glob("*.html")):
        meta = manifest.get(html.name)
        if meta is None:
            print(f"  SKIP {html.name}: not in manifest"); continue
        if not meta.title or not meta.source_url:
            print(f"  WARN {html.name}: missing title/source_url")
        all_chunks.extend(process_file(html, meta))

    print(f"\nTotal chunks: {len(all_chunks)}")
    if args.dry_run:
        args.out.write_text(
            json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"Dry run → {args.out}")
        return
    upsert_to_qdrant(all_chunks, args.embed_model, args.collection,
                     args.ollama_url, args.qdrant_url)


if __name__ == "__main__":
    main()
