# 🚗 Lawra — Local Agentic RAG for Singapore Road Traffic Law

> **Lawra** is a fully local, privacy-first agentic Retrieval-Augmented Generation (RAG) system that answers questions about Singapore road traffic legislation — no cloud API keys required.

Accessible via a locally-hosted web app and a Telegram bot at **[@the_lawra_bot](https://t.me/the_lawra_bot)**.

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![llama3.1:8b](https://img.shields.io/badge/LLM-llama3.1%3A8b-blueviolet?style=for-the-badge)
![gemma4:e2b](https://img.shields.io/badge/Judge-gemma4%3Ae2b-orange?style=for-the-badge)
![n8n](https://img.shields.io/badge/n8n-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FF4081?style=for-the-badge&logo=qdrant&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-F97316?style=for-the-badge&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram%20Bot-@the__lawra__bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)

![Lawra Cover Art](_img/github-lawra.png)

---

## Table of Contents

1. [Business Problem](#1-business-problem)
2. [Solution Overview](#2-solution-overview)
3. [Architecture](#3-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Knowledge Base](#5-knowledge-base)
6. [How to Build](#6-how-to-build)
7. [How to Run the System](#7-how-to-run-the-system)
8. [Evaluation](#8-evaluation)
9. [Key Learnings](#9-key-learnings)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Business Problem

Singapore's road traffic legislation is spread across multiple statutes, subsidiary legislation, and codes — all publicly available on the Singapore Statutes Online (SSO) portal, but individually dense and hard to navigate quickly:

| Document                                                   | Coverage                                 |
| ---------------------------------------------------------- | ---------------------------------------- |
| Road Traffic Act 1961                                      | Primary offences, licensing, enforcement |
| Highway Code                                               | Rules of the road for all road users     |
| Active Mobility Act 2017                                   | PMDs, PABs, shared paths                 |
| Motor Vehicles (Third-Party Risks & Compensation) Act 1960 | Compulsory insurance                     |
| Parking Places Act 1974                                    | Parking regulation & LTA authority       |
| Road Traffic (Restriction of Speed on Roads) Notification  | Speed limits by zone                     |

**The problem**: Members of the public, drivers, and road users often have specific, time-sensitive questions ("What is the penalty for drink driving?", "Can I ride my e-scooter here?") but:

- The legislation is verbose and cross-referenced
- SSO search returns document pages, not direct answers
- Consulting a lawyer for basic statutory questions is impractical
- Existing AI assistants (ChatGPT, etc.) hallucinate or cite outdated law

**The need**: A trustworthy, citation-grounded Q&A assistant that answers from the actual legislation text — running locally so sensitive queries never leave the device.

---

## 2. Solution Overview

Lawra is a **local agentic RAG pipeline** that:

1. **Ingests** Singapore road traffic legislation HTML files (downloaded from SSO) into a vector database, preserving section structure and legal citations
2. **Retrieves** the most semantically relevant legal provisions for any user question
3. **Generates** a grounded, cited answer using a local LLM — refusing to answer from general knowledge
4. **Remembers** the conversation context within a session using semantic memory (ChromaDB)
5. **Evaluates** its own quality automatically using RAGAS metrics with a local LLM judge

Users interact through:

- **Web app** — served locally at `http://localhost:3000`
- **Telegram bot** — [@the_lawra_bot](https://t.me/the_lawra_bot) (tunnelled via ngrok to the local n8n instance)

Every answer includes the source legislation and section, and a disclaimer that it is based on an unofficial consolidation.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interfaces                            │
│                                                                     │
│   Web App (localhost:3000)          Telegram Bot (@the_lawra_bot)   │
│   frontend/server.py                n8n Telegram Trigger            │
└───────────────────────────┬─────────────────────────┬──────────────┘
                            │                         │
                            ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    n8n Workflow Engine (port 5678)                  │
│                                                                     │
│  Webhook → Embed Query → Search Qdrant → Format Context             │
│         → Generate Answer (llama3.1:8b) → Format Response          │
└───────────┬───────────────────────────────────┬─────────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌─────────────────────────────────┐
│  Ollama (host:11434)  │         │  Qdrant Vector Store (port 6333)│
│  llama3.1:8b  (LLM)  │         │  Collection: "lawra"            │
│  nomic-embed-text     │         │  ~2,000+ chunks of legislation  │
│  gemma4:e2b (judge)   │         └─────────────────────────────────┘
└───────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              Session Memory (ChromaDB, in-process)                  │
│  Persists Q&A pairs per session · Semantic search over history      │
│  Stored in frontend/.chromadb/                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                 Evaluation Harness (eval/)                          │
│  RAGAS: faithfulness · answer_relevancy · context_precision/recall  │
│  Judge: gemma4:e2b · Golden dataset: 15 curated Q&A pairs          │
└─────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline (n8n workflow)

```
User Question
     │
     ▼
[Embed Query]  →  nomic-embed-text via Ollama
     │
     ▼
[Search Qdrant]  →  cosine similarity, top-5 chunks
     │
     ▼
[Format Context]  →  assemble retrieved text + source metadata + session memory
     │
     ▼
[Generate Answer]  →  llama3.1:8b with strict system prompt
     │                (cite sources, refuse if not in retrieved text)
     ▼
[Format Response]  →  { answer, sources[], trace, disclaimer }
     │
     ▼
Webhook / Telegram Response
```

---

## 4. Tech Stack

| Layer                       | Technology                            | Purpose                                                                |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------------------- |
| **Container orchestration** | Docker + Docker Compose               | Reproducible local deployment of all services                          |
| **Workflow engine**         | [n8n](https://n8n.io)                 | Visual agentic pipeline; handles webhooks, Telegram, HTTP calls        |
| **LLM runtime**             | [Ollama](https://ollama.ai)           | Runs LLMs locally; no GPU cloud required                               |
| **Primary LLM**             | `llama3.1:8b`                         | Generates grounded answers from retrieved legal text                   |
| **LLM Judge**               | `gemma4:e2b`                          | Evaluates answer quality via RAGAS (separate from the answering model) |
| **Embedding model**         | `nomic-embed-text`                    | Encodes queries and legislation chunks into dense vectors              |
| **Vector store**            | [Qdrant](https://qdrant.tech)         | Stores and retrieves legislation chunks by semantic similarity         |
| **Session memory**          | [ChromaDB](https://www.trychroma.com) | In-process semantic memory for multi-turn conversation context         |
| **n8n database**            | PostgreSQL 16                         | Stores n8n workflow state, credentials, and execution history          |
| **Ingestion**               | Python + BeautifulSoup + lxml         | Parses SSO print-HTML, chunks by section, upserts to Qdrant            |
| **Frontend server**         | Python `http.server`                  | Serves the web UI and proxies queries through memory → n8n             |
| **Tunnel**                  | [ngrok](https://ngrok.com)            | Exposes local n8n webhook to Telegram's HTTPS callback                 |
| **Evaluation**              | [RAGAS](https://ragas.io) + LangChain | Automated RAG quality measurement against a golden dataset             |

---

## 5. Knowledge Base

Six Singapore road traffic documents are indexed, downloaded from [Singapore Statutes Online (SSO)](https://sso.agc.gov.sg):

| File                                                              | Title                                                      | Doc Type               |
| ----------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------- |
| `road-traffic-act-1961.html`                                      | Road Traffic Act 1961                                      | Act                    |
| `highway-code.html`                                               | Highway Code                                               | Code (grouped by Part) |
| `active-mobility-act-2017.html`                                   | Active Mobility Act 2017                                   | Act                    |
| `motor-vehicles-third-party-risks-and-compensation-act-1960.html` | Motor Vehicles (Third-Party Risks & Compensation) Act 1960 | Act                    |
| `parking-places-act-1974.html`                                    | Parking Places Act 1974                                    | Act                    |
| `restriction-of-speed-on-roads.html`                              | Road Traffic (Restriction of Speed on Roads) Notification  | Rules                  |

**Chunking strategy**: Acts and Rules are chunked per section (one chunk per provision). The Highway Code uses grouped chunking (one chunk per Part) because its provisions are short. Long provisions are sub-chunked with 200-character overlap.

---

## 6. How to Build

### Prerequisites

- **Docker Desktop** (with Docker Compose v2)
- **Ollama** installed on the host machine — [ollama.ai](https://ollama.ai)
- **Python 3.11+** (for ingestion and the frontend server)
- **ngrok** account (free tier) — for Telegram webhook tunnelling

### Step 1 — Pull required Ollama models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull gemma4:e2b        # LLM judge for evaluation
```

### Step 2 — Clone and configure

```bash
git clone <repo-url>
cd lawra

# Copy the example env file and fill in your values
cp .env.example .env
```

Key `.env` values:

```dotenv
POSTGRES_PASSWORD=<choose a password>

# Generate a fresh key — never reuse one from another installation
N8N_ENCRYPTION_KEY=<generate with: python3 -c "import secrets; print(secrets.token_hex(32))">

# For Telegram support via ngrok:
N8N_HOST=<your-ngrok-subdomain>.ngrok-free.app
N8N_PROTOCOL=https
WEBHOOK_URL=https://<your-ngrok-subdomain>.ngrok-free.app/
```

> **Important:** `N8N_ENCRYPTION_KEY` is used to AES-encrypt all credentials stored in n8n. Generate a unique key per installation and never commit a real key to version control. If you lose or change the key after credentials have been saved in n8n, you will need to re-enter them.

### Step 3 — Start Docker services

```bash
docker compose up -d
```

This starts:

- **Qdrant** on `127.0.0.1:6333` (REST) and `127.0.0.1:6334` (gRPC) — localhost only
- **PostgreSQL** on internal Docker network only
- **n8n** on `127.0.0.1:5678` — localhost only

All ports are bound to `127.0.0.1` so they are not reachable from other machines on the local network.

Verify: open `http://localhost:5678` and complete the n8n first-run setup.

### Step 4 — Import the n8n workflow

1. In n8n, go to **Workflows → Import from file**
2. Import `frontend/lawra_query_workflow.json`
3. **Activate** the workflow (toggle to ON)
4. Verify the webhook is live at `http://localhost:5678/webhook/lawra/query`

If you are using Telegram: add a **Telegram Trigger** node at the start of the workflow, connect it to your bot token (from [@BotFather](https://t.me/BotFather)), and route messages through the same pipeline.

### Step 5 — Install Python dependencies

```bash
# For ingestion
pip install beautifulsoup4 lxml requests qdrant-client chromadb

# For evaluation
cd eval
pip install -r requirements.txt    # ragas langchain-ollama datasets qdrant-client requests
```

### Step 6 — Download and ingest legislation

Download legislation HTML files from SSO:

1. Open the SSO URL for each act (e.g., `https://sso.agc.gov.sg/Act/RTA1961`)
2. In the print panel, select **Print → HTML**
3. Save the resulting page as **"Webpage, HTML Only"** into `docs/`

Then initialise the manifest (first time only):

```bash
python lawra_ingest_html.py --html-dir ./docs --init-manifest manifest.json
# Edit manifest.json to fill in title, source_url, doc_type, label_prefix for each file
```

Inspect the parsed structure:

```bash
python lawra_ingest_html.py --html-dir ./docs --manifest manifest.json --inspect
```

Dry-run (preview chunks without uploading):

```bash
python lawra_ingest_html.py --html-dir ./docs --manifest manifest.json --dry-run
```

Ingest to Qdrant:

```bash
python lawra_ingest_html.py --html-dir ./docs --manifest manifest.json
# Uses bge-m3 embedding model and "lawra" collection by default
# Override: --embed-model nomic-embed-text --collection lawra
```

---

## 7. How to Run the System

### Start Docker services (if not already running)

```bash
docker compose up -d
```

### Start the frontend server

```bash
cd frontend
python server.py              # serves at http://localhost:7890
python server.py --port 8080  # alternative port
```

Open `http://localhost:7890` in your browser.

### Telegram bot

Start ngrok to expose n8n:

```bash
ngrok http 5678
```

Copy the `https://` URL into your `.env` as `WEBHOOK_URL` and `N8N_HOST`, then restart n8n:

```bash
docker compose restart n8n
```

> **Note:** n8n runs with `N8N_SECURE_COOKIE: true`. This is correct when accessing n8n over the ngrok HTTPS tunnel. For local-only setups (no ngrok), you can set `N8N_SECURE_COOKIE: false` in `docker-compose.yml`.

The bot is accessible at [@the_lawra_bot](https://t.me/the_lawra_bot).

### Service URLs summary

| Service             | URL                                          | Notes                  |
| ------------------- | -------------------------------------------- | ---------------------- |
| Web App             | `http://localhost:7890`                      |                        |
| n8n Workflow Editor | `http://localhost:5678`                      | localhost only         |
| Qdrant Dashboard    | `http://localhost:6333/dashboard`            | localhost only         |
| Ollama API          | `http://localhost:11434`                     |                        |
| Telegram Bot        | [@the_lawra_bot](https://t.me/the_lawra_bot) | requires ngrok tunnel  |

### Stop everything

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop and delete all data (destructive!)
```

---

## 8. Evaluation

Lawra includes a RAGAS-based evaluation harness that measures RAG quality against a curated golden dataset of 15 Singapore road traffic law questions.

### Metrics

| Metric                | What it measures                                                     |
| --------------------- | -------------------------------------------------------------------- |
| **Faithfulness**      | Hallucination resistance — is the answer grounded in retrieved text? |
| **Answer Relevancy**  | Does the answer address the question asked?                          |
| **Context Precision** | Are the most relevant chunks ranked highest?                         |
| **Context Recall**    | Are all necessary chunks retrieved?                                  |

### Running evaluations

```bash
cd eval

# Quick check — 5 questions (~5–10 min on M4)
python lawra_eval.py --quick

# Full run — all 15 questions (~15–45 min locally)
python lawra_eval.py

# Use gemma4:e2b as the judge (recommended — avoids self-evaluation bias)
python lawra_eval.py --judge-model gemma4:e2b

# Skip context_recall (slowest metric)
python lawra_eval.py --no-context-recall
```

Results are saved to `eval/results/lawra_eval_<timestamp>.json` and automatically copied to `frontend/eval_results.json` for display in the web UI's **Evaluate** tab.

You can also trigger evaluation from the web UI without the command line.

---

## 9. Key Learnings

### RAG & Retrieval

- **Chunk boundaries matter more than chunk size.** Splitting on legal section boundaries (one provision per chunk) dramatically outperformed fixed-size splitting. Every chunk carries its section header so any retrieved fragment is still clearly anchored.
- **Hybrid chunking is necessary.** The Highway Code's short provisions worked best grouped by Part (to stay above the minimum meaningful context threshold), while Acts needed per-section granularity. Auto-detecting the right mode from average provision length was a pragmatic win.
- **The SSO DOM is consistent but fragile.** All legislation content lives in `div#tocView`. Never modify the DOM while iterating — extract values by string operations to avoid stale reference bugs.

### Local LLMs & Tooling

- **Strict system prompts are load-bearing.** Without an explicit "if not in the text, say so" instruction, `llama3.1:8b` would confidently fabricate plausible-sounding (but wrong) statutory penalties. The refusal clause is essential for a legal assistant.
- **n8n is excellent for rapid prototyping of agentic pipelines.** Wiring Webhook → Embed → Retrieve → Generate → Respond as visual nodes accelerated iteration significantly versus code-only approaches. The tradeoff is debugging complex JS expressions in the n8n Code nodes.
- **`llama3.1:8b` judging itself is a weak signal.** Evaluation scores from a model judging its own outputs are optimistically biased. Using `gemma4:e2b` as a separate judge gives more meaningful faithfulness scores.
- **ChromaDB in-process is surprisingly capable.** Using ChromaDB as an embedded library (no extra Docker service) for session memory kept the stack simpler while still providing semantic retrieval over conversation history rather than naive recency-based context injection.

### Infrastructure & DevOps

- **`host.docker.internal` is the key bridge.** Ollama runs on the host (for direct GPU/Metal access), while n8n and Qdrant run in Docker. Using `host.docker.internal` in the n8n Ollama HTTP calls, and `extra_hosts: host.docker.internal:host-gateway` in the compose file, bridges this gap cleanly.
- **ngrok for Telegram is acceptable at prototype scale.** The free ngrok tier is sufficient for a personal bot. Production would require a proper domain and reverse proxy.
- **Deterministic chunk IDs (UUID5) enable safe re-ingestion.** Using a namespace UUID seeded by `doc_id|section_label|chunk_index` means re-running ingestion after document updates upserts (not duplicates) existing chunks.

### Evaluation

- **RAGAS with local models is slow but viable.** A full 15-question run with 4 metrics takes 15–45 minutes on an M4 MacBook Pro. The async evaluation trigger in the web UI (poll `/api/eval-status`) makes this tolerable.
- **Ground truth quality caps evaluation quality.** Poorly written ground truths in `golden_dataset.json` will produce misleading `context_recall` scores. Each ground truth was verified against the actual SSO text.

---

## 10. Future Enhancements

### Retrieval Quality

- [ ] **Hybrid search**: combine dense vector search with BM25 keyword search (Qdrant supports this natively) to improve recall for exact statutory references like "s 67" or "para 2"
- [ ] **Re-ranking**: add a cross-encoder re-ranker step after initial retrieval to improve context precision
- [ ] **Multi-vector retrieval**: store both passage-level and document-level embeddings for coarse-to-fine retrieval

### Knowledge Base

- [ ] **Expanded coverage**: add subsidiary legislation, LTA circulars, and Traffic Police advisories
- [ ] **Automated SSO sync**: scheduled ingestion pipeline that detects and re-ingests amended legislation
- [ ] **Version-aware retrieval**: surface the `version_date` of retrieved chunks and warn users when legislation may have been updated

### Agentic Capabilities

- [ ] **Multi-hop reasoning**: for questions that span multiple acts (e.g., "Can I use a PAB on a road and what insurance do I need?"), implement an agent loop that issues multiple retrieval steps
- [ ] **Tool use**: give the agent access to tools for calculating penalties (e.g., with demerit point tables) or checking public holiday dates for offence context
- [ ] **Researcher agent**: DuckDuckGo search integration for questions about LTA advisories and press releases not in the statutory corpus

### User Experience

- [ ] **Voice input/output**: speech-to-text for hands-free queries (relevant for drivers)
- [ ] **Citation deep-links**: link source citations directly to the specific SSO page and section
- [ ] **Multi-language support**: Mandarin, Malay, and Tamil responses for accessibility

### Operations

- [ ] **Containerised frontend**: Dockerfile for `frontend/server.py` so the whole stack is `docker compose up`
- [ ] **Persistent ngrok domain**: move to a stable domain for the Telegram webhook instead of ephemeral ngrok URLs
- [ ] **Structured logging & tracing**: integrate OpenTelemetry or Langfuse for query-level observability
- [ ] **CI evaluation gate**: run `--quick` RAGAS evaluation on every document corpus update

---

## Project Structure

```
lawra/
├── docker-compose.yml          # Qdrant + PostgreSQL + n8n services
├── .env                        # Environment config (not committed with secrets)
├── manifest.json               # Legislation file metadata for ingestion
├── lawra_ingest_html.py        # SSO HTML parser & Qdrant ingestion pipeline
│
├── docs/                       # Source legislation HTML files (from SSO)
│   ├── road-traffic-act-1961.html
│   ├── highway-code.html
│   ├── active-mobility-act-2017.html
│   └── ...
│
├── frontend/                   # Web application
│   ├── index.html              # Single-page web UI
│   ├── server.py               # HTTP server + memory-augmented query proxy
│   ├── memory.py               # ChromaDB-backed semantic session memory
│   ├── lawra_query_workflow.json  # n8n workflow export (import into n8n)
│   └── eval_results.json       # Latest evaluation results (auto-updated)
│
└── eval/                       # Evaluation harness
    ├── lawra_eval.py           # RAGAS evaluation pipeline
    ├── golden_dataset.json     # 15 curated Q&A pairs with ground truths
    ├── requirements.txt        # Eval-specific Python dependencies
    └── results/                # Timestamped evaluation run outputs
```

---

## Disclaimer

Lawra is based on an unofficial consolidation of Singapore legislation for informational and educational purposes. It is **not legal advice**. The authoritative text of Singapore law is the Government Gazette. Always verify at [Singapore Statutes Online (SSO)](https://sso.agc.gov.sg) or consult a qualified legal professional.
