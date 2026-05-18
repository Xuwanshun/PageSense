# Project Interview Guide — PDF RAG Agent

A learning guide for understanding this project end-to-end and talking about it confidently in a technical interview.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Core Concept: RAG](#2-the-core-concept-rag)
3. [Stage 1 — Document Processing](#3-stage-1--document-processing)
4. [Stage 2 — RAG Pipeline](#4-stage-2--rag-pipeline)
5. [Stage 3 — API and Auth](#5-stage-3--api-and-auth)
6. [Infrastructure on AWS](#6-infrastructure-on-aws)
7. [Key Design Decisions](#7-key-design-decisions)
8. [The Fine-Tuned VLM](#8-the-fine-tuned-vlm)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Interview Q&A](#10-interview-qa)

---

## 1. What is this project?

A **production-grade, multi-user document question-answering system**. Users upload PDFs, the system reads and understands them (OCR, layout detection, embeddings), and users can ask questions in plain English and get answers grounded in the document content.

**Key capabilities:**
- Handles complex PDFs with tables, figures, and multi-column layouts — not just plain text
- Multi-user with Google OAuth and JWT auth
- Persistent conversation history
- Deployed on AWS ECS Fargate, scales to zero when idle
- Fine-tuned vision model (Qwen3-VL) for describing charts and tables

**The tech stack at a glance:**
- **Backend:** Python, FastAPI, PaddleOCR, PaddleX Layout Detection
- **AI:** OpenAI GPT-4.1-mini (answers), text-embedding-3-small (embeddings), GPT-4o vision (table/figure descriptions)
- **Database:** PostgreSQL (RDS) for auth + conversations, custom JSON vector store for embeddings
- **Infra:** AWS ECS Fargate, ALB, S3, Secrets Manager, ECR — all provisioned with AWS CDK (Python)
- **CI/CD:** GitHub Actions

---

## 2. The Core Concept: RAG

**RAG = Retrieval-Augmented Generation**

The problem with asking an LLM about your PDF directly:
- LLMs have a context window limit — you can't fit a 200-page PDF in a prompt
- LLMs hallucinate — they invent facts that aren't in the document
- LLMs have a training cutoff — they don't know about your proprietary documents

RAG solves this by splitting the problem into two phases:

```
INDEXING (done once when PDF is uploaded):
  PDF → extract text + images → split into chunks → convert to vectors → store

QUERYING (done each time a user asks a question):
  Question → convert to vector → find similar chunks → feed chunks + question to LLM → answer
```

The LLM only ever sees the most relevant 3-5 chunks, not the whole document. This keeps it grounded and within the context limit.

**Why it works:** Text that means the same thing has vectors that point in the same direction in high-dimensional space. "Revenue growth" and "sales increased" are semantically close even though they share no words.

---

## 3. Stage 1 — Document Processing

This is the most technically complex part. Most RAG systems just extract raw text from PDFs. This system goes much further.

### The Processing Pipeline

```
PDF
 ↓
DocumentLoaderService    — copy PDF to working directory, extract metadata
 ↓
OCRService               — PaddleOCR extracts text with bounding boxes (x,y,w,h for each word)
 ↓
ReadingOrderService      — sort text blocks into reading order (handles multi-column layouts)
 ↓
LayoutDetectionService   — PP-DocLayout detects tables, figures, section headers, paragraphs
 ↓
AssociationService       — link nearby text to detected regions (captions → figures, etc.)
 ↓
CroppingService          — save PNG crops of each detected table/figure
 ↓
VLM Enrichment           — GPT-4o looks at each crop image and writes a description
 ↓
Frozen Artifacts         → document.json, chunks.json, visual_summaries.json, crop images
```

### Why not just use PyPDF or pdfplumber?

Text-based PDF extraction only works for PDFs with embedded text layers. It completely fails for:
- Scanned documents (images of pages)
- Tables (extracts as garbled text with no structure)
- Charts and figures (no text to extract)
- Multi-column layouts (merges columns incorrectly)

PaddleOCR reads the actual pixels, so it works on all of these.

### What is PP-DocLayout?

A Paddle model trained to detect document structure. It draws bounding boxes around:
- `text` — regular paragraphs
- `table` — tabular data
- `figure` — charts, graphs, images
- `title` / `header` — section headings

This allows the system to treat tables and figures differently from regular text — crop them as images and send them to a vision model instead of relying on garbled OCR text.

### VLM Enrichment — Why it matters

For a table like this:

```
| Quarter | Revenue | Growth |
|---------|---------|--------|
| Q1 2024 | $4.2B   | +12%   |
| Q2 2024 | $4.8B   | +14%   |
```

OCR extracts: `"Quarter Revenue Growth Q1 2024 4.2B 12 Q2 2024 4.8B 14"` — structurally meaningless.

GPT-4o looks at the image and writes: *"Revenue table covering Q1–Q2 2024. Revenue grew from $4.2B to $4.8B, a 12% and 14% year-over-year growth rate respectively."*

That description is semantically rich and will match user queries like "how did revenue grow?" — the OCR text would not.

### Memory Management (important design choice)

PaddleOCR and LayoutDetection models are ~1.4 GB combined. Loading them on every upload request would be extremely slow and would OOM the container.

**Solution:** `@lru_cache(maxsize=1)` singleton — models load once per process and stay in memory. A `threading.Semaphore(1)` ensures only one OCR job runs at a time, preventing two uploads from doubling RAM usage and crashing the 8 GB container.

---

## 4. Stage 2 — RAG Pipeline

This is where the retrieval and answer generation happens. The system implements several enhancements on top of basic RAG.

### Basic flow

```
Question
 ↓
Query Enhancement    — make the question better for retrieval
 ↓
Hybrid Retrieval     — find relevant chunks
 ↓
LLM Reranking        — score and filter the retrieved chunks
 ↓
Context Compression  — strip irrelevant sentences
 ↓
Answer Generation    — GPT-4.1-mini synthesizes the answer
 ↓
Faithfulness Check   — verify the answer is supported by sources
```

### Query Enhancement

**Problem:** Users ask vague or short questions. "Revenue?" is a bad search query.

**Two techniques:**

1. **HyDE (Hypothetical Document Embeddings):** Instead of embedding the question itself, ask the LLM to write a hypothetical answer paragraph — then embed that. A fake answer is much closer in vector space to the real document text than the question is.

   - Question: "What was revenue in Q2?"
   - HyDE: "In Q2 2024, total revenue reached $4.8 billion, representing a 14% year-over-year increase..."
   - This fake paragraph will retrieve the actual revenue section much more reliably.

2. **Query decomposition:** For complex questions like "Compare Q1 and Q2 revenue and explain the growth drivers", the system splits it into sub-queries and retrieves for each independently.

### Hybrid Retrieval (BM25 + Vector)

**Problem:** Pure vector search misses exact keyword matches. If a user asks "What is the EBITDA margin?" and the document says exactly "EBITDA margin is 23%", pure vector search might miss this because other semantically-similar passages score higher.

**Solution:** Run two searches in parallel:

1. **Dense (vector) search:** Semantic similarity via embeddings — good for concepts, paraphrases
2. **Sparse (BM25) search:** Keyword matching — good for exact terms, numbers, proper nouns

Then fuse them with **Reciprocal Rank Fusion (RRF):**

```python
score = 1/(rank_dense + k) + 1/(rank_sparse + k)
```

A chunk that ranks highly in both searches gets a very high fused score. Neither search alone is as reliable.

**Region-type boosting:** Chunks from tables and figures get score boosts when the query has data/number intent keywords ("how many", "percentage", "compare", "top 5"). This ensures data questions retrieve data chunks, not prose.

### LLM Reranking

After retrieval, a second LLM pass scores each chunk 0.0–1.0 for relevance to the specific question. Chunks below 0.3 are dropped. This is more accurate than vector similarity but more expensive — so it's applied to a small set of already-filtered candidates, not the entire corpus.

**Batch scoring:** All candidates are scored in a single LLM call (not one call per chunk) to keep latency reasonable.

### Context Compression

Even after reranking, retrieved passages often contain sentences that are irrelevant to the specific question. Compression strips those out, reducing the context window used by the synthesis LLM by ~40%. This lowers cost and improves answer quality (less noise).

### Faithfulness Check

After the answer is generated, a second LLM pass checks each claim in the answer against the source passages. Claims marked UNSUPPORTED or INFERRED are automatically rewritten or removed. This is the anti-hallucination layer — the answer can only say things that are explicitly supported by the retrieved text.

### Vector Store

Uses a custom JSON-based vector store (`store.json`) rather than a third-party database like Pinecone or ChromaDB. Reasons:
- No extra infrastructure to manage
- Persists to S3 (ECS is stateless)
- Simple cosine similarity search is fast enough for document-scale corpora
- ChromaDB is available as an opt-in (`PREFER_CHROMA=true`) for larger scale

---

## 5. Stage 3 — API and Auth

### FastAPI Structure

The API follows the factory pattern — `create_app(settings)` returns the app instance. This makes testing clean: tests create their own app instances with test settings, no global state.

### Authentication

**JWT + HttpOnly cookies:**
- Access token (15 min) — stored in JavaScript memory only, never localStorage (XSS protection)
- Refresh token (7 days) — stored in HttpOnly cookie, JavaScript cannot read it (CSRF protection)
- Silent re-auth: when access token expires, the app automatically calls `/auth/refresh` using the HttpOnly cookie — users never see a login prompt

**Token rotation:** Every refresh issues a new refresh token and deletes the old one. If a stolen refresh token is used, the next legitimate refresh will fail (old token already deleted), alerting the user.

**Google OAuth flow:**
```
Browser → /auth/oauth/google → redirect to Google
Google authenticates user → callback to /auth/oauth/google/callback
System gets user email from Google → upsert into users table → issue JWT
```

### Conversation History

Every query is saved as a message in a conversation. The `/conversations` endpoint returns a list of conversations (newest first) with a preview of the first message. Users can continue old conversations or start new ones. All data is scoped to the authenticated user.

---

## 6. Infrastructure on AWS

### Why ECS Fargate over EC2?

- **No server management** — AWS handles OS patches, capacity, health
- **Pay per second** — when `desired_count=0`, Fargate costs exactly $0
- **Scales to zero** — for a learning project, this saves ~$60-80/month vs always-on EC2

### Why no NAT Gateway?

Standard VPC setup: public subnets + private subnets + NAT Gateway. Cost: ~$73/month just for the NAT.

**This project's choice:** ECS tasks run in public subnets with `assign_public_ip=True`. The tasks still have security groups that block all inbound traffic except from the ALB — so they're not publicly accessible. This saves $73/month with equivalent security.

### Why S3 for artifacts?

ECS Fargate containers are **ephemeral** — when a container is replaced (deploy, crash, scale event), all local data is wiped. S3 is the durable store.

On startup: `sync_from_s3()` downloads processed artifacts and the vector store to local disk.
After preprocessing: `sync_to_s3()` uploads new artifacts.

This means the container can be replaced at any time without losing data.

### Why three CDK stacks?

```
RagAgentNetwork  →  RagAgentDatabase  →  RagAgentApp
```

**Separation reason:** RDS has `termination_protection=True` and `removal_policy=RETAIN`. If the App stack is accidentally deleted and recreated, the database is not touched. You can redeploy the App 100 times without risk of losing user data.

### CDK vs Terraform

The project originally used Terraform but migrated to CDK. Reasons:
- Python CDK = same language as the app code, no HCL to learn
- CDK constructs like `grant_read_write()` generate minimal IAM policies automatically
- CDK `outputs` make it easy to pipe values (ECR URI, cluster name) into GitHub secrets

### Secrets Manager

All sensitive values are in Secrets Manager, never in environment variables or source code. ECS fetches them at container startup and injects them as env vars. The app code just reads `os.environ` — no AWS SDK calls in the app.

**Important:** All secrets use full ARNs (`from_secret_complete_arn`), not name-based lookup. AWS appends a random suffix to secret names (`rag-agent/openai-api-key-7KOsxs`), and name-based lookup produces an incomplete ARN that causes `ResourceNotFoundException` at container startup.

### ALB Idle Timeout (600 seconds)

The default ALB idle timeout is 60 seconds. OCR on a large PDF can take 2-5 minutes. Without this change, the ALB would cut the connection mid-processing, the browser would show an error, but the backend would keep running — creating a broken state where processing completed but the response was never delivered.

---

## 7. Key Design Decisions

These are the questions an interviewer will ask "why did you do X instead of Y?"

### Why PaddleOCR over Tesseract or AWS Textract?

- **Tesseract** is slower and less accurate on complex layouts
- **Textract** costs money per page and requires internet (can't run locally)
- **PaddleOCR** is free, fast, highly accurate, runs locally, and comes with PP-DocLayout for layout detection in the same framework

### Why a custom JSON vector store over Pinecone/Weaviate?

- **No operational overhead** — no extra service to manage, monitor, or pay for
- **Persists to S3** natively — fits the stateless ECS pattern
- **Sufficient for document scale** — a 200-page PDF produces ~200-300 chunks; brute-force cosine similarity on 1000 chunks takes milliseconds
- **Pinecone** makes sense at 1M+ vectors; for document QA, it's overengineering

### Why GPT-4.1-mini for answers instead of GPT-4o?

- GPT-4.1-mini is 80-90% as capable at synthesis tasks at ~10x lower cost
- The hard AI work (understanding the document) is done in preprocessing with GPT-4o vision
- Answer synthesis from already-retrieved text chunks is a simpler task

### Why store tokens in memory, not localStorage?

- **localStorage** is accessible by JavaScript and can be stolen by XSS attacks
- **Memory** (JS variable) is wiped on page refresh — forces re-auth, which uses the HttpOnly refresh cookie silently
- Trade-off: user needs to refresh cookie every 7 days. For a document QA tool, this is acceptable.

### Why BM25 + vector instead of just vector?

Vector search alone has a known weakness: it's great for semantic similarity but poor at exact matching. In document QA, users often search for specific terms (product names, model numbers, exact percentages). BM25 handles exact matching. The fusion gives the best of both worlds.

---

## 8. The Fine-Tuned VLM

### What was trained

A **Qwen3-VL-4B** vision-language model was fine-tuned on domain-specific examples of (document image → description) pairs. The goal: produce descriptions that match the vocabulary users actually type into a search box, not generic chart descriptions.

**Base model:** Qwen3-VL-4B (from Alibaba, open weights)
**Fine-tuning method:** LoRA (Low-Rank Adaptation) — trains only a small adapter instead of the full model weights, far cheaper and faster
**Hosted on:** Modal.com (serverless GPU, A10G, scales to zero after 5 minutes idle, ~$0.60/hr when active)

### Why not OpenAI fine-tuning?

OpenAI's fine-tuning is text-only — vision models can't be fine-tuned through their API. Open weights models (Qwen, LLaVA, etc.) are the only option for fine-tuning vision models.

### Why Modal over HuggingFace Inference Endpoints?

HuggingFace Inference Endpoints don't support `transformers>=5.0`, which is required by Qwen3-VL architecture. Modal lets you build a custom Docker image with any dependencies.

### The fallback design

```python
if settings.vlm_base_url:
    try:
        return _call_vlm(base_url=settings.vlm_base_url, model="qwen3-vl-rag", ...)
    except Exception:
        logger.warning("Self-hosted VLM unavailable — falling back to gpt-4o")

return _call_vlm(base_url=None, model=settings.vlm_model, ...)  # gpt-4o
```

The self-hosted model is always optional. If Modal is cold-starting (takes >30s), the timeout exception triggers the gpt-4o fallback automatically. The system degrades gracefully, never blocks preprocessing.

---

## 9. CI/CD Pipeline

### ci.yml (every push)
```
lint (ruff) → unit tests → Docker build check (deps stage only)
```
The Docker build check only builds up to the `deps` stage (pip install), not the full image with Paddle model download. This keeps CI fast (<5 min) while still catching broken Dockerfiles.

### deploy.yml (push to main only)
```
lint + tests → build full Docker image → push to ECR → update ECS task def → health-check rollout
```

**Rollback:** ECS uses rolling deployment. The new task must pass health checks (`GET /health`) before the old task is terminated. If health checks fail, the old task keeps running — automatic rollback with zero downtime.

**Concurrency control:** `cancel-in-progress: false` means if two pushes arrive quickly, the second queues and waits. This prevents partial deploys from racing each other.

**ECR login fix:** The `docker/build-push-action` uses buildx internally, which has a separate credential store from the Docker daemon. Using `aws ecr get-login-password | docker login` + plain `docker build/push` bypasses this and works reliably.

---

## 10. Interview Q&A

**"Walk me through what happens when a user uploads a PDF."**

> The upload endpoint saves the PDF to disk, registers a job status in memory, and spawns a background thread. It immediately returns `{ document_id, status: "preprocessing" }` so the browser doesn't hang. In the background: PaddleOCR extracts text with bounding boxes, PP-DocLayout detects tables/figures, the association service links captions to their regions, and the cropping service saves PNG images of each table/figure. GPT-4o vision then describes each crop — this is important because OCR text from a table is structurally meaningless, but a vision model can read it as a human would. The chunks are embedded with text-embedding-3-small, saved to the JSON vector store, and the whole thing is uploaded to S3. Job status flips to "ready".

**"How does the retrieval work?"**

> When a user asks a question, the system first enhances it using HyDE — instead of embedding the raw question, it generates a hypothetical answer paragraph and embeds that. A hypothetical answer is much closer in vector space to actual document text than the question itself. Then it runs two searches in parallel: dense vector search (semantic similarity) and BM25 sparse search (keyword matching). These are fused with Reciprocal Rank Fusion, which rewards chunks that rank highly in both. The top candidates go through LLM reranking — a single batch LLM call scores each chunk 0.0-1.0 for relevance, dropping anything below 0.3. The surviving chunks are compressed to remove irrelevant sentences, then sent to GPT-4.1-mini for synthesis. Finally, a faithfulness checker verifies each claim in the answer against the sources, rewriting any hallucinated claims.

**"Why does the answer sometimes say 'the retrieved evidence does not contain information'?"**

> This means the retrieval step failed to find relevant chunks — the question's embedding was too far from any chunk's embedding in vector space. Common causes: the document wasn't about the topic asked, the preprocessing failed and chunks weren't indexed, or the query enhancement step produced a hypothetical answer that drifted from the actual document vocabulary. We've been investigating this specifically with the Qwen VLM variant to see if different table descriptions change retrieval quality.

**"How did you handle the stateless nature of ECS Fargate?"**

> ECS containers are ephemeral — any restart wipes local storage. The design separates durable state (S3, RDS) from ephemeral state (container filesystem). Processed PDF artifacts and the vector store live in S3. On container startup, `sync_from_s3()` downloads everything to the local filesystem. After preprocessing, `sync_to_s3()` uploads new artifacts. The container can be replaced at any time without data loss.

**"How does the auth work?"**

> Users get an access token (15-minute JWT) stored in JS memory only — never localStorage, which is vulnerable to XSS. A refresh token (7 days) is stored in an HttpOnly cookie that JS can't read, protecting against token theft via script injection. When the access token expires, the frontend automatically calls `/auth/refresh`, which reads the HttpOnly cookie and issues a new token pair — users never see a login screen. Every refresh rotates the refresh token, so a stolen token is detected on the next legitimate use.

**"Why three CDK stacks?"**

> The Database stack has termination protection and `removal_policy=RETAIN`. Separating it means you can destroy and recreate the App stack without touching the database. In practice: every deploy updates the App stack. If something goes wrong and you need to tear down and redeploy from scratch, the user data in RDS survives. This is a common production pattern.

**"What would you improve?"**

> A few things: First, proper async processing with SQS — right now preprocessing runs in a background thread in the same process, which means a container restart during processing loses the job. SQS would decouple upload from processing with retry logic. Second, replacing the JSON vector store with a proper ANN index (FAISS or pgvector) for scale. Third, adding streaming responses to the `/query` endpoint so users see the answer being generated rather than waiting for the full response. Fourth, evaluation metrics — right now there's no automated way to measure retrieval quality or answer accuracy after changes.

---

## Summary: What makes this project interesting to talk about

1. **It's not a tutorial project** — it handles real production concerns: memory management, stateless containers, token rotation, OCR on complex PDFs
2. **Multi-layer RAG** — most tutorials show basic vector search. This implements query enhancement, hybrid retrieval, reranking, compression, and faithfulness checking
3. **Fine-tuned vision model** — trained a domain-specific model, deployed it serverlessly, designed graceful fallback
4. **Infrastructure as code** — CDK stacks, proper IAM least-privilege, Secrets Manager, cost-optimized VPC
5. **Design decisions you can defend** — every "why X over Y" has a concrete answer based on constraints (cost, scale, operational overhead)
