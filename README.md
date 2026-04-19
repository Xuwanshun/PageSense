# PDF RAG

Process PDFs from `Data/Raw`, freeze processed artifacts in `Data/Processed`, build embeddings in `Data/Embedded`, and answer grounded questions from the saved document package and index.

## Requirements

- Linux with Python 3.11
- `OPENAI_API_KEY` in the shell environment or a local `.env` file for `--index` and `--ask`

Optional:
- `OPENAI_BASE_URL` if you are using an OpenAI-compatible endpoint
- `chromadb` only if you want to switch from the built-in JSON store to Chroma

## Local Linux Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your environment values in `.env`:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

## Project Layout

- Put source PDFs in `Data/Raw`
- Processed artifacts are written to `Data/Processed`
- Embeddings are written to `Data/Embedded`

These directories are created automatically when the app runs.

## Architecture

```mermaid
flowchart TD
    subgraph Input
        PDF[📄 PDF File]
        CLI["CLI\npython main.py"]
        HTTP["HTTP Client\n/api/documents"]
    end

    subgraph document_process["document_process/ — Stage 1: PDF → Artifacts"]
        DL[DocumentLoaderService\nload + copy PDF]
        OCR[OCRService\nPaddleOCR]
        RO[ReadingOrderService\nresolve block order]
        LD[LayoutDetectionService\nPP-DocLayout_plus-L]
        AS[AssociationService\nlink text ↔ regions]
        CR[CroppingService\ncrop tables & figures]
        VLM["VLM Enrichment\ngpt-4o\n(USE_VLM_SUMMARIES=true)"]
        ARTS["Frozen Artifacts\ndocument.json · chunks.json · crops/"]
    end

    subgraph rag["rag/ — Stage 2: Artifacts → Index → Answer"]
        CH[chunk.py\nProcessedChunk → ChunkRecord]
        EM[embed.py\nOpenAI text-embedding-3-small]
        VS{"VectorStore\n(PREFER_CHROMA)"}
        JVS[JsonVectorStore\nstore.json]
        CVS[ChromaVectorStore]
        QA[qa.py\ngpt-4.1-mini\nQAResponse + sources]
    end

    subgraph api["api/ — FastAPI HTTP Layer"]
        APP[create_app factory]
        HR[/health]
        DR["/api/documents\nupload · list · delete"]
        QR["/api/query\nPOST question"]
        UI[Static Frontend]
    end

    subgraph infra["Infrastructure (AWS)"]
        S3[(S3\nartifacts + vectorstore)]
        ECS[ECS Fargate\n2 vCPU / 8 GB]
        ALB[ALB]
    end

    PDF --> CLI
    PDF --> DR
    CLI --> DL
    DR --> DL

    DL --> OCR --> RO --> LD --> AS --> CR
    CR --> VLM
    CR --> ARTS
    VLM --> ARTS

    ARTS --> CH --> EM --> VS
    VS -->|default| JVS
    VS -->|opt-in| CVS
    JVS --> QA
    CVS --> QA

    APP --> HR & DR & QR & UI
    QR --> QA

    ARTS <-->|sync| S3
    JVS <-->|sync| S3
    ECS --> APP
    ALB --> ECS
```

## Run

Show CLI help:

```bash
python main.py --help
```

Freeze preprocessing outputs:

```bash
python main.py --preprocess
```

Build embeddings from frozen chunks:

```bash
python main.py --index
```

Ask a question against the frozen index and artifacts:

```bash
python main.py --ask "What is the goal of the AI RMF?"
```

## AWS Deployment

```
Internet
    │  HTTP :80
    ▼
┌─────────────────┐
│   ALB            │  Application Load Balancer — public entry point
│  (alb.tf)        │  Routes to healthy ECS tasks, health-checks /health
└────────┬────────┘
         │ HTTP :8000 (only from ALB)
         ▼
┌─────────────────┐
│  ECS Fargate     │  Runs your Docker container (serverless — no EC2 to manage)
│  (ecs.tf)        │  2 vCPU / 8 GB RAM per task
│                  │
│  env vars:       │  APP_MODE=api → starts FastAPI on port 8000
│  S3_BUCKET_NAME ─┼──────────────────────────────────────┐
│  OPENAI_API_KEY ◄┼── from Secrets Manager               │
│                  │                                       │
│  /app/paddle_models ◄── EFS mount                       │
└──────────────────┘                                       │
         │                                                 │
         ▼                                                 ▼
┌────────────────┐                              ┌─────────────────────┐
│  EFS            │                              │  S3                 │
│  (efs.tf)       │                              │                     │
│  Paddle models  │                              │  processed/         │
│  ~1.5 GB        │                              │  embedded/          │
│  persists across│                              │  (vector store)     │
│  deploys        │                              └─────────────────────┘
└────────────────┘
```

| Service | Purpose |
|---|---|
| **S3** | Persists processed artifacts and vector store across container restarts |
| **EFS** | Persists Paddle model cache (~1.5 GB) so models are not re-downloaded on every deploy |
| **ECR** | Private Docker image registry — ECS pulls the image from here |
| **ALB** | Public internet entry point, routes traffic to healthy containers |
| **Secrets Manager** | Injects `OPENAI_API_KEY` into the container at startup — never stored in the image |

The container filesystem is ephemeral. On every startup, `sync_from_s3()` pulls the latest processed artifacts and vector store from S3 to local disk. After preprocessing or indexing, the app pushes results back to S3. Set `S3_BUCKET_NAME` to enable; leave it empty for local development (no-op).

## Detection Notes

The preprocessing pipeline uses:

- PDF page rendering from `pypdfium2`
- PaddleOCR for OCR text extraction, bounding boxes, and confidence scores
- Paddle `LayoutDetection` (`PP-DocLayout_plus-L`) for text blocks, tables, and figure/image regions
- Reading order based on OCR text boxes only
- Frozen visual summaries derived from saved OCR/layout/chunk outputs

Important limitations:

- The first run downloads official Paddle models into `.paddlex/`
- Figure detection comes from the layout detector's real visual labels such as `image` and `figure`; there is no separate chart classifier in the active path
- The only LLM backend used for indexing/query-time reasoning is OpenAI
