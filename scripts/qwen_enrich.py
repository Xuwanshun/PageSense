"""Re-enrich a processed doc's visual_summaries.json via the deployed Qwen VLM.

Runs the VLM summary step decoupled from OCR: reads existing crops, calls the
self-hosted Qwen endpoint (VLM_BASE_URL) concurrently, and writes the enriched
summaries back. Used because OCR was run with USE_VLM_SUMMARIES=false to avoid
GPT-4o hangs; this applies the fine-tuned Qwen model afterward.

    python scripts/qwen_enrich.py data/processed/<doc_id> [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
from document_process.models import VisualRegionSummary
from document_process.vlm import _describe_crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_dir", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    settings = Settings()
    if not settings.vlm_base_url:
        raise SystemExit("VLM_BASE_URL not set — cannot reach Qwen endpoint.")

    vs_path = args.doc_dir / "visual_summaries.json"
    raw = json.loads(vs_path.read_text(encoding="utf-8"))
    summaries = [VisualRegionSummary.model_validate(x) for x in raw]

    def work(idx: int, s: VisualRegionSummary):
        crop = Path(s.crop_path) if s.crop_path else None
        if not crop or not crop.exists():
            return idx, None, None
        try:
            desc, meaningful = _describe_crop(
                crop_path=crop,
                region_type=s.region_type,
                context_text=s.summary_text,
                settings=settings,
            )
            return idx, desc, meaningful
        except Exception as exc:  # keep OCR fallback on failure
            return idx, f"__ERR__{type(exc).__name__}", None

    done = ok = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, s) for i, s in enumerate(summaries)]
        for fut in as_completed(futs):
            idx, desc, meaningful = fut.result()
            done += 1
            if desc is None:
                continue
            if desc.startswith("__ERR__"):
                failed += 1
                continue
            if meaningful:
                summaries[idx] = summaries[idx].model_copy(update={"summary_text": desc})
            else:
                summaries[idx] = summaries[idx].model_copy(update={"is_meaningful": False})
            ok += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(summaries)} processed (ok={ok}, failed={failed})", flush=True)

    vs_path.write_text(
        json.dumps([s.model_dump() for s in summaries], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DONE {args.doc_dir.name[:12]}… enriched={ok} failed={failed} total={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
