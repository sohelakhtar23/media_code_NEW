
import os
import warnings
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import time

load_dotenv()
warnings.filterwarnings("ignore")

# ── Refined dimensions focused on memorability ────────────────────────────────
LLM_DIMENSIONS_V2 = {
    "emotional_valence":     "Emotional tone: 0=cold/negative → 10=warm/uplifting/positive",
    "human_presence":        "Are real people (faces, presenters, spokespeople) prominent? 0=none, 10=dominant",
    "message_simplicity":    "How simple and easy to remember is the core message? 0=complex/forgettable, 10=single clear takeaway",
    "novelty_surprise":      "How unexpected or counter-intuitive is the message? 0=predictable, 10=surprising",
    "narrative_arc":         "Clear story structure (problem→solution or journey)? 0=none, 10=strong arc",
    "brand_prominence":      "How centrally/repeatedly is the brand name featured? 0=never, 10=constant",
    "repetition_hooks":      "Repeated phrases, slogans, or memorable hooks? 0=none, 10=strong repetition",
    "direct_memorability":   "Overall: how memorable would this video be to a viewer? 0=very forgettable, 10=highly memorable",
}

LLM_KEYS_V2 = list(LLM_DIMENSIONS_V2.keys())

_SYSTEM = (
    "You are an expert in advertising psychology and memory research. "
    "Given a commercial video's metadata and transcript, score it on specific dimensions. "
    "Return ONLY a valid JSON object — no markdown, no commentary."
)

def build_prompt(row, stt_text, max_words=600):
    stt_clip     = " ".join(stt_text.split()[:max_words])
    schema_lines = "\n".join(f'  "{k}": {v}' for k, v in LLM_DIMENSIONS_V2.items())
    keys_json    = "{" + ", ".join(f'"{k}": ...' for k in LLM_KEYS_V2) + "}"
    return (
        f"Commercial video:\n"
        f"- Brand   : {row.get('channelName', 'Unknown')}\n"
        f"- Title   : {row.get('title', '')}\n"
        f"- Desc    : {str(row.get('description', ''))[:400]}\n"
        f"- Duration: {row.get('durationSeconds', 0):.0f}s\n"
        f"- Transcript (~{max_words} words): {stt_clip}\n\n"
        f"Score on these dimensions (floats 0-10):\n{schema_lines}\n\n"
        f"Return ONLY: {keys_json}"
    )



client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
df     = pd.read_csv("devset_videolist_GT.csv")
stt_dir = Path("devset-stt/")

cache_path = Path("llm_scalar_cache_v2.json")
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

for _, row in df.iterrows():
    vid_id = row["id"]
    if vid_id in cache:
        continue

    stt_file = stt_dir / f"{vid_id}.txt"
    stt_text = stt_file.read_text(encoding="utf-8", errors="ignore") if stt_file.exists() else ""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": build_prompt(row, stt_text)},
            ],
        )
        scores = json.loads(resp.choices[0].message.content)
        cache[vid_id] = {k: float(scores.get(k, 5.0)) for k in LLM_KEYS_V2}
        print(f"  {vid_id}: {cache[vid_id]}")
    except Exception as e:
        print(f"  [ERR] {vid_id}: {e}")
        cache[vid_id] = {k: 5.0 for k in LLM_KEYS_V2}

    cache_path.write_text(json.dumps(cache, indent=2))
    time.sleep(0.1)

print(f"Done. {len(cache)} videos in cache.")
