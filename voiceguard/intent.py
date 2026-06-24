"""Intent signal for Stage 2: generate realistic call transcripts per archetype and
score per-turn scam-intent with an LLM (any OpenAI-compatible gateway).

This runs off-GPU. The model both writes the dialogue and rates each turn's scam
intent, so the intent[t] series is a real model signal (not hand-set). Results are
cached to data/dialogues.json so we only pay for generation once.

Auth: reads the key from env VG_LLM_KEY (never hard-coded here).
"""
import os, json, re, random
from openai import OpenAI

BASE_URL = os.environ.get("VG_LLM_BASE_URL")  # OpenAI-compatible gateway; unset -> OpenAI default
MODEL = os.environ.get("VG_LLM_MODEL", "gpt-4o-mini")

SCENARIOS = {
    "scam": ["grandchild impersonation needing bail money", "bank fraud-dept urgent transfer",
             "tax-authority threat demanding gift cards", "stranded-abroad friend needing a wire",
             "tech-support refund scam", "kidnapping ransom hoax"],
    "benign": ["weekend catch-up with a relative", "planning a birthday dinner",
               "coordinating a weekend visit", "chatting about a recipe",
               "talking about a TV show", "arranging a ride to the airport for fun"],
    "urgent": ["real son in a minor car accident needing a tow", "daughter locked out, phone dying",
               "spouse at the ER with a sprained ankle", "parent's flight cancelled, needs a pickup",
               "friend's car broke down on the highway", "sibling needs an urgent ride from work"],
}

PROMPT = """Generate a realistic ~30-second phone call transcript for this scenario:
{desc}

Output STRICT JSON only, no prose or code fences:
{{"turns":[{{"t":0.0,"speaker":"caller","text":"...","scam_intent":0.0}}]}}
Rules: 5-9 turns; t is relative time in [0,1], strictly increasing; scam_intent in [0,1]
is YOUR estimate that THIS turn shows a scam/fraud attempt (impersonation, secrecy,
pressure, money/gift-card/wire request). Benign chat ~0.0-0.2. A genuine but urgent
emergency may look elevated (0.2-0.5) yet is NOT a scam. A real scam escalates to 0.8-1.0."""


def client():
    return OpenAI(api_key=os.environ["VG_LLM_KEY"], base_url=BASE_URL or None)


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def gen_dialogue(cl, desc):
    r = cl.chat.completions.create(model=MODEL, temperature=0.9, max_tokens=1200,
                                   messages=[{"role": "user", "content": PROMPT.format(desc=desc)}])
    turns = sorted(_parse(r.choices[0].message.content)["turns"], key=lambda x: float(x["t"]))
    return [(float(t["t"]), float(t["scam_intent"])) for t in turns]


def build_pools(per=15, seed=0, cache="data/dialogues.json"):
    """Return {archetype: [dialogue, ...]} where dialogue = [(t_frac, scam_intent), ...].
    Cached after first generation."""
    if os.path.exists(cache):
        return json.load(open(cache))
    cl = client()
    pools = {}
    for arch, scens in SCENARIOS.items():
        pool = []
        for i in range(per):
            try:
                pool.append(gen_dialogue(cl, scens[i % len(scens)]))
            except Exception as e:
                print("gen fail", arch, i, str(e)[:70])
        pools[arch] = pool
        print(f"generated {len(pool)} {arch} dialogues")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(pools, open(cache, "w"))
    return pools
