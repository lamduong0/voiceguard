"""Drag-drop web app: upload (or record) a call, get the VoiceGuard risk verdict.

  pip install -r requirements-app.txt        # gradio, on top of requirements-acoustic
  PYTHONPATH=. .venv/bin/python app.py       # opens a local page in your browser

Shares the core in voiceguard/assess.py (m2 detector, honest caveat). The first
request loads the models; later ones reuse them.
"""
import gradio as gr
from voiceguard.assess import assess
from voiceguard.detect import format_score

LEVEL_COLOR = {"LOW": "#2e7d32", "MEDIUM": "#ef6c00", "HIGH": "#c62828"}


def run(audio_path):
    if not audio_path:
        return "Drop in or record an audio file, then press Check.", ""
    r = assess(audio_path)
    color = LEVEL_COLOR.get(r["level"], "#555")
    seg = ""
    if r["peak_segment"]:
        ps = r["peak_segment"]
        seg = (f"\n- most-synthetic segment **{ps['start']}–{ps['end']}s** "
               f"(whole-file mean {r['acoustic_overall']:.2f}, "
               f"{int(r['frac_synthetic'] * 100)}% of scored windows above 0.5)")
    reasons = "\n\n".join(r["reasons"])
    verdict = (
        f"## <span style='color:{color}'>{r['level'].replace('_', ' ')}</span>\n\n"
        f"Voice-clone scam score **{format_score(r['risk'])}**\n\n"
        f"{r['advice']}\n\n{reasons}\n\n"
        f"- acoustic score **{format_score(r['acoustic'])}** → clone score **{format_score(r['clone_score'])}** "
        f"({r['detector'].split('/')[-1]}, {r['pooling']}){seg}\n"
        f"- scam-intent **{format_score(r['intent'])}** ({r['intent_source']})\n\n"
        f"> {r['caveat']}"
    )
    transcript = "\n".join(f"[{t['t']:5.1f}s] {t['text']}" for t in r["transcript"])
    return verdict, transcript or "(no speech detected)"


with gr.Blocks(title="VoiceGuard") as demo:
    gr.Markdown("# VoiceGuard\nDrop in a call recording — it transcribes, scores the voice "
                "and the conversational intent, and scores voice-clone scam risk.")
    audio = gr.Audio(type="filepath", label="Call recording (mp3 / wav / m4a)")
    btn = gr.Button("Check", variant="primary")
    verdict = gr.Markdown()
    transcript = gr.Textbox(label="Transcript", lines=6)
    btn.click(run, inputs=audio, outputs=[verdict, transcript])


if __name__ == "__main__":
    demo.launch()
