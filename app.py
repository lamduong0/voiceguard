"""Drag-drop web app: upload (or record) a call, get the VoiceGuard risk verdict.

  pip install -r requirements-app.txt        # gradio, on top of requirements-acoustic
  PYTHONPATH=. .venv/bin/python app.py       # opens a local page in your browser

Shares the core in voiceguard/assess.py (m2 detector, honest caveat). The first
request loads the models; later ones reuse them.
"""
import gradio as gr
from voiceguard.assess import assess

LEVEL_COLOR = {"LOW": "#2e7d32", "MEDIUM": "#ef6c00", "HIGH": "#c62828"}


def run(audio_path):
    if not audio_path:
        return "Drop in or record an audio file, then press Check.", ""
    r = assess(audio_path)
    color = LEVEL_COLOR.get(r["level"], "#555")
    verdict = (
        f"## <span style='color:{color}'>{r['level']}</span> &nbsp; risk {r['risk']:.2f}\n\n"
        f"{r['advice']}\n\n"
        f"- acoustic P(synthetic) **{r['acoustic']:.2f}** → P(clone) **{r['p_clone']:.2f}** "
        f"({r['detector'].split('/')[-1]})\n"
        f"- scam-intent **{r['intent']:.2f}** ({r['intent_source']})\n\n"
        f"> {r['caveat']}"
    )
    transcript = "\n".join(f"[{t['t']:5.1f}s] {t['text']}" for t in r["transcript"])
    return verdict, transcript or "(no speech detected)"


with gr.Blocks(title="VoiceGuard") as demo:
    gr.Markdown("# VoiceGuard\nDrop in a call recording — it transcribes, scores the voice "
                "and the conversational intent, and rates the scam risk.")
    audio = gr.Audio(type="filepath", label="Call recording (mp3 / wav / m4a)")
    btn = gr.Button("Check", variant="primary")
    verdict = gr.Markdown()
    transcript = gr.Textbox(label="Transcript", lines=6)
    btn.click(run, inputs=audio, outputs=[verdict, transcript])


if __name__ == "__main__":
    demo.launch()
