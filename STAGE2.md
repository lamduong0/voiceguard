# Stage 2 — real-data validation

Goal: re-run the exact `parallel` vs `bayesian` comparison from Stage 1, but with the
two signals produced by real models on real audio/transcripts. If the ambiguous-regime
detection edge (~+6-7 pts in sim) survives, the paper has its result.

The evaluation harness does **not** change — `voiceguard/evaluate.py` (tune + metrics) and
`voiceguard/fusion.py` (parallel/bayesian controllers) are reused verbatim. Stage 2 only
swaps the *source* of the `acoustic[N,T]` and `intent[N,T]` matrices, via
`voiceguard/realsignals.py`.

## Signals
- **Acoustic**: pretrained anti-spoofing model (AASIST / RawNet2 / wav2vec2-AASIST),
  inference only, per ~1s chunk -> synthetic-voice probability. CPU/MPS-feasible for a
  few hundred clips. No training in v1.
- **Intent**: LLM (Azure OpenAI) scores scam-intent per transcript turn -> probability.
  API-based, no GPU.

Both are aligned to a common T-step timeline (`realsignals.align_to_timeline`): acoustic =
latest chunk score per step; intent = forward-filled latest turn score (intent persists
once the "ask" is made).

## Dataset (the validity crux)
No public dataset pairs cloned-voice AUDIO with scam-INTENT dialogue in the same call. We
construct a benchmark by crossing:
- audio: genuine vs spoofed clips (ASVspoof 2019/2021 LA, In-the-Wild)
- transcript: benign vs scam dialogue (scam-call corpora, or LLM-generated + human-checked)

into the four archetypes from Stage 1 (clone_scam, genuine_benign, genuine_urgent,
clone_benign). The constructed benchmark is itself a paper artifact.

### Threats to validity (must be stated in the paper)
1. **Independence assumption.** Pairing independent audio with independent transcripts
   removes any natural correlation between acoustic artifacts and scripted delivery. This
   can over- or under-state the fusion benefit. Mitigate: report sensitivity to the
   clone/scam correlation, and seek any naturally-paired scam recordings.
2. **Domain shift.** ASVspoof TTS != real phone voice-clone scams (codec, channel,
   3-second-clone tooling). Report on In-the-Wild separately; note the gap.
3. **Hard negatives.** genuine_urgent (real, emotional, legit calls) are scarce; may need
   curated/synthesized urgent-but-genuine transcripts over genuine audio.
4. **Intent timing.** Forward-fill is an approximation of how intent emerges over turns.

## Compute plan (minimize spend)
- v1: Azure OpenAI for intent + CPU/MPS acoustic inference. **No paid GPU.**
- GPU only if we (a) fine-tune the acoustic model to the phone-scam domain, or (b) scale to
  thousands of clips. Then: prefer an owned/cheaper GPU over paid Brev.
