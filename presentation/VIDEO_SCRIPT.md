# Recording guide: prep presentation (max 15 min)

The course rules: all members speak, face cam on, **do not read the slides**, and stay under 15:00 (target 13:30 to leave slack).
Each slide's speaker notes (in the .pptx) repeat the talking points below.

| # | Slide | Speaker | Time | Talking points (say it, don't read it) |
|---|---|---|---|---|
| 1 | Title | M1 | 0:20 | This is a mid-project prep talk. Hook: "a model trained on the hills doesn't know the plains". First cycle done, second half still open. |
| 2 | Outline | M1 | 0:20 | Eight parts. Say we have a first campaign, not a finished product. |
| 3 | Problem | M1 | 0:50 | Walk the hilly → plain story. The key constraint is that target futures are never used for training. Stock and transformer analogues. |
| 4 | What shifts | M1 | 0:45 | Point at the non-overlapping histograms. There are three kinds of shift; normalisation fixes only the first. |
| 5 | Forecasters | M1 | 0:40 | Why lightweight. DLinear, TCN, PatchTST, RevIN, and which idea we took from each. |
| 6 | DA bound | M2 | 0:50 | Explain the bound term by term and map each term to a loss term. The λ* trap is the key insight. |
| 7 | Toolbox | M2 | 0:35 | Five families; we compose four of them. |
| 8 | TS-DA papers | M2 | 0:40 | One takeaway per paper, then the gap we target. |
| 9 | Datasets | M2 | 0:40 | Three pairs; say why each is a genuine domain shift. |
| 10 | Pipeline | M2 | 0:45 | Six steps. Stress the per-domain scaler and the leakage guards. |
| 11 | Other shifts | M2 | 0:25 | Energy shows a strong shift; finance a subtle one (foreshadows results). |
| 12 | LiteTCN | M3 | 0:45 | Trace one window through the model. z is where we align; the recon head gives a label-free signal. |
| 13 | CARE-DA | M3 | 0:50 | What the name stands for; one training step; the balance between the terms. |
| 14 | Loss | M3 | 0:45 | Each term → which part of the bound it controls. Weights fixed a priori. |
| 15 | Metrics | M3 | 0:40 | MASE for scale-free comparison; gap-closure ratio as our headline adaptation metric. |
| 16 | Protocol | M3 | 0:30 | Same checkpoint for every method, fixed epochs, no target labels for selection. |
| 17 | Results table | M4 | 0:50 | Call it a first campaign. v1 −SSL beat full v1. v2 is a seed-0 probe, not locked. |
| 18 | Gap closure | M4 | 0:35 | Alignment helps; DANN/CORAL hurt. Climate best 3-seed is still MMD. v2 energy/finance are provisional. |
| 19 | Curves | M4 | 0:35 | Adaptation logs. Feature MMD drops; target error for pure alignment can drift. |
| 20 | Qualitative | M4 | 0:30 | PCA before/after; one forecast window explained. |
| 21 | Insights | M4 | 0:40 | SSL hurt; that is why the second half exists. Quote the seed-0 v2 probe, then say 3 seeds are still to come. |
| 22 | Interface | M4 | 0:40 | Show the prototype, then list what is still missing (batch upload, polish). |
| 23 | Roadmap | M1 | 0:35 | First half vs second half. Ask TAs to verify datasets. Nothing is locked yet. |
| 24-25 | References + Thanks | M1 | 0:15 | Show briefly; one-sentence wrap-up. |

**Total ≈ 13:40.**

## Recording checklist
- OBS: scene = slides (window capture) + webcam overlay (bottom-right), 1080p, mic check.
- Before recording, fill in the team name and roll numbers: edit `TEAM`/`MEMBERS` at the top of `presentation/build_deck.js` and rebuild (`node build_deck.js`), or edit slide 1 directly in PowerPoint.
- Live demo: start `streamlit run app/app.py` before recording and keep the browser tab ready.
- Upload the .pptx (or a PDF export) and the video to the Google Drive folder `YourTeamName_IE643_CourseProject_Prep`, share it with the instructor and all TAs, and test the link in an incognito window.
