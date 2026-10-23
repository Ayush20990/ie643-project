# DeepGen recording scripts

Open `CODE_WALKTHROUGH.html` and `DASHBOARD_DEMO.html` in a browser. Both work offline and include a rehearsal clock, speaker timeline, screen actions, spoken text and printable views. The code script also includes verified line references and excerpts.

- Code: Rishik 0:00-4:35; Ayush 4:35-9:15. Leave 45 seconds within the maximum ten minutes.
- Demo: Ayush 0:00-2:30; Rishik 2:30-4:45. Reserve 15 seconds for transitions in the five-minute video.

Use both faces in the code recording, readable screen text and clear audio. Keep these scripts on another device or non-shared screen. Rehearse the real navigation. Do not run training or download data during recording. `source_map.json` contains the script text and exact source anchors; regenerate after source changes with `python -m src.build_recording_guides`.

The app is launched from the repository root with `python -m streamlit run app/app.py`. In the extracted complete package, `python START_APP.py` also works. Use the URL printed by the server. Initial demo state: climate, Next hour, Peak-calibrated hybrid, Domain 1 source, bundled sample, input-only checkbox off. Upload the Delhi sample after switching to the target domain.
