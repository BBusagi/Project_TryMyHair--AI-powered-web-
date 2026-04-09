# HairDesigner Backend Integration Draft

This directory is a thin integration layer for local hair-transfer repositories.

It does not install dependencies, download weights, or guarantee inference yet. The goal is to keep third-party repos outside this project while giving the frontend a stable route to call later.

## Fetch model repos

From `HTML/HairDesigner`:

```bash
bash scripts/clone_model_repos.sh
```

The script clones shallow copies into `external_models/`.

That directory is intentionally git-ignored.

## Current adapter status

- `BarbershopAdapter`: maps source / shape / color inputs to the upstream `main.py` style command.
- `HairFastGANAdapter`: maps source / shape / color inputs to upstream `main.py` and writes one result path.
- `StableHairAdapter`: placeholder only. Upstream `infer_full.py` is config-heavy; keep it behind a backend route instead of calling it directly from the browser.

## Important runtime gap

The clone script fetches source code only.

Before inference you still need each upstream project's dependency and weight setup:

- Barbershop: upstream README asks for II2S / StyleGAN / segmentation related pretrained files.
- HairFastGAN: upstream README downloads pretrained models from the Hugging Face project and moves them into `pretrained_models/`.
- Stable-Hair: upstream project depends on Stable Diffusion / ControlNet / Stable-Hair stage weights and config-driven inference.

Keep this as a backend task. The static HTML page should never load these models directly.

## Next implementation step

Add a local API server with endpoints:

- `POST /validate-portrait`
- `POST /generate-hairstyle`

Do not expose model repos or OpenAI keys to static frontend code.

## FastAPI validation server

`server.py` is a FastAPI draft.

Install:

```bash
cd /mnt/d/GitProject/TryMyHair
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
```

Run from project root:

```bash
.venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

Or:

```bash
bash scripts/run_backend.sh
```

It currently implements `POST /validate-portrait`.

Current status:

- basic file payload decode
- brightness proxy
- blur proxy
- resolution rule
- MediaPipe Face Detection when `mediapipe`, model file, and system graphics runtime are available
- OpenCV Haar face detection fallback

It intentionally does not yet run landmark alignment or hair segmentation.

## Health check

```bash
curl http://127.0.0.1:8000/health
```

The response includes:

- `mediapipePackageInstalled`
- `mediapipeModelExists`
- `mediapipeRuntimeReady`
- `mediapipeRuntimeError`
- `opencvInstalled`

## MediaPipe on WSL

In this environment MediaPipe imported successfully, but the Tasks runtime needed an extra system library:

```text
libGLESv2.so.2
```

If `/health` reports that error, install the Ubuntu package:

```bash
sudo apt update
sudo apt install libgles2
```

Until then, `/validate-portrait` falls back to OpenCV Haar face detection.
