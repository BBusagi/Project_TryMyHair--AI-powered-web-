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
- `StableHairAdapter`: inspects repo / runner / required weights, writes per-request yaml config, and builds a backend runner command.

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

It also implements:

- `GET /stable-hair/status`
- `POST /validate-hair-reference`
- `POST /generate-hairstyle`

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

## Stable-Hair integration

The browser never calls Stable-Hair directly.

Current flow:

1. Frontend uploads portrait A to `POST /validate-portrait`.
2. Frontend uploads hair reference B to `POST /validate-hair-reference`.
3. Frontend uploads portrait A and hair reference B to `POST /generate-hairstyle`.
4. Backend saves inputs under `uploads/stable-hair/<request-id>/`.
5. Backend writes a prepared `hair_transfer.yaml`.
6. Backend returns `commandText`.
7. Real model execution is skipped unless the API payload explicitly sets `executeModel: true`.

Status check:

```bash
curl http://127.0.0.1:8000/stable-hair/status
```

Required upstream weights:

```text
external_models/Stable-Hair/models/stage1/pytorch_model.bin
external_models/Stable-Hair/models/stage2/pytorch_model.bin
external_models/Stable-Hair/models/stage2/pytorch_model_1.bin
external_models/Stable-Hair/models/stage2/pytorch_model_2.bin
```

Recommended dependency boundary:

```bash
cd /mnt/d/GitProject/TryMyHair/external_models/Stable-Hair
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If you use another Python environment, start FastAPI with:

```bash
export STABLE_HAIR_PYTHON=/path/to/stable-hair/python
bash scripts/run_backend.sh
```

`/stable-hair/status` also checks whether that Python can import `torch`, `diffusers`, and `omegaconf`.

## Hair reference validation

Current `POST /validate-hair-reference` is a lightweight gate for reference image B.

It checks:

- image decode
- resolution
- brightness
- clarity proxy
- face/head detection
- multi-face warning
- hair crop risk proxy based on face box

TODO:

- replace crop-risk proxy with real hair segmentation / face parsing
- return a visual hair mask preview
- validate hair area, hair border contact, occluders, and dominant hair color
