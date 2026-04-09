from __future__ import annotations

import base64
from ctypes.util import find_library
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageStat

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    np = None
    mp_python = None
    mp_vision = None


app = FastAPI(title="HairDesigner Backend POC")
PROJECT_DIR = Path(__file__).resolve().parents[1]
MEDIAPIPE_FACE_MODEL_PATH = PROJECT_DIR / "backend/models/blaze_face_short_range.tflite"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValidationOptions(BaseModel):
    faceDetection: bool = True
    landmarkAlignment: bool = False
    hairSegmentation: bool = False


class ValidatePortraitRequest(BaseModel):
    portraitBase64: str
    strictness: float = Field(default=0.7, ge=0, le=1)
    validationOptions: ValidationOptions = Field(default_factory=ValidationOptions)


class GenerateHairstyleRequest(BaseModel):
    portraitBase64: str
    hairstylePrompt: str = ""
    negativePrompt: str = ""
    identityStrength: float = 0.88
    editStrength: float = 0.52
    composedPrompt: str = ""


def decode_base64_image(image_base64: str) -> Image.Image:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    data = base64.b64decode(image_base64)
    return Image.open(BytesIO(data)).convert("RGB")


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_brightness(image: Image.Image) -> float:
    grayscale = image.convert("L").resize((128, 128))
    mean = ImageStat.Stat(grayscale).mean[0]
    return clamp_score(mean / 255)


def estimate_clarity(image: Image.Image) -> float:
    grayscale = image.convert("L").resize((96, 96))
    pixels = grayscale.load()
    diff_sum = 0
    count = 0

    for y in range(grayscale.height - 1):
        for x in range(grayscale.width - 1):
            current = pixels[x, y]
            diff_sum += abs(current - pixels[x + 1, y])
            diff_sum += abs(current - pixels[x, y + 1])
            count += 2

    return clamp_score((diff_sum / max(count, 1)) / 42)


def issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def check(label: str, status: str, detail: str) -> dict[str, str]:
    return {"label": label, "status": status, "detail": detail}


def thresholds(strictness: float) -> dict[str, float]:
    return {
        "min_size_warn": 512 if strictness > 0.8 else 320,
        "min_size_block": 192,
        "brightness_warn": 0.14 + strictness * 0.1,
        "brightness_block": 0.08,
        "clarity_warn": 0.05 + strictness * 0.07,
        "clarity_block": 0.035,
        "face_height_ratio_warn": 0.16 + strictness * 0.06,
        "face_height_ratio_block": 0.1,
        "frontal_warn": 0.35 + strictness * 0.12,
        "frontal_block": 0.15,
        "headroom_warn": 0.01 + strictness * 0.02,
        "large_secondary_face": 0.14,
        "media_pipe_confidence": 0.32 + strictness * 0.18,
    }


def get_mediapipe_runtime_error() -> str | None:
    if mp is None or np is None or mp_python is None or mp_vision is None:
        return "MediaPipe Python package is not installed."

    if not MEDIAPIPE_FACE_MODEL_PATH.exists():
        return f"MediaPipe face detector model is missing: {MEDIAPIPE_FACE_MODEL_PATH}"

    # MediaPipe Tasks Python loads libmediapipe_c.so, which depends on GLES on Linux.
    if find_library("GLESv2") is None:
        return "System library libGLESv2.so.2 is missing. On Ubuntu/WSL install package: libgles2"

    return None


def relative_bbox_to_dict(relative_bbox: Any) -> dict[str, float]:
    return {
        "x": clamp_score(relative_bbox.xmin),
        "y": clamp_score(relative_bbox.ymin),
        "width": clamp_score(relative_bbox.width),
        "height": clamp_score(relative_bbox.height),
    }


def estimate_frontal_score(relative_keypoints: list[Any]) -> float | None:
    if len(relative_keypoints) < 3:
        return None

    right_eye = relative_keypoints[0]
    left_eye = relative_keypoints[1]
    nose = relative_keypoints[2]
    eye_distance = max(abs(left_eye.x - right_eye.x), 0.001)
    eye_mid_x = (left_eye.x + right_eye.x) / 2

    nose_offset = abs(nose.x - eye_mid_x) / eye_distance
    eye_tilt = abs(left_eye.y - right_eye.y) / eye_distance
    score = 1 - min(nose_offset / 0.36, 1) * 0.75 - min(eye_tilt / 0.28, 1) * 0.25
    return clamp_score(score)


def detect_faces_with_mediapipe(image: Image.Image, strictness: float) -> dict[str, Any]:
    if mp is None or np is None or mp_python is None or mp_vision is None:
        raise RuntimeError("MediaPipe package is not installed.")

    runtime_error = get_mediapipe_runtime_error()
    if runtime_error:
        raise RuntimeError(runtime_error)

    threshold = thresholds(strictness)["media_pipe_confidence"]
    image_array = np.asarray(image)

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_FACE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=threshold,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    with mp_vision.FaceDetector.create_from_options(options) as detector:
        results = detector.detect(mp_image)

    detections = results.detections or []
    faces = []

    for detection in detections:
        box = detection.bounding_box
        confidence = float(detection.categories[0].score) if detection.categories else 0.0
        bbox = {
            "x": clamp_score(box.origin_x / image.width),
            "y": clamp_score(box.origin_y / image.height),
            "width": clamp_score(box.width / image.width),
            "height": clamp_score(box.height / image.height),
        }
        faces.append(
            {
                "bbox": bbox,
                "confidence": confidence,
                "frontalScore": estimate_frontal_score(detection.keypoints),
            }
        )

    faces.sort(key=lambda face: face["bbox"]["width"] * face["bbox"]["height"], reverse=True)
    return {"detectorBackend": "mediapipe", "faceCount": len(faces), "faces": faces}


def detect_faces_with_opencv(image: Image.Image, strictness: float) -> dict[str, Any]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV fallback is not installed.")

    image_array = np.asarray(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    raw_faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(48, 48),
    )

    faces = []
    for x, y, width, height in raw_faces:
        bbox = {
            "x": clamp_score(float(x) / image.width),
            "y": clamp_score(float(y) / image.height),
            "width": clamp_score(float(width) / image.width),
            "height": clamp_score(float(height) / image.height),
        }
        faces.append(
            {
                "bbox": bbox,
                "confidence": None,
                # Haar frontal cascade is already a frontal-face detector; keep this conservative.
                "frontalScore": 0.7,
            }
        )

    faces.sort(key=lambda face: face["bbox"]["width"] * face["bbox"]["height"], reverse=True)
    return {"detectorBackend": "opencv-haar", "faceCount": len(faces), "faces": faces}


def detect_faces(image: Image.Image, strictness: float) -> dict[str, Any]:
    try:
        return detect_faces_with_mediapipe(image, strictness)
    except Exception as media_pipe_error:
        try:
            result = detect_faces_with_opencv(image, strictness)
            result["fallbackReason"] = str(media_pipe_error)
            return result
        except Exception as opencv_error:
            raise RuntimeError(
                f"Face detection failed. MediaPipe: {media_pipe_error}; OpenCV: {opencv_error}"
            ) from opencv_error


def add_face_detection_result(
    *,
    result: dict[str, Any],
    checks: list[dict[str, str]],
    issues: list[dict[str, str]],
    image: Image.Image,
    strictness: float,
) -> None:
    face_result = detect_faces(image, strictness)
    face_count = face_result["faceCount"]
    result["faceCount"] = face_count
    result["faces"] = face_result["faces"]
    result["faceDetectionBackend"] = face_result["detectorBackend"]
    threshold = thresholds(strictness)

    if face_count == 0:
        issues.append(issue("NO_FACE_DETECTED", "未检测到有效人脸。"))
        checks.append(check("Face Detection", "fail", "0 张脸"))
        return

    large_faces = [
        face
        for face in face_result["faces"]
        if face["bbox"]["height"] >= threshold["large_secondary_face"]
    ]

    if len(large_faces) > 1:
        issues.append(issue("MULTIPLE_PEOPLE", f"检测到 {len(large_faces)} 张主要人脸，当前仅支持单人肖像。"))
        checks.append(check("是否单人", "fail", f"{len(large_faces)} 张主要人脸"))
    elif face_count > 1:
        checks.append(check("是否单人", "warn", f"检测到 {face_count} 张脸；仅 1 张主要人脸"))
    else:
        checks.append(check("是否单人", "pass", "1 张脸"))

    largest_face = face_result["faces"][0]
    bbox = largest_face["bbox"]
    face_height_ratio = bbox["height"]
    face_size_score = clamp_score(face_height_ratio / 0.45)
    frontal_score = largest_face["frontalScore"]
    headroom = bbox["y"]
    headroom_score = clamp_score(headroom / 0.15)

    result["faceSizeScore"] = face_size_score
    result["frontalScore"] = frontal_score
    result["headroomScore"] = headroom_score

    if face_height_ratio < threshold["face_height_ratio_block"]:
        issues.append(issue("FACE_TOO_SMALL", "脸部在画面中极小，后续对齐和迁移会不稳定。"))
        checks.append(check("脸是否足够大", "fail", f"脸部高度占图 {face_height_ratio:.0%}"))
    elif face_height_ratio < threshold["face_height_ratio_warn"]:
        checks.append(check("脸是否足够大", "warn", f"偏小；脸部高度占图 {face_height_ratio:.0%}"))
    else:
        checks.append(check("脸是否足够大", "pass", f"脸部高度占图 {face_height_ratio:.0%}"))

    if frontal_score is None:
        checks.append(check("是否正脸/近正脸", "warn", "MediaPipe detection 未返回足够关键点"))
    elif frontal_score < threshold["frontal_block"]:
        issues.append(issue("PROFILE_TOO_STRONG", "侧脸角度过大，建议上传更正面的肖像。"))
        checks.append(check("是否正脸/近正脸", "fail", f"{frontal_score:.2f}"))
    elif frontal_score < threshold["frontal_warn"]:
        checks.append(check("是否正脸/近正脸", "warn", f"{frontal_score:.2f}；姿态可能影响迁移"))
    else:
        checks.append(check("是否正脸/近正脸", "pass", f"{frontal_score:.2f}"))

    if headroom < threshold["headroom_warn"]:
        checks.append(check("头顶是否被截断", "warn", f"脸框上边距 {headroom:.0%}；建议保留头发上方空间"))
    else:
        checks.append(check("头顶是否被截断", "pass", f"脸框上边距 {headroom:.0%}"))

    confidence = largest_face["confidence"]
    confidence_text = f"{confidence:.2f}" if confidence is not None else "N/A"
    detail = f"{face_result['detectorBackend']} / 最大脸置信度 {confidence_text}"
    if face_result.get("fallbackReason"):
        detail += " / MediaPipe fallback"
    checks.append(check("Face Detection", "pass", detail))


def validate_portrait(payload: ValidatePortraitRequest) -> dict[str, Any]:
    strictness = payload.strictness
    image = decode_base64_image(payload.portraitBase64)
    threshold = thresholds(strictness)

    brightness = estimate_brightness(image)
    clarity = estimate_clarity(image)

    issues: list[dict[str, str]] = []
    checks: list[dict[str, str]] = []

    min_dimension = min(image.width, image.height)
    resolution_status = "pass"
    brightness_status = "pass"
    clarity_status = "pass"

    if min_dimension < threshold["min_size_block"]:
        resolution_status = "fail"
        issues.append(issue("LOW_RESOLUTION", "图片分辨率极低，建议上传更清晰的肖像图。"))
    elif min_dimension < threshold["min_size_warn"]:
        resolution_status = "warn"

    if brightness < threshold["brightness_block"]:
        brightness_status = "fail"
        issues.append(issue("TOO_DARK", "画面严重过暗，建议换一张明亮、均匀照明的肖像。"))
    elif brightness < threshold["brightness_warn"]:
        brightness_status = "warn"

    if clarity < threshold["clarity_block"]:
        clarity_status = "fail"
        issues.append(issue("BLURRY_IMAGE", "图片严重模糊，人脸检测或迁移可能失败。"))
    elif clarity < threshold["clarity_warn"]:
        clarity_status = "warn"

    result: dict[str, Any] = {
        "valid": False,
        "faceCount": None,
        "faceSizeScore": None,
        "frontalScore": None,
        "clarityScore": clarity,
        "brightnessScore": brightness,
        "headroomScore": None,
        "occlusionScore": None,
        "validationOptions": payload.validationOptions.model_dump(),
        "faceDetectionBackend": None,
        "checks": checks,
        "issues": issues,
    }

    if payload.validationOptions.faceDetection:
        try:
            add_face_detection_result(
                result=result,
                checks=checks,
                issues=issues,
                image=image,
                strictness=strictness,
            )
        except RuntimeError as error:
            issues.append(issue("FACE_DETECTION_UNAVAILABLE", str(error)))
            checks.append(check("Face Detection", "fail", "后端依赖未安装或不可用"))
    else:
        checks.append(check("Face Detection", "warn", "已由前端选项跳过"))

    if payload.validationOptions.landmarkAlignment:
        checks.append(check("Landmark / Alignment", "warn", "当前 POC 不单独执行；后续可接 Face Mesh"))

    if payload.validationOptions.hairSegmentation:
        checks.append(check("Hair Segmentation / Parsing", "warn", "当前 POC 不单独执行；后续可接 parsing"))

    checks.extend(
        [
            check("是否有大面积遮挡", "warn", "当前由后续 landmark / parsing / VLM 再确认"),
            check("光照是否过暗", brightness_status, f"{brightness:.2f}"),
            check("分辨率是否太低", resolution_status, f"{image.width}x{image.height}"),
            check("是否明显模糊", clarity_status, f"{clarity:.2f}"),
        ]
    )

    result["valid"] = len(issues) == 0
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    media_pipe_runtime_error = get_mediapipe_runtime_error()
    return {
        "ok": True,
        "mediapipePackageInstalled": mp is not None,
        "mediapipeModelExists": MEDIAPIPE_FACE_MODEL_PATH.exists(),
        "mediapipeRuntimeReady": media_pipe_runtime_error is None,
        "mediapipeRuntimeError": media_pipe_runtime_error,
        "opencvInstalled": cv2 is not None,
    }


@app.post("/validate-portrait")
def validate_portrait_endpoint(payload: ValidatePortraitRequest) -> dict[str, Any]:
    try:
        return validate_portrait(payload)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/generate-hairstyle")
def generate_hairstyle_endpoint(payload: GenerateHairstyleRequest) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail="发型迁移接口尚未接入具体模型。")
