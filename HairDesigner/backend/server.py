from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageStat

try:
    import mediapipe as mp
    import numpy as np
except ImportError:
    mp = None
    np = None


app = FastAPI(title="HairDesigner Backend POC")

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
        "min_size": 512 if strictness > 0.75 else 320,
        "brightness": 0.18 + strictness * 0.12,
        "clarity": 0.08 + strictness * 0.08,
        "face_height_ratio": 0.2 + strictness * 0.08,
        "frontal": 0.45 + strictness * 0.15,
        "headroom": 0.015 + strictness * 0.025,
        "media_pipe_confidence": 0.45 + strictness * 0.2,
    }


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


def detect_faces(image: Image.Image, strictness: float) -> dict[str, Any]:
    if mp is None or np is None:
        raise RuntimeError(
            "MediaPipe is not installed. Install backend requirements before enabling Face Detection."
        )

    threshold = thresholds(strictness)["media_pipe_confidence"]
    image_array = np.asarray(image)
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=threshold,
    )

    with face_detection as detector:
        results = detector.process(image_array)

    detections = results.detections or []
    faces = []

    for detection in detections:
        location = detection.location_data
        relative_bbox = location.relative_bounding_box
        confidence = float(detection.score[0]) if detection.score else 0.0
        bbox = relative_bbox_to_dict(relative_bbox)
        faces.append(
            {
                "bbox": bbox,
                "confidence": confidence,
                "frontalScore": estimate_frontal_score(list(location.relative_keypoints)),
            }
        )

    faces.sort(key=lambda face: face["bbox"]["width"] * face["bbox"]["height"], reverse=True)
    return {"faceCount": len(faces), "faces": faces}


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

    if face_count == 0:
        issues.append(issue("NO_FACE_DETECTED", "未检测到有效人脸。"))
        checks.append(check("Face Detection", "fail", "0 张脸"))
        return

    if face_count > 1:
        issues.append(issue("MULTIPLE_PEOPLE", f"检测到 {face_count} 张脸，当前仅支持单人肖像。"))
        checks.append(check("是否单人", "fail", f"{face_count} 张脸"))
    else:
        checks.append(check("是否单人", "pass", "1 张脸"))

    largest_face = face_result["faces"][0]
    bbox = largest_face["bbox"]
    face_height_ratio = bbox["height"]
    face_size_score = clamp_score(face_height_ratio / 0.45)
    frontal_score = largest_face["frontalScore"]
    headroom = bbox["y"]
    headroom_score = clamp_score(headroom / 0.15)
    threshold = thresholds(strictness)

    result["faceSizeScore"] = face_size_score
    result["frontalScore"] = frontal_score
    result["headroomScore"] = headroom_score

    if face_height_ratio < threshold["face_height_ratio"]:
        issues.append(issue("FACE_TOO_SMALL", "脸部在画面中太小，后续对齐和迁移会不稳定。"))
        checks.append(check("脸是否足够大", "fail", f"脸部高度占图 {face_height_ratio:.0%}"))
    else:
        checks.append(check("脸是否足够大", "pass", f"脸部高度占图 {face_height_ratio:.0%}"))

    if frontal_score is None:
        checks.append(check("是否正脸/近正脸", "warn", "MediaPipe detection 未返回足够关键点"))
    elif frontal_score < threshold["frontal"]:
        issues.append(issue("PROFILE_TOO_STRONG", "侧脸角度过大，建议上传更正面的肖像。"))
        checks.append(check("是否正脸/近正脸", "fail", f"{frontal_score:.2f}"))
    else:
        checks.append(check("是否正脸/近正脸", "pass", f"{frontal_score:.2f}"))

    if headroom < threshold["headroom"]:
        issues.append(issue("HEAD_TOP_CUT", "脸框距离画面顶部过近；请换一张头顶留有空间的照片。"))
        checks.append(check("头顶是否被截断", "fail", f"脸框上边距 {headroom:.0%}"))
    else:
        checks.append(check("头顶是否被截断", "pass", f"脸框上边距 {headroom:.0%}"))

    checks.append(check("Face Detection", "pass", f"最大脸置信度 {largest_face['confidence']:.2f}"))


def validate_portrait(payload: ValidatePortraitRequest) -> dict[str, Any]:
    strictness = payload.strictness
    image = decode_base64_image(payload.portraitBase64)
    threshold = thresholds(strictness)

    brightness = estimate_brightness(image)
    clarity = estimate_clarity(image)

    issues: list[dict[str, str]] = []
    checks: list[dict[str, str]] = []

    resolution_ok = image.width >= threshold["min_size"] and image.height >= threshold["min_size"]
    brightness_ok = brightness >= threshold["brightness"]
    clarity_ok = clarity >= threshold["clarity"]

    if not resolution_ok:
        issues.append(issue("LOW_RESOLUTION", "图片分辨率过低，建议上传更清晰的肖像图。"))

    if not brightness_ok:
        issues.append(issue("TOO_DARK", "画面光照过暗，建议换一张明亮、均匀照明的肖像。"))

    if not clarity_ok:
        issues.append(issue("BLURRY_IMAGE", "图片清晰度不足，存在明显模糊。"))

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
            check("光照是否过暗", "pass" if brightness_ok else "fail", f"{brightness:.2f}"),
            check("分辨率是否太低", "pass" if resolution_ok else "fail", f"{image.width}x{image.height}"),
            check("是否明显模糊", "pass" if clarity_ok else "fail", f"{clarity:.2f}"),
        ]
    )

    result["valid"] = len(issues) == 0
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mediapipeInstalled": mp is not None,
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
