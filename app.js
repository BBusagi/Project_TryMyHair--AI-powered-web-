const maxUploadSizeMb = 12;

const portraitInput = document.getElementById("portraitInput");
const validateButton = document.getElementById("validateButton");
const resetButton = document.getElementById("resetButton");
const validateEndpointInput = document.getElementById("validateEndpointInput");
const mockToggle = document.getElementById("mockToggle");
const faceDetectionOption = document.getElementById("faceDetectionOption");
const landmarkOption = document.getElementById("landmarkOption");
const segmentationOption = document.getElementById("segmentationOption");
const strictnessRange = document.getElementById("strictnessRange");
const strictnessValue = document.getElementById("strictnessValue");

const portraitPreview = document.getElementById("portraitPreview");
const portraitEmpty = document.getElementById("portraitEmpty");

const validationBadge = document.getElementById("validationBadge");
const statusText = document.getElementById("statusText");
const faceCountValue = document.getElementById("faceCountValue");
const faceSizeValue = document.getElementById("faceSizeValue");
const frontalScoreValue = document.getElementById("frontalScoreValue");
const clarityScoreValue = document.getElementById("clarityScoreValue");
const brightnessScoreValue = document.getElementById("brightnessScoreValue");
const headroomScoreValue = document.getElementById("headroomScoreValue");
const occlusionScoreValue = document.getElementById("occlusionScoreValue");
const issuesList = document.getElementById("issuesList");
const ruleCheckList = document.getElementById("ruleCheckList");

const errorModal = document.getElementById("errorModal");
const modalSummary = document.getElementById("modalSummary");
const modalIssues = document.getElementById("modalIssues");
const closeModalButton = document.getElementById("closeModalButton");

const state = {
  portraitFile: null,
  portraitDataUrl: "",
};

const frontendIssueCatalog = {
  NO_IMAGE: "请先上传用户图片。",
  NOT_IMAGE_FILE: "请选择 jpg、png、webp 等图片文件。",
  FILE_TOO_LARGE: `图片文件过大。当前 POC 前端限制为 ${maxUploadSizeMb} MB 以内。`,
  FILE_READ_FAILED: "图片读取失败，请更换图片后重试。",
  VALIDATE_API_FAILED: "后端 validate-portrait 调用失败，请确认本地服务已经启动。",
};

function setBadge(text, tone = "neutral") {
  validationBadge.textContent = text;
  validationBadge.className = `badge ${tone}`;
}

function updateStrictnessOutput() {
  strictnessValue.textContent = Number(strictnessRange.value).toFixed(2);
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function getBase64Payload(dataUrl) {
  const [, base64 = ""] = String(dataUrl).split(",");
  return base64;
}

function resetMetrics() {
  faceCountValue.textContent = "-";
  faceSizeValue.textContent = "-";
  frontalScoreValue.textContent = "-";
  clarityScoreValue.textContent = "-";
  brightnessScoreValue.textContent = "-";
  headroomScoreValue.textContent = "-";
  occlusionScoreValue.textContent = "-";
  issuesList.innerHTML = "<li>尚未执行后端检测</li>";
  ruleCheckList.innerHTML = "<li>尚未执行后端检测</li>";
  setBadge("待检测", "neutral");
  statusText.textContent = "上传图片后调用后端检测";
}

function resetFlow() {
  portraitInput.value = "";
  state.portraitFile = null;
  state.portraitDataUrl = "";
  portraitPreview.src = "";
  portraitPreview.style.display = "none";
  portraitEmpty.hidden = false;
  hideErrorModal();
  resetMetrics();
}

function showIssues(target, issues, fallbackText = "未发现问题") {
  const safeIssues = issues.length ? issues : [{ message: fallbackText }];
  target.innerHTML = safeIssues
    .map((issue) => `<li>${issue.message || issue.code || "未知问题"}</li>`)
    .join("");
}

function showRuleChecks(checks) {
  if (!checks || checks.length === 0) {
    ruleCheckList.innerHTML = "<li>后端未返回硬规则详情</li>";
    return;
  }

  ruleCheckList.innerHTML = checks
    .map((check) => `<li class="${check.status || "warn"}">${check.label}：${check.detail}</li>`)
    .join("");
}

function showErrorModal(issues) {
  const safeIssues = issues.length
    ? issues
    : [{ code: "VALIDATE_API_FAILED", message: frontendIssueCatalog.VALIDATE_API_FAILED }];

  modalSummary.textContent = safeIssues[0].message || frontendIssueCatalog[safeIssues[0].code];
  showIssues(modalIssues, safeIssues);
  errorModal.hidden = false;
}

function hideErrorModal() {
  errorModal.hidden = true;
}

function validateFileBeforePreview(file) {
  if (!file) {
    return [{ code: "NO_IMAGE", message: frontendIssueCatalog.NO_IMAGE }];
  }

  if (!file.type.startsWith("image/")) {
    return [{ code: "NOT_IMAGE_FILE", message: frontendIssueCatalog.NOT_IMAGE_FILE }];
  }

  if (file.size > maxUploadSizeMb * 1024 * 1024) {
    return [{ code: "FILE_TOO_LARGE", message: frontendIssueCatalog.FILE_TOO_LARGE }];
  }

  return [];
}

async function validatePortraitViaApi(endpoint, strictness) {
  const payload = {
    portraitBase64: getBase64Payload(state.portraitDataUrl),
    strictness,
    validationOptions: {
      faceDetection: faceDetectionOption.checked,
      landmarkAlignment: landmarkOption.checked,
      hairSegmentation: segmentationOption.checked,
    },
  };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json();
}

function applyValidationResult(result) {
  faceCountValue.textContent =
    typeof result.faceCount === "number" ? String(result.faceCount) : "-";
  faceSizeValue.textContent = formatScore(result.faceSizeScore);
  frontalScoreValue.textContent = formatScore(result.frontalScore);
  clarityScoreValue.textContent = formatScore(result.clarityScore);
  brightnessScoreValue.textContent = formatScore(result.brightnessScore);
  headroomScoreValue.textContent = formatScore(result.headroomScore);
  occlusionScoreValue.textContent = formatScore(result.occlusionScore);
  showRuleChecks(result.checks || []);
  showIssues(
    issuesList,
    result.issues || [],
    result.valid ? "后端检测通过" : "后端未返回问题列表",
  );

  if (result.valid) {
    setBadge("后端检测通过", "success");
    statusText.textContent = "图片满足后端 validatePortrait 的当前规则";
    return;
  }

  setBadge("检测失败", "danger");
  statusText.textContent = "图片不符合后端 validatePortrait 的输入要求";
  showErrorModal(result.issues || []);
}

async function handlePortraitChange(event) {
  const [file] = event.target.files;
  const fileIssues = validateFileBeforePreview(file);

  if (fileIssues.length > 0) {
    resetFlow();
    showErrorModal(fileIssues);
    return;
  }

  try {
    const dataUrl = await fileToDataUrl(file);
    state.portraitFile = file;
    state.portraitDataUrl = dataUrl;
    portraitPreview.src = dataUrl;
    portraitPreview.style.display = "block";
    portraitEmpty.hidden = true;
    resetMetrics();
    statusText.textContent = "图片已通过前端文件校验，可以调用后端检测";
  } catch (error) {
    resetFlow();
    showErrorModal([
      {
        code: "FILE_READ_FAILED",
        message: frontendIssueCatalog.FILE_READ_FAILED,
      },
    ]);
    console.error(error);
  }
}

async function handleValidate() {
  if (!state.portraitFile) {
    showErrorModal([{ code: "NO_IMAGE", message: frontendIssueCatalog.NO_IMAGE }]);
    return;
  }

  setBadge("调用后端中", "warning");
  statusText.textContent = "正在调用后端 validatePortrait";

  try {
    const strictness = Number(strictnessRange.value);
    const result = await validatePortraitViaApi(validateEndpointInput.value.trim(), strictness);
    applyValidationResult(result);
  } catch (error) {
    setBadge("接口异常", "danger");
    statusText.textContent = frontendIssueCatalog.VALIDATE_API_FAILED;
    showRuleChecks([
      {
        label: "后端连接",
        status: "fail",
        detail: "无法获取 validate-portrait 响应",
      },
    ]);
    showErrorModal([
      {
        code: "VALIDATE_API_FAILED",
        message: frontendIssueCatalog.VALIDATE_API_FAILED,
      },
    ]);
    console.error(error);
  }
}

portraitInput.addEventListener("change", handlePortraitChange);
validateButton.addEventListener("click", handleValidate);
resetButton.addEventListener("click", resetFlow);
closeModalButton.addEventListener("click", hideErrorModal);
strictnessRange.addEventListener("input", updateStrictnessOutput);

errorModal.addEventListener("click", (event) => {
  if (event.target === errorModal) {
    hideErrorModal();
  }
});

mockToggle.checked = false;
mockToggle.disabled = true;
mockToggle.parentElement.querySelector("span").textContent =
  "前端只做文件校验；肖像、人脸、头发相关判断统一交给后端";

window.addEventListener("DOMContentLoaded", () => {
  updateStrictnessOutput();
  resetFlow();
});
