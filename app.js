const maxUploadSizeMb = 12;

const portraitInput = document.getElementById("portraitInput");
const referenceHairInput = document.getElementById("referenceHairInput");
const validateButton = document.getElementById("validateButton");
const resetButton = document.getElementById("resetButton");
const checkStableHairButton = document.getElementById("checkStableHairButton");
const prepareStableHairButton = document.getElementById("prepareStableHairButton");
const validateEndpointInput = document.getElementById("validateEndpointInput");
const hairReferenceEndpointInput = document.getElementById("hairReferenceEndpointInput");
const generateEndpointInput = document.getElementById("generateEndpointInput");
const validateHairReferenceButton = document.getElementById("validateHairReferenceButton");
const executeStableHairToggle = document.getElementById("executeStableHairToggle");
const mockToggle = document.getElementById("mockToggle");
const faceDetectionOption = document.getElementById("faceDetectionOption");
const landmarkOption = document.getElementById("landmarkOption");
const segmentationOption = document.getElementById("segmentationOption");
const strictnessRange = document.getElementById("strictnessRange");
const strictnessValue = document.getElementById("strictnessValue");

const portraitPreview = document.getElementById("portraitPreview");
const portraitEmpty = document.getElementById("portraitEmpty");
const referenceHairPreview = document.getElementById("referenceHairPreview");
const referenceHairEmpty = document.getElementById("referenceHairEmpty");
const stableHairResultPanel = document.getElementById("stableHairResultPanel");
const stableHairResultPreview = document.getElementById("stableHairResultPreview");

const validationBadge = document.getElementById("validationBadge");
const stableHairStatusBadge = document.getElementById("stableHairStatusBadge");
const statusText = document.getElementById("statusText");
const resultSummary = document.getElementById("resultSummary");
const resultTitle = document.getElementById("resultTitle");
const resultDetail = document.getElementById("resultDetail");
const faceCountValue = document.getElementById("faceCountValue");
const faceSizeValue = document.getElementById("faceSizeValue");
const frontalScoreValue = document.getElementById("frontalScoreValue");
const clarityScoreValue = document.getElementById("clarityScoreValue");
const brightnessScoreValue = document.getElementById("brightnessScoreValue");
const headroomScoreValue = document.getElementById("headroomScoreValue");
const occlusionScoreValue = document.getElementById("occlusionScoreValue");
const issuesList = document.getElementById("issuesList");
const ruleCheckList = document.getElementById("ruleCheckList");
const stableHairSummary = document.getElementById("stableHairSummary");
const stableHairTitle = document.getElementById("stableHairTitle");
const stableHairDetail = document.getElementById("stableHairDetail");
const hairReferenceSummary = document.getElementById("hairReferenceSummary");
const hairReferenceTitle = document.getElementById("hairReferenceTitle");
const hairReferenceDetail = document.getElementById("hairReferenceDetail");
const hairReferenceCheckList = document.getElementById("hairReferenceCheckList");
const stableHairStatusList = document.getElementById("stableHairStatusList");
const stableHairResponseText = document.getElementById("stableHairResponseText");

const errorModal = document.getElementById("errorModal");
const modalSummary = document.getElementById("modalSummary");
const modalIssues = document.getElementById("modalIssues");
const closeModalButton = document.getElementById("closeModalButton");

const state = {
  portraitFile: null,
  portraitDataUrl: "",
  referenceHairFile: null,
  referenceHairDataUrl: "",
};

const frontendIssueCatalog = {
  NO_IMAGE: "请先上传用户图片。",
  NOT_IMAGE_FILE: "请选择 jpg、png、webp 等图片文件。",
  FILE_TOO_LARGE: `图片文件过大。当前 POC 前端限制为 ${maxUploadSizeMb} MB 以内。`,
  FILE_READ_FAILED: "图片读取失败，请更换图片后重试。",
  VALIDATE_API_FAILED: "后端 validate-portrait 调用失败，请确认本地服务已经启动。",
  NO_REFERENCE_IMAGE: "请先上传发型参考图 B。",
  GENERATE_API_FAILED: "后端 generate-hairstyle 调用失败，请确认本地服务已经启动。",
};

function setBadge(text, tone = "neutral") {
  validationBadge.textContent = text;
  validationBadge.className = `badge ${tone}`;
}

function setStableHairBadge(text, tone = "neutral") {
  stableHairStatusBadge.textContent = text;
  stableHairStatusBadge.className = `badge ${tone}`;
}

function setResultSummary(title, detail, tone = "neutral") {
  resultTitle.textContent = title;
  resultDetail.textContent = detail;
  resultSummary.className = `result-summary ${tone}`;
}

function setStableHairSummary(title, detail, tone = "neutral") {
  stableHairTitle.textContent = title;
  stableHairDetail.textContent = detail;
  stableHairSummary.className = `result-summary ${tone}`;
}

function setHairReferenceSummary(title, detail, tone = "neutral") {
  hairReferenceTitle.textContent = title;
  hairReferenceDetail.textContent = detail;
  hairReferenceSummary.className = `result-summary compact-summary ${tone}`;
}

function updateStrictnessOutput() {
  strictnessValue.textContent = Number(strictnessRange.value).toFixed(2);
}

function getApiBaseUrl() {
  const validateUrl = new URL(validateEndpointInput.value.trim());
  return validateUrl.origin;
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
  setResultSummary("等待检测", "上传图片后，点击“开始检测”。", "neutral");
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

function resetStableHairInput() {
  referenceHairInput.value = "";
  state.referenceHairFile = null;
  state.referenceHairDataUrl = "";
  referenceHairPreview.src = "";
  referenceHairPreview.style.display = "none";
  referenceHairEmpty.hidden = false;
  stableHairResultPreview.src = "";
  stableHairResultPanel.hidden = true;
  showHairReferenceChecks([]);
  setHairReferenceSummary("等待检测", "上传发型参考图 B 后，点击“检测发型参考图”。", "neutral");
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

function showStableHairChecks(checks) {
  stableHairStatusList.innerHTML = checks
    .map((check) => `<li class="${check.status || "warn"}">${check.label}：${check.detail}</li>`)
    .join("");
}

function showHairReferenceChecks(checks) {
  if (!checks || checks.length === 0) {
    hairReferenceCheckList.innerHTML = "<li>后端未返回 B 图检测详情</li>";
    return;
  }

  hairReferenceCheckList.innerHTML = checks
    .map((check) => `<li class="${check.status || "warn"}">${check.label}：${check.detail}</li>`)
    .join("");
}

function showStableHairResponse(response) {
  stableHairResponseText.textContent = JSON.stringify(response, null, 2);
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

async function fetchStableHairStatus() {
  const response = await fetch(`${getApiBaseUrl()}/stable-hair/status`);

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json();
}

async function validateHairReferenceViaApi() {
  const payload = {
    hairReferenceBase64: getBase64Payload(state.referenceHairDataUrl),
    strictness: Number(strictnessRange.value),
  };

  const response = await fetch(hairReferenceEndpointInput.value.trim(), {
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

async function prepareStableHairViaApi() {
  const payload = {
    model: "stable-hair",
    portraitBase64: getBase64Payload(state.portraitDataUrl),
    hairReferenceBase64: getBase64Payload(state.referenceHairDataUrl),
    executeModel: executeStableHairToggle.checked,
    step: 30,
    guidanceScale: 1.5,
    controlnetConditioningScale: 1,
    hairEncoderScale: 1,
    size: 512,
    seed: -1,
  };

  const response = await fetch(generateEndpointInput.value.trim(), {
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

function applyStableHairStatus(status) {
  const checks = [
    {
      label: "外部仓库",
      status: status.repoExists ? "pass" : "fail",
      detail: status.repoExists ? status.repoDir : "缺少 external_models/Stable-Hair",
    },
    {
      label: "推理入口",
      status: status.inferenceScriptExists && status.runnerExists ? "pass" : "fail",
      detail: status.runnerExists ? "infer_full.py + backend runner" : "缺少 backend/stable_hair_infer.py",
    },
    {
      label: "权重文件",
      status: status.missingWeights && status.missingWeights.length > 0 ? "fail" : "pass",
      detail: status.missingWeights && status.missingWeights.length > 0
        ? `缺少 ${status.missingWeights.length} 个权重`
        : "stage1 / stage2 权重齐全",
    },
    {
      label: "Python 环境",
      status: status.pythonDependenciesOk ? "pass" : "fail",
      detail: status.pythonDependenciesOk
        ? `${status.pythonExecutable} / ${status.pythonDependenciesDetail}`
        : `不可用：${status.pythonExecutable}`,
    },
  ];

  showStableHairChecks(checks);
  showStableHairResponse(status);

  if (status.ready) {
    setStableHairBadge("可执行", "success");
    setStableHairSummary("Stable-Hair 已配置", "可以准备请求；如需真正执行，后端请求需启用 executeModel。", "success");
    return;
  }

  setStableHairBadge("待配置", "warning");
  setStableHairSummary("已接入，待配置", "当前可以保存输入和生成配置；还需要 Stable-Hair 权重和专用 Python 环境。", "warning");
}

function applyHairReferenceResult(result) {
  const warnCount = [
    ...(result.checks || []).filter((check) => check.status === "warn"),
    ...(result.suggestions || []),
  ].length;

  showHairReferenceChecks(result.checks || []);

  if (result.valid) {
    setHairReferenceSummary(
      warnCount > 0 ? "B 图可以试用" : "B 图检测通过",
      warnCount > 0
        ? `${warnCount} 个风险提示；当前未发现阻断问题。`
        : "参考图可定位到头部，基础画质通过。",
      warnCount > 0 ? "warning" : "success",
    );
    return;
  }

  setHairReferenceSummary(
    "B 图暂不建议使用",
    `${(result.issues || []).length || 1} 个阻断问题；建议更换更清晰完整的发型参考图。`,
    "danger",
  );
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
    result.valid ? "未发现阻断问题" : "后端未返回阻断问题列表",
  );

  if (result.valid) {
    const warnCount = (result.checks || []).filter((check) => check.status === "warn").length;
    setBadge("后端检测通过", "success");
    setResultSummary(
      warnCount > 0 ? "可以继续" : "检测通过",
      warnCount > 0 ? `${warnCount} 个风险提示；未发现阻断问题。` : "未发现阻断问题，可以进入后续发型迁移。",
      warnCount > 0 ? "warning" : "success",
    );
    statusText.textContent = "未发现阻断问题；警告项可作为拍摄质量建议";
    return;
  }

  setBadge("检测失败", "danger");
  setResultSummary(
    "暂不建议继续",
    `${(result.issues || []).length || 1} 个阻断问题；建议更换图片后重试。`,
    "danger",
  );
  statusText.textContent = "图片存在阻断问题，暂不建议进入发型迁移";
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
    setResultSummary("图片已载入", "点击“开始检测”调用后端 Face Detection。", "neutral");
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

async function handleReferenceHairChange(event) {
  const [file] = event.target.files;
  const fileIssues = validateFileBeforePreview(file);

  if (fileIssues.length > 0) {
    resetStableHairInput();
    showErrorModal(fileIssues);
    return;
  }

  try {
    const dataUrl = await fileToDataUrl(file);
    state.referenceHairFile = file;
    state.referenceHairDataUrl = dataUrl;
    referenceHairPreview.src = dataUrl;
    referenceHairPreview.style.display = "block";
    referenceHairEmpty.hidden = true;
    setStableHairSummary("参考图已载入", "可以点击“准备 Stable-Hair 请求”，后端会保存 A/B 两张输入图。", "neutral");
    setHairReferenceSummary("B 图已载入", "点击“检测发型参考图”检查头部定位、基础画质和裁切风险。", "neutral");
    showHairReferenceChecks([]);
  } catch (error) {
    resetStableHairInput();
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
  setResultSummary("检测中", "正在上传图片并等待后端返回检测结论。", "warning");
  statusText.textContent = "正在调用后端 validatePortrait";

  try {
    const strictness = Number(strictnessRange.value);
    const result = await validatePortraitViaApi(validateEndpointInput.value.trim(), strictness);
    applyValidationResult(result);
  } catch (error) {
    setBadge("接口异常", "danger");
    setResultSummary("后端连接失败", "请确认 FastAPI 在 127.0.0.1:8000 运行。", "danger");
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

async function handleValidateHairReference() {
  if (!state.referenceHairFile) {
    showErrorModal([
      {
        code: "NO_REFERENCE_IMAGE",
        message: frontendIssueCatalog.NO_REFERENCE_IMAGE,
      },
    ]);
    return;
  }

  setHairReferenceSummary("B 图检测中", "正在调用后端 validate-hair-reference。", "warning");

  try {
    const result = await validateHairReferenceViaApi();
    applyHairReferenceResult(result);
  } catch (error) {
    setHairReferenceSummary("B 图检测失败", "无法调用 validate-hair-reference；请确认后端服务运行中。", "danger");
    showHairReferenceChecks([
      {
        label: "后端连接",
        status: "fail",
        detail: "无法获取 validate-hair-reference 响应",
      },
    ]);
    console.error(error);
  }
}

async function handleCheckStableHair() {
  setStableHairBadge("检查中", "warning");
  setStableHairSummary("检查中", "正在读取后端 Stable-Hair 接入状态。", "warning");

  try {
    const status = await fetchStableHairStatus();
    applyStableHairStatus(status);
  } catch (error) {
    setStableHairBadge("接口异常", "danger");
    setStableHairSummary("检查失败", "无法调用 /stable-hair/status；请确认 FastAPI 后端运行中。", "danger");
    showStableHairChecks([
      {
        label: "后端连接",
        status: "fail",
        detail: "无法获取 Stable-Hair 状态",
      },
    ]);
    console.error(error);
  }
}

async function handlePrepareStableHair() {
  if (!state.portraitFile) {
    showErrorModal([{ code: "NO_IMAGE", message: frontendIssueCatalog.NO_IMAGE }]);
    return;
  }

  if (!state.referenceHairFile) {
    showErrorModal([
      {
        code: "NO_REFERENCE_IMAGE",
        message: frontendIssueCatalog.NO_REFERENCE_IMAGE,
      },
    ]);
    return;
  }

  stableHairResultPanel.hidden = true;
  stableHairResultPreview.src = "";
  setStableHairBadge(executeStableHairToggle.checked ? "执行中" : "准备中", "warning");
  setStableHairSummary(
    executeStableHairToggle.checked ? "模型执行中" : "准备请求中",
    executeStableHairToggle.checked
      ? "正在运行 Stable-Hair；512px 扩散模型可能需要数分钟。"
      : "正在上传用户图和参考发型图，后端会生成 Stable-Hair 配置。",
    "warning",
  );

  try {
    const result = await prepareStableHairViaApi();
    showStableHairResponse(result);
    applyStableHairStatus(result.stableHair);

    if (result.status === "NOT_CONFIGURED") {
      setStableHairBadge("请求已准备", "warning");
      setStableHairSummary(
        "请求已准备，模型待配置",
        `输入和 config 已保存；缺少 ${result.stableHair.missingWeights.length} 个权重，Python 依赖状态：${result.stableHair.pythonDependenciesOk ? "OK" : "未就绪"}。`,
        "warning",
      );
      return;
    }

    if (result.status === "COMPLETED" && result.resultImageDataUrl) {
      stableHairResultPreview.src = result.resultImageDataUrl;
      stableHairResultPreview.style.display = "block";
      stableHairResultPanel.hidden = false;
      setStableHairBadge("生成完成", "success");
      setStableHairSummary("Stable-Hair 生成完成", "后端已返回结果图。下面显示官方拼接结果。", "success");
      return;
    }

    if (result.status === "FAILED") {
      setStableHairBadge("执行失败", "danger");
      setStableHairSummary("Stable-Hair 执行失败", "查看“最近一次 Stable-Hair 响应”里的 stderr/stdout。", "danger");
      return;
    }

    setStableHairBadge("请求已准备", "success");
    setStableHairSummary("Stable-Hair 请求已准备", "后端已保存输入、生成 config，并返回了可执行命令。", "success");
  } catch (error) {
    setStableHairBadge("接口异常", "danger");
    setStableHairSummary("准备失败", frontendIssueCatalog.GENERATE_API_FAILED, "danger");
    console.error(error);
  }
}

portraitInput.addEventListener("change", handlePortraitChange);
referenceHairInput.addEventListener("change", handleReferenceHairChange);
validateButton.addEventListener("click", handleValidate);
validateHairReferenceButton.addEventListener("click", handleValidateHairReference);
resetButton.addEventListener("click", resetFlow);
checkStableHairButton.addEventListener("click", handleCheckStableHair);
prepareStableHairButton.addEventListener("click", handlePrepareStableHair);
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
  "前端只校验格式/大小；人脸检测走后端";

window.addEventListener("DOMContentLoaded", () => {
  updateStrictnessOutput();
  resetFlow();
  resetStableHairInput();
});
