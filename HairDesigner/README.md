# HairDesigner System Design

当前阶段先不做真正的发型生成，而是先把整套输入审核和交互端口设计好。

目标是把系统先拆成输入校验和模型选型两个明确部分：

1. `validatePortrait`
   负责判断用户上传的图片是否满足后续模型处理条件。
2. 模型输入要求对比
   先对齐 Barbershop、HairFastGAN、Stable-Hair 的输入条件，再决定后续接哪条推理链。

## 当前已完成的前端能力

- 上传用户图片 A
- POC 子面板展示
- `validatePortrait` 输入门控
- 检测失败时弹窗提示
- 检测结果面板
- 错误码列表展示
- `validatePortrait` API 端口输入框
- Barbershop 输入要求面板
- HairFastGAN 输入要求面板
- Stable-Hair 输入要求面板

入口文件：

- [index.html](/mnt/d/GitProject/BBusagi/HTML/HairDesigner/index.html)
- [styles.css](/mnt/d/GitProject/BBusagi/HTML/HairDesigner/styles.css)
- [app.js](/mnt/d/GitProject/BBusagi/HTML/HairDesigner/app.js)

## `validatePortrait` 需要覆盖的过滤规则

当前目标过滤规则：

- 是否单人
- 是否正脸/近正脸
- 脸是否足够大
- 是否有大面积遮挡
- 头顶是否被截断
- 光照是否过暗
- 分辨率是否太低
- 是否明显模糊

## 当前前端职责

当前 `app.js` 不做肖像、人脸、头发相关判断。

前端只做：

- 文件是否存在
- 文件 MIME 是否是 `image/*`
- 文件大小是否超过 POC 限制
- 本地图片预览
- 调用后端 `validate-portrait`
- 发送勾选的后端校验步骤
- 展示后端返回的 metrics / checks / issues

当前前端**不做人脸检测**。

也就是说，以下项目必须从后端 `validate-portrait` 返回：

- 是否单人
- 人脸数量
- 人脸框
- 脸部高度占比
- 人脸框距离上边缘的 headroom
- 正脸/近正脸
- 大面积遮挡

这样做是为了避免浏览器差异。比如 Chrome 也不一定提供 `window.FaceDetector`。

## 当前 POC 的后端校验开关

页面中有三个后端步骤选项：

- `Face Detection`
  当前可勾选，默认开启。目标是由后端确认是否有脸、是否单人、脸框、脸部大小、头顶边距等。
- `Landmark / Face Alignment`
  当前灰色禁用。Barbershop、HairFastGAN、Stable-Hair 这类迁移链路通常自带或要求自己的 alignment 预处理，POC 先不重复做。
- `Hair Segmentation / Face Parsing`
  当前灰色禁用。迁移模型链路通常会包含 parsing / mask / bald converter 等阶段，POC 先不独立启用。

## 推荐的成熟检测方案

后续不建议继续靠手写 canvas 逻辑判断所有条件。建议后端逐步接：

- OpenCV Haar / DNN face detector：做人脸数量和人脸框初筛
- MediaPipe Face Detection / Face Mesh：做人脸框、关键点、姿态和遮挡 proxy
- RetinaFace / SCRFD：更稳定的人脸检测和五点关键点
- Laplacian variance：服务端模糊检测
- 简单 luminance histogram：过暗、过曝检测
- 规则化裁切检查：face box / landmarks 和图像边界距离

## 本地后端占位

已经新增一个最小本地后端：

- [backend/server.py](/mnt/d/GitProject/BBusagi/HTML/HairDesigner/backend/server.py)

启动方式：

```bash
cd HTML/HairDesigner
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

也可以：

```bash
bash scripts/run_backend.sh
```

前端使用方式：

1. 启动上述 server
2. 打开 POC 页面
3. 点击“开始检测”

当前 server 是后端占位，只做：

- 分辨率
- 亮度
- 模糊 proxy
- MediaPipe Face Detection：安装 `mediapipe` 后启用

它保留了 `checks` 字段，后面可以继续接 MediaPipe / RetinaFace / OpenAI。

如果选择 Stable-Hair 这类“源人像 + 发型参考图”的方案，还需要再增加 `validateHairReference`：

- 判断参考图里是否有清晰头发区域
- 判断参考图是否能提供目标发型结构
- 判断参考图是否能提供目标发色/纹理
- 判断参考发型是否被帽子、手、饰品大面积遮挡

## 前端工作流

### Step 1. 用户上传图片

前端读取本地文件，显示预览，并清空旧检测结果。

### Step 2. 调用 `validatePortrait`

前端发送用户图片和校验严格度。

如果返回失败：

- 禁止进入生成阶段
- 在检测面板展示错误列表
- 弹出错误提示框

如果返回通过：

- 解锁生成按钮
- 保留检测指标
- 进入下一阶段

### Step 3. 选择迁移模型

当前 POC 页面展示 3 个候选发型迁移模型输入条件：

- Barbershop
- HairFastGAN
- Stable-Hair

### Step 4. 调用 `generateHairstylePreview`

前端发送：

- 用户肖像
- 发型描述
- 负向限制词
- 人物保持强度
- 发型编辑强度

后续接 OpenAI 时，建议你在自己的后端中转层里完成：

- prompt 组装
- 图片预处理
- 人脸区域约束
- 调用图像编辑模型
- 返回最终结果图

## 接口设计

### 1. 肖像检测接口

默认地址：

- `POST http://127.0.0.1:8000/validate-portrait`

请求：

```json
{
  "portraitBase64": "...",
  "strictness": 0.7,
  "validationOptions": {
    "faceDetection": true,
    "landmarkAlignment": false,
    "hairSegmentation": false
  }
}
```

推荐返回：

```json
{
  "valid": false,
  "faceCount": 2,
  "faceSizeScore": 0.2,
  "frontalScore": 0.35,
  "clarityScore": 0.41,
  "brightnessScore": 0.62,
  "headroomScore": 0.1,
  "occlusionScore": 0.72,
  "checks": [
    {
      "label": "是否单人",
      "status": "fail",
      "detail": "2 张脸"
    }
  ],
  "issues": [
    {
      "code": "MULTIPLE_PEOPLE",
      "message": "检测到多人，当前仅支持单人肖像"
    }
  ]
}
```

### 2. 发型生成接口

默认地址：

- `POST http://127.0.0.1:8000/generate-hairstyle`

请求：

```json
{
  "portraitBase64": "...",
  "hairstylePrompt": "保留人物五官与脸型，改成法式波波头，深栗色，空气刘海，写实照片风格",
  "negativePrompt": "不要改变年龄感，不要改变表情，不要改变背景，不要出现帽子和发饰",
  "identityStrength": 0.88,
  "editStrength": 0.52,
  "composedPrompt": "仅修改人物发型，保持人物身份、五官、脸型、年龄感、背景和镜头语言尽量不变..."
}
```

返回：

```json
{
  "imageBase64": "..."
}
```

或：

```json
{
  "imageUrl": "http://..."
}
```

## 推荐错误码

前端已经为以下错误码准备好了交互入口：

- `NO_IMAGE`
- `LOW_RESOLUTION`
- `BLURRY_IMAGE`
- `BAD_FRAMING`
- `FACE_TOO_SMALL`
- `MULTIPLE_PEOPLE`
- `FACE_OCCLUDED`
- `NO_FACE_DETECTED`
- `PROFILE_TOO_STRONG`
- `HEAD_TOP_CUT`
- `TOO_DARK`

建议后端统一输出这些 code，前端无需再改结构。

## 外部模型代码库

第三方模型代码不要直接提交进本项目。

先运行：

```bash
bash scripts/clone_model_repos.sh
```

脚本会把浅克隆放到：

```text
external_models/
```

这个目录已被 git 忽略。

当前保留了后端适配占位：

- [backend/model_adapters.py](/mnt/d/GitProject/BBusagi/HTML/HairDesigner/backend/model_adapters.py)
- [backend/README.md](/mnt/d/GitProject/BBusagi/HTML/HairDesigner/backend/README.md)

## 后续接 OpenAI 的建议方式

前端不要直接持有真正的 OpenAI 密钥。

建议你新增一个自己的后端服务：

1. 前端把图片和参数发给你的后端
2. 后端做图片审核和结构化判定
3. 后端再调用 OpenAI 或其他图像模型
4. 后端把标准化结果返回给前端

这样可以保证：

- API Key 不暴露
- 错误码统一
- 模型切换时前端无需重构

## 当前适合的下一步

最合理的下一个阶段不是继续堆前端，而是补一个最小后端：

1. `validate-portrait`
2. `generate-hairstyle`

只要这两个端点有了，当前前端就能直接接上。
