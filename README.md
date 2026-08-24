## 目标

从一张真实生活场景图出发，通过 VLM 理解场景并识别适合学习的物品/动作/概念，逐层叠加母语、目标外语、发音信息，再通过 Visual Cue 视觉动效引导（Ken Burns 局部缩放 + 聚光灯聚焦 + HUD 悬浮教学卡片）与 TTS 单词例句配音，自动生成一段沉浸式情境词汇学习视频。

```
Input (如 carriage.png)
│
├── 01 Scene Analyzer
│     VLM 理解图片 提取 8~12 个均匀分布的高价值词汇与坐标 (x, y)
│
├── 02 Language Generator
│     中文 → 地道英文 + 英式国际音标 (IPA) + 场景例句与翻译
│
├── 03 Visual Renderer
│     中文层、双语层、发音音标层、词汇复习表格
│
└── 04 Video Composer (Visual Cue 动效合成)
      片头 (3s 全景引导)
      → 视觉动效漫游 (Ken Burns 平滑推镜至物体 + 聚光灯高亮 + 悬浮大字卡片 + 单词发音与例句朗读)
      → 片尾 (4s 汇总打卡表格，引导长按截图保存)
      → 输出 1080x1920 高清短视频 MP4
```

## 快速复刻指南

### 1. 环境要求

- **Windows 10/11**(字体与路径默认按 Windows 配置;其他系统需自行改 `src/config.py` 中的字体路径)
- **Python 3.12+** 与 [uv](https://docs.astral.sh/uv/) 包管理器
- **ffmpeg**:需在 PATH 中(视频合成、混音依赖它),`winget install ffmpeg` 或从官网下载
- 一个 **OpenAI 兼容的 LLM/VLM API**(如 DeepInfra、OpenRouter、自建 vLLM 等)

### 2. 安装步骤

```powershell
# ① 克隆项目并安装依赖
git clone <本仓库地址>
cd scene_language
uv sync

# ② 配置 .env
copy .env.example .env
# 编辑 .env,填入你的 LLM / VLM 接口地址、模型名和 API Key(见下表)

# ③ 下载本地 TTS 模型 (Kokoro-82M ONNX, 约 400MB)
# 从 https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX 下载,
# 需要其中 onnx/model.onnx 和 voices/ 目录,
# 然后在 .env 里把 KOKORO_MODEL_DIR 指向该目录
```

### 3. `.env` 必填项

| 变量 | 说明 |
| --- | --- |
| `BASE_URL` | LLM 接口地址(OpenAI 兼容),用于翻译/音标/例句 |
| `MODEL` | LLM 模型名 |
| `API_KEY` | LLM 的 API Key |
| `VLM_BASE_URL` | VLM 接口地址(不填则复用 BASE_URL) |
| `VLM_MODEL` | 视觉模型名(需支持图片输入) |
| `VLM_API_KEY` | VLM 的 API Key |
| `KOKORO_MODEL_DIR` | 本地 Kokoro ONNX 模型目录(可选,不填用代码内默认路径) |

可选:`MAX_TOKENS`、`TIMEOUT`、`MAX_RETRIES`(请求参数)、`--voice/--speed/--zoom`(运行时参数)。

### 4. 目录说明

```
input_pics/          放入场景图片 (png/jpg/webp, 建议 9:16 竖图)
scene_catalog/       可选:预定义词表 <图片同名>.json,
                     格式 {"targets": [{"order": 1, "zh": "沙发", "en": "sofa"}, ...]}
                     有词表时 VLM 只负责定位坐标,词汇以词表为准;
                     没有则由 VLM 自由识别 8~12 个词
scripts/             维护词表的辅助脚本
src/music/           BGM (默认 booty.wav,可替换)
output/              全部产物:json/ 分层图/ 音频/ 视频/
```

### 5. 验证安装

```powershell
uv run python -m src.main --list-voices   # 能列出音色说明 TTS 就绪
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png
```

跑通后视频输出在 `output/videos/<图片名>.mp4`,中间产物在 `output/` 各子目录。

## 资源

- 使用 .env 里面的 vlm 资源识别图片
- 使用本地 Kokoro-82M ONNX 做英式/美式 TTS
- 使用 .env 里面的 llm 来翻译、生成音标与场景例句

## 当前状态

支持 `en`，默认英式男声 `bm_george`。示例输入：`input_pics/01_居家生活/H01_entrance.png`，图片与视频输出在 `output/` 下。


## 运行
```bash
# 对单个图片生成完整视频 (支持 Visual Cue 镜头缩放与聚焦)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png
```

### 其他用法
```bash
# 批量处理 input_pics/ 下的所有场景图片
uv run python -m src.main --all

# 只批量处理某个子目录 (input_pics 太大时)
uv run python -m src.main input_pics/02_饮食与购物 --all

# 强制重新请求 VLM/LLM (不使用中间缓存)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --no-cache

# 指定音色、语速与运镜放大倍率 (默认 zoom 1.7)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --voice bm_george --speed 1.1 --zoom 1.7

# 仅渲染分层图片 (不合成视频)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --no-video

# 也可以单独执行某个步骤 (1=VLM识别, 2=LLM翻译与音标, 3=图片渲染, 4=Visual Cue 视频合成)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --step 1 --no-cache
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --step 2
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --step 3
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --step 4 --no-cache

# 支持音色
uv run python -m src.main --list-voices
```

## 自动上传到各平台

使用 [social-auto-upload](https://github.com/dreammis/social-auto-upload)（已克隆安装到 `../social-auto-upload`），基于浏览器自动化，支持 抖音、小红书、视频号、Bilibili、YouTube 等平台。无需 API key，扫码登录一次即可。

### 登录

```powershell
cd C:\Users\lawrence\PycharmProjects\social-auto-upload
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& c:\Users\lawrence\PycharmProjects\social-auto-upload\.venv\Scripts\Activate.ps1)

uv run sau douyin login --account lawrence        # 抖音，扫码
uv run sau xiaohongshu login --account lawrence   # 小红书，扫码

uv run sau tencent login --account lawrence       # 视频号，扫码
uv run sau bilibili login --account lawrence      # B站
```

登录后可用 `uv run sau douyin check --account lawrence` 校验 cookie 是否有效，失效则重新 login。

### 上传视频

```powershell
uv run sau douyin upload-video --account lawrence --file ../scene_language/output/videos/H01_entrance.mp4 --title "第一集-居家生活-玄关场景" --desc "居家生活-玄关场景-场景词汇学习" --tags "每日英语,零基础英语,英语单词速记" --collection "场景英语词汇学习" --thumbnail-portrait ../scene_language/output/pronunciation/H01_entrance.png --declaration "内容为个人观点或见解"

uv run sau xiaohongshu upload-video --account lawrence --file ../scene_language/output/videos/H01_entrance.mp4 --title "第一集-居家生活-玄关场景" --desc "居家生活-玄关场景-场景词汇学习" --tags "每日英语,零基础英语,英语单词速记"

# 其他平台同理，把 douyin 换成 xiaohongshu / tencent / bilibili / youtube
# 支持 --schedule "2026-08-23 09:00" 定时发布（抖音/小红书/视频号/B站支持）
# --thumbnail-landscape 4:3 横版封面 --thumbnail-portrait 3:4 竖版封面
```
