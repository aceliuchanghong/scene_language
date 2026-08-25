# 日语 (ja) 与韩语 (ko) 场景词汇视频生成指南

本项目已全面支持**日语（`ja`）**与**韩语（`ko`）**情境词汇视频生成。

基于 **LLM 多语言地道释义与注音生成** + **Windows 原生高清 CJK 字体排版** + **OmniVoice 本地多语言 Zero-Shot TTS 引擎**，一键输出沉浸式日语/韩语场景词汇学习短视频。

---

## 1. 语言与注音规范

| 语言代码 (`--lang`) | 目标词汇（Target） | 发音标注（Pron / 注音） | 例句与中文对照 | 字体支持 |
| :--- | :--- | :--- | :--- | :--- |
| **`en` (英语, 默认)** | 英文单词 (如 `pram`) | 国际音标 IPA (如 `/ˈpræm/`) | 英文例句 + 中文翻译 | Segoe UI (`segoeui.ttf`) |
| **`ja` (日语)** | 日文汉字/假名 (如 `玄関`) | 平假名读音 + 罗马音 (如 `げんかん / genkan`) | 日文口语例句 + 中文翻译 | 游黑体 Yu Gothic (`YuGothM.ttc`) |
| **`ko` (韩语)** | 韩文谚文 (如 `현관`) | 标准罗马拼音 (如 `hyeon-gwan`) | 韩文口语例句 + 中文翻译 | Malgun Gothic (`malgun.ttf`) |

---

## 2. 环境与前置依赖

1. **Windows 字体**：系统自带（Windows 10/11 已内置）。
   - 日文：`C:\Windows\Fonts\YuGothM.ttc`（游黑体）
   - 韩文：`C:\Windows\Fonts\malgun.ttf`（Malgun Gothic，完整支持谚文排版）
2. **TTS 引擎**：
   - 英语：使用本地极速 Kokoro-82M ONNX。
   - 日语 & 韩语：使用本地 **OmniVoice** 多语言 Diffusion TTS。
   - OmniVoice 默认路径：`C:\Users\lawrence\PycharmProjects\luoci_log\z_using_file\tools\OmniVoice`（可通过环境变量 `OMNIVOICE_DIR` 自定义）。

---

## 3. 快速上手命令

### 3.1 单张图片生成

系统已默认内置日语【温柔女声】和韩语【稳重男声】的 `.pt` 音色凭据，直接运行即可：

```powershell
# 1. 生成日语场景词汇视频 (默认加载 ja_gentle_female.pt 温柔女声)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --lang ja

# 2. 生成韩语场景词汇视频 (默认加载 ko_calm_male.pt 稳重男声)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --lang ko

# 3. 生成英语场景词汇视频 (默认)
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png
```

### 3.2 批量生成

```powershell
# 批量为 input_pics/ 下所有场景生成日语版视频
uv run python -m src.main --all --lang ja

# 批量为 02_饮食与购物 子目录下所有场景生成韩语版视频
uv run python -m src.main input_pics/02_饮食与购物 --all --lang ko
```

---

## 4. OmniVoice 专属音色 Prompt (.pt) 与音色调优

为避免每次依靠随机种子浮动，系统支持提取并固定专属音色 Prompt（`.pt` 文件），实现跨语言 100% 绝对一致的声音克隆复刻。

### 4.1 系统内置专属音色 (`src/voices/`)

| 语言 | 音色定位 | 内置 Prompt 文件 | 状态 | 参考样本音频 |
| :--- | :--- | :--- | :--- | :--- |
| **日语 (`ja`)** | **自然清晰女声 2** | `src/voices/ja_f2.pt` | **默认启用** | `src/voices/ja_f2.wav` |
| **日语 (`ja`)** | **知性自然女声 1** | `src/voices/ja_f.pt` | 备选 | `src/voices/ja_f.wav` |
| **韩语 (`ko`)** | **清晰自然女声** | `src/voices/ko_f.pt` | **默认启用** | `src/voices/ko_f.wav` |

> **自动加载机制**：当未指定 `--voice` 属性时，系统自动优先加载对应语言的默认 `.pt` 文件（日语默认加载 `ja_f2.pt`，韩语默认加载 `ko_f.pt`），无需每次手动传参或设置 `--seed`！


### 4.2 加载自定义 `.pt` 音色

若你有其他提取好的 `.pt` 声音凭证，可通过 `--voice-pt` 或直接在 `--voice` 中传入文件路径：

```powershell
# 显式指定自定义 .pt 音色文件
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --lang ja --voice-pt path/to/my_custom_voice.pt

# 或直接作为 --voice 路径传入
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --lang ko --voice path/to/my_ko_voice.pt
```

### 4.3 临时切换音色属性描述（Voice Design）

若不想使用 `.pt` 而是临时探索其他声线，传入文字描述即可覆盖：

```powershell
# 1. 临时切换为日语少女音
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --lang ja --voice "female, teenager, high pitch" --seed 123

# 2. 临时切换为韩语活泼青年女声
uv run python -m src.main input_pics/01_居家生活/H01_entrance.png --lang ko --voice "female, young adult, high pitch" --seed 42
```

---

## 5. 跨语言缓存与 scene_catalog 疑问解答

### 问：日语和韩语需要提前准备 `scene_catalog` 的 JSON 吗？
**答：完全不需要！**

1. **底层逻辑**：
   - `scene_catalog` 是场景视觉与画面的策划表，其本质是定义画面中的**空间物体概念**（如“前门”、“钥匙”、“鞋架”以及生图提示词）。
   - **画面物体与其空间坐标是语言无关的**：无论英语、日语还是韩语，画面中的“前门”都在同一个坐标 `(x, y)`。
2. **自动化流程**：
   - **已有英文版本**：若该场景已生成过英文版（已有 `output/json/<stem>.json`），`step1` 会直接从英文版 JSON 提取坐标与中文词，**0 视觉成本毫秒级复用**。
   - **从零直接跑日/韩**：`step1` 会直接读取 `scene_catalog` 中的中文 `targets[].zh` 词表给 VLM 去画面中识别坐标。
   - **翻译由 LLM 实时完成**：`step2` 会根据日韩专属提示词（`PROMPT_JA` / `PROMPT_KO`），将这些中文概念转换为最地道的目标外语词汇、注音及场景例句。

因此，**同一份中文 `scene_catalog` 可以通用于所有外语**，无需为每种语言繁琐地维护多套 catalog。

---

## 6. 单步调试执行

```powershell
# 单独执行 Step 1 (提取坐标):
uv run python -m src.step1.cli --image input_pics/01_居家生活/H01_entrance.png --lang ja

# 单独执行 Step 2 (LLM 生成日文/韩文词汇与注音):
uv run python -m src.step2.cli --image input_pics/01_居家生活/H01_entrance.png --lang ja

# 单独执行 Step 3 (渲染 3 张分层标注图):
uv run python -m src.step3.cli --image input_pics/01_居家生活/H01_entrance.png --lang ja

# 单独执行 Step 4 (合成 MP4 短视频):
uv run python -m src.step4.cli --image input_pics/01_居家生活/H01_entrance.png --lang ja
```

---

## 7. 产物输出目录结构

不同语言的产物自动隔离，互不覆盖：

```
output/
├── json/
│   ├── H01_entrance.json        # 英语 JSON
│   ├── H01_entrance_ja.json     # 日语 JSON
│   └── H01_entrance_ko.json     # 韩语 JSON
├── source_language/
│   └── H01_entrance.png         # 中文层 (各语言共用)
├── target_language/
│   ├── H01_entrance.png         # 中英双语对照图
│   ├── H01_entrance_ja.png      # 中日双语对照图
│   └── H01_entrance_ko.png      # 中韩双语对照图
├── pronunciation/
│   ├── H01_entrance.png         # 英语发音音标图
│   ├── H01_entrance_ja.png      # 日语发音注音图
│   └── H01_entrance_ko.png      # 韩语发音注音图
└── videos/
    ├── H01_entrance.mp4         # 英语 1080x1920 高清视频
    ├── H01_entrance_ja.mp4      # 日语 1080x1920 高清视频
    └── H01_entrance_ko.mp4      # 韩语 1080x1920 高清视频
```