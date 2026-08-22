## 目标

从一张真实生活场景图出发，通过 VLM 理解场景并识别适合学习的物品/动作/概念，逐层叠加母语、目标外语、发音信息、表格，再通过 TTS 配音，自动生成一段情景词汇学习视频。

```
Input
│
├── 01 Scene Analyzer
│     VLM 理解图片 选择值得学习的词
│
├── 02 Language Generator
│     中文 → 目标语言
│     发音 / 音标
│
├── 03 Visual Renderer
│     中文版
│     双语版
│     外语+音标版
│     表格
│
├── 04 Video Composer
      朗读单词
      输出MP4
```

## 资源

- 使用 .env 里面的 vlm 资源识别图片
- 使用 r'C:\Users\lawrence\PycharmProjects\luoci_log\z_using_file\tools\Kokoro' 来做英式 tts
- 使用 .env 里面的 llm 来翻译加音标

## 当前状态

只做`en`. 给出了示例输入:`input_pics/生活场景/carriage.png`, 其中图片与视频输出在`output/`下。

## 安装
```powershell
uv sync
```

## 运行
```powershell
# 对单个图片生成
uv run python -m src.main --image input_pics/生活场景/carriage.png

# 批量处理 input_pics/ 下的所有场景图片
uv run python -m src.main --all

# 强制重新请求 VLM/LLM (不使用中间缓存)
uv run python -m src.main --image input_pics/生活场景/carriage.png --no-cache

# 指定音色 (默认英式男声 bm_george，可选 bf_emma, bm_daniel 等) 与语速
uv run python -m src.main --image input_pics/生活场景/carriage.png --voice bf_emma --speed 1.2

# 仅渲染分层图片 (不合成视频)
uv run python -m src.main --image input_pics/生活场景/carriage.png --no-video

# 单独执行某个步骤 (1=VLM识别, 2=LLM翻译与音标, 3=图片渲染, 4=视频合成)
uv run python -m src.main --image input_pics/生活场景/carriage.png --step 1 --no-cache
uv run python -m src.main --image input_pics/生活场景/carriage.png --step 2
uv run python -m src.main --image input_pics/生活场景/carriage.png --step 3
uv run python -m src.main --image input_pics/生活场景/carriage.png --step 4 --no-cache

# 支持音色
uv run python -m src.main --list-voices
```

## 自动上传到各平台

使用 [social-auto-upload](https://github.com/dreammis/social-auto-upload)（已克隆安装到 `C:\Users\lawrence\PycharmProjects\social-auto-upload`），基于浏览器自动化，支持 抖音、小红书、视频号、Bilibili、YouTube 等平台。无需 API key，扫码登录一次即可。

### 登录（每个平台只需一次，cookie 会保存）

```powershell
cd C:\Users\lawrence\PycharmProjects\social-auto-upload

uv run sau douyin login --account lawrence        # 抖音，扫码
uv run sau xiaohongshu login --account lawrence   # 小红书，扫码
uv run sau tencent login --account lawrence       # 视频号，扫码
uv run sau bilibili login --account lawrence      # B站
```

登录后可用 `uv run sau douyin check --account lawrence` 校验 cookie 是否有效，失效则重新 login。

### 上传视频

```powershell
uv run sau douyin upload-video --account lawrence `
  --file output/carriage.mp4 `
  --title "马车 carriage" --desc "场景词汇学习" --tags "英语,词汇"

# 其他平台同理，把 douyin 换成 xiaohongshu / tencent / bilibili / youtube
# 支持 --schedule "2026-08-23 09:00" 定时发布（抖音/小红书/视频号/B站支持）
# 支持 --headless 后台运行 / --debug 调试
```
