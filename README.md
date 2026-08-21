## 目标

从一张真实生活场景图出发，通过 VLM 理解场景并识别适合学习的物品/动作/概念，逐层叠加母语、目标外语、发音信息，再通过 TTS 配音，自动生成一段情景词汇学习视频。

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
│
├── 04 Video Composer
      朗读音频
      输出MP4
```

## 资源

- 使用 .env 里面的 vlm 资源识别图片
- 使用 r'C:\Users\lawrence\PycharmProjects\luoci_log\z_using_file\tools\Kokoro' 来做英式 tts
- 使用 .env 里面的 llm 来翻译加音标

## 当前状态

只做`en`. 给出了示例输入:`input_pics/生活场景/carriage.png`,其中图片在`output/`下
