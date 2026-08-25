# 场景目录与课程说明

`scene_catalog/` 是第一期场景词汇课程的机器可读任务目录。课程共包含 **82 张竖版场景图、每图 10 个目标词、820 个学习位**；按英文表达去重后为 813 个词条。

| 批次 | 模块 | 图片数 | 学习位 |
|---|---|---:|---:|
| A | 居家生活、饮食与购物、城市与交通 | 32 | 320 |
| B | 学习与工作、医疗与公共服务、社交与休闲 | 20 | 200 |
| C | 旅行与户外、数字生活 | 12 | 120 |
| D | 文化与艺术、运动与竞技、个人护理与时尚 | 18 | 180 |

## 内容原则

- 每张图固定 10 个目标词；每个词必须对应一个清晰、可见、可定位的实体或动作。
- 使用常见英式英语，例如 `pavement`、`underground`、`trolley`、`lift` 和 `petrol`。
- 十个目标应自然地存在于同一个真实场景中，不能为了凑词做成物品陈列柜。
- 图片采用 9:16，目标物在上、中、下区域大致按 3、4、3 分布，同时兼顾左右区域。
- 源图不生成标签、字幕、水印、Logo 或可读文字；文字由后续渲染流程叠加。
- 含准确界面文字的场景（W05、D02、D03、D05、D06）优先使用真实截图、HTML 模拟界面或后期合成，不依赖图片模型生成 UI 文字。

## 目录结构

```text
scene_catalog/
├── index.json                  # 总索引、批次和统计
├── 01_居家生活/
│   ├── _category.json          # 分类索引
│   ├── H01_entrance.json       # 一个场景一个任务文件
│   └── ...
├── 08_数字生活/
│   └── ...
└── 11_个人护理与时尚/
    └── ...
```

批处理程序的读取顺序是：

```text
index.json → 分类 _category.json → 单场景 JSON → generation.prompt
```

每个场景 JSON 包含：

- `targets`：固定的 10 个中英文目标词；
- `generation.prompt`：可以直接提交给图片模型的英文提示词；
- `generation.asset_strategy`：素材策略；
- `image.path`、`width`、`height`：图片保存位置和目标尺寸；
- `requirements`、`qa`：验收要求和结果；
- `workflow`：当前流水线状态；
- `content_signature`：策划内容签名。

## 数据源与维护

各分类目录中的单场景 JSON 是唯一数据源。修改场景名称或 `targets` 后运行：

```powershell
uv run python scripts/maintain_scene_catalog.py
```

维护脚本会：

- 校验 82 个场景、每场景 10 个词、共 820 个学习位和 813 个独立英文词条；
- 根据 `scene` 和 `targets` 自动重建 `generation.prompt`；
- 重新计算 `content_signature`；
- 重建每个 `_category.json` 和根目录 `index.json`；
- 保留场景 JSON 中的 `workflow` 和 `qa` 状态。

只检查、不写入文件时使用：

```powershell
uv run python scripts/maintain_scene_catalog.py --check
```

## 生成图片

目前仓库只负责准备生图任务，**尚未提供自动调用图片模型的批量生图脚本**。生成单张图的操作如下：

1. 选择一个场景 JSON，例如 `01_居家生活/H01_entrance.json`。
2. 查看 `generation.asset_strategy`。值为 `ai_image` 时可直接生图；`programmatic_ui_*` 应改用界面截图或合成；`existing_or_ai_image` 可先检查旧素材。
3. 把 `generation.prompt` 完整提交给支持图片生成的模型。
4. 将结果裁切或导出为 JSON 中要求的 1080×1920 PNG，并保存到 `image.path`。
5. 按下方清单验收；不通过时局部修图或重新生成。

在 PowerShell 中可用下面的命令读取提示词和目标路径：

```powershell
$scene = Get-Content -Raw -Encoding UTF8 `
  "scene_catalog/01_居家生活/H01_entrance.json" | ConvertFrom-Json

$scene.generation.prompt       # 复制到图片模型
$scene.image.path              # 生成结果应保存到这里
```

H01 的目标保存位置是：

```text
input_pics/01_居家生活/H01_entrance.png
```

生成的 PNG 才是现有视频流水线的输入。例如：

```powershell
uv run python -m src.main `
  --image "input_pics/01_居家生活/H01_entrance.png"
```

T03 的 `source.existing_asset` 保留了旧素材路径；如果该文件已不存在，就按 `generation.prompt` 重新生成。

## 可以直接修改某个 JSON 里的单词吗？

可以，而且现在这就是正式的修改方式。

1. 打开对应的单场景 JSON。
2. 修改 `targets` 中对应项的 `zh` 和/或 `en`，不要手工修改 `generation.prompt` 或 `content_signature`。
3. 运行 `uv run python scripts/maintain_scene_catalog.py`。
4. 检查自动更新后的提示词，再生成图片。

维护脚本不会替你翻译。因此，如果英文词和中文词都发生变化，需要同时修改该目标的 `zh` 与 `en`。脚本还会校验课程的整体数量和独立英文词条数；如果替换造成意外重复，会直接报错，避免静默改变课程规模。

## 单图验收

图片只有同时满足以下条件才进入视频流水线：

- 十个指定目标全部存在，没有被相似物替代；
- 每个词只有一个明确的主要指认对象；
- 手机屏幕上仍能辨认，且没有被人物、家具或画面边缘遮挡；
- 目标覆盖上、中、下和左右区域，空间关系仍然自然；
- 没有明显畸形、错误数量、乱码、品牌水印或多余标签；
- 英文词与画面严格对应。

推荐状态流：

```text
planned → generated → qa_passed → localized → rendered → published
```

整体流水线是：

```text
固定十词 → 生成场景图 → VLM 逐项检查并定位
         → LLM 只补充 IPA 和例句 → 图片标注、TTS、视频合成
```
