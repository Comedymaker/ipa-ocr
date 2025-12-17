# ipa-ocr

## 数据集生成

使用下列命令从 `data/ipa_list.csv` 渲染完整的国际音标 OCR 训练集（默认两张增强样本/条目，图像保存在 `artifacts/full_dataset/images/`，标签写入 `labels.tsv`，同时输出生成参数到 `metadata.json`）：

```
python scripts/generate_ipa_dataset.py \
  --csv data/ipa_list.csv \
  --output-dir artifacts/full_dataset \
  --canvas-width 384 \
  --canvas-height 128 \
  --samples-per-text 2 \
  --metadata
```

> 提示：生成过程中图像数量较大（约 17 万张，单次运行可能持续数小时），建议先清空旧输出目录 `rm -rf artifacts/full_dataset`，并搭配 `nohup … &` 或 `tmux/screen` 等工具保持任务不中断。若需要分批生成，可通过 `--max-samples` 参数拆分为若干子任务。以上流程的后续步骤也将同步记录在此文档中，便于按顺序执行。

### 验证集抽样

若需要一个较小的验证集，可以从完整数据集中随机抽样若干条目（默认 2000 条），生成一个独立的 `labels.tsv` 与 `images/` 目录：

```
python scripts/create_val_subset.py \
  --source-root artifacts/full_dataset \
  --output-dir artifacts/val_subset \
  --count 2000
```

- 可通过 `--fraction 0.01` 等参数按比例采样，或调节 `--count` 控制样本数量；若需多次运行到同一目录，添加 `--force` 以覆盖。
- 输出目录结构与训练集一致，因此可直接在训练/评估脚本中将 `--dataset-root`/`--labels` 指向该验证集。

## OCR 模型设计与准备

1. **构建 IPA 字符集**

   ```
   python scripts/build_ipa_charset.py \
     --labels artifacts/full_dataset/labels.tsv \
     --output artifacts/full_dataset/charset.json \
     --freq artifacts/full_dataset/char_freq.tsv
   ```

   - 作用：扫描标签文件提取覆盖的全部字符，为后续模型定义输出词表与索引映射；`charset.json` 写入字符列表及数量，`char_freq.tsv` 提供频次参考以便处理罕见符号。

2. **模型方案（设计说明）**

   - **架构选择**：采用 CRNN（卷积特征提取 + 双向 LSTM 序列建模 + CTC 解码），兼顾图像特征学习与序列识别能力，适合无对齐标注的 OCR 任务。
   - **特征骨干**：推荐以轻量 ResNet 或 VGG Block 堆叠作为卷积前端，输出宽度方向的特征序列；根据算力可在 3–5 个卷积阶段之间调节通道数。
   - **序列建模**：使用两层双向 LSTM（每层隐藏单元 256–384）捕捉长距离依赖，输出序列长度随输入宽度而缩放。
   - **输出层与损失**：全连接映射到字符集大小 + 1（CTC blank），配合 `nn.CTCLoss` 训练；推理阶段结合束搜索或贪心解码，并可选用语言模型 / LLM 提示做后处理。
   - **训练要点**：按标签长度过滤或动态调整 batch size；将图像归一化到 `[0,1]` 或标准化到均值 0.5、方差 0.5；监控 CER/WER 与 CTC loss；定期保存模型权重与日志。
   - **后续工作**：实现数据加载器（读取 `labels.tsv`，按需裁剪/补齐图像）、训练脚本、评估脚本以及推理接口。完成后将在 README 中补充具体命令。

## 模型训练

执行以下命令训练 CRNN+CTC 模型（默认使用 GPU，如无可用 GPU 会自动回退到 CPU；训练日志会输出到终端并将最新、最佳权重分别保存为 `last.pt` 与 `best.pt`）：

```
python scripts/train_crnn.py \
  --dataset-root artifacts/full_dataset \
  --labels artifacts/full_dataset/labels.tsv \
  --charset artifacts/full_dataset/charset.json \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --output-dir artifacts/models/crnn \
  --device cuda:0
```

- 训练前请确保字符集文件已通过上一节命令生成。
- `--val-split` 默认划分 2% 样本用于验证，可按需调整；其余超参可根据显存与项目需求调整。
- 训练脚本会在每个 epoch 后评估验证集损失与 CER（字符错误率），并在验证集指标刷新时更新 `best.pt` 备份。

## 模型评估

使用最佳权重对数据集进行整体评估，并输出样例预测：

```
python scripts/evaluate_crnn.py \
  --dataset-root artifacts/val_subset \
  --labels artifacts/val_subset/labels.tsv \
  --charset artifacts/full_dataset/charset.json \
  --checkpoint artifacts/models/crnn/best.pt \
  --batch-size 64 \
  --device cuda:0 \
  --report-samples 5

```

- 若需在抽样验证集上评估，请将 `--dataset-root` 与 `--labels` 同时替换为 `artifacts/val_subset`；不要混合不同目录，否则标签内的相对路径会指向不存在的图像。
- 若只想快速抽样验证，可添加 `--max-samples 2000` 等参数限制样本量。
- 输出中会给出平均 CTC loss、CER，以及若干预测与参考文本对照，可用于分析错误类型。
- 若需在 CPU 上验证，将 `--device` 改为 `cpu` 即可。

## 端到端推理

对任意单张 IPA 图像执行推理：

```
python scripts/infer_crnn.py \
  --image artifacts/sample_dataset/images/00000_00.png \
  --charset artifacts/full_dataset/charset.json \
  --checkpoint artifacts/models/crnn/best.pt \
  --device cuda:0
```

- 图片会被缩放到训练时的分辨率（默认 384×128），输出预测的 IPA 字符串。若你的图片尺寸不同，可通过 `--image-width`/`--image-height` 调整预处理。
- 同样可改 `--device` 为 `cpu` 或其它 GPU 索引。
