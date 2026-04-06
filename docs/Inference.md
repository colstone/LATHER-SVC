# 推理指南

## CLI 推理

### 命令

```bash
python scripts/infer.py \
  --config configs/config_quality.yaml \
  --exp_name my_experiment \
  --input source.wav \
  --output output.wav \
  --speaker speaker_A \
  --pitch_shift 0
```

### 完整参数说明

| 参数              | 必须  | 默认值  | 说明                                                                                 |
| --------------- | --- | ---- | ---------------------------------------------------------------------------------- |
| `--config`      | 是   | -    | 配置文件路径。可以是训练配置（如 `configs/config_quality.yaml`），也可以是 checkpoint 目录下的 `config.yaml` |
| `--exp_name`    | 否   | `''` | 实验名称。使用训练配置时需要提供，用于定位 checkpoint 目录                                                |
| `--input`       | 是   | -    | 输入音频文件路径（任意采样率，自动重采样）                                                              |
| `--output`      | 是   | -    | 输出音频文件路径                                                                           |
| `--speaker`     | 否   | None | 说话人名称或 ID。多说话人模型必须提供；单说话人模型可省略                                                     |
| `--pitch_shift` | 否   | 0.0  | 音高偏移（单位：半音）。正值升调，负值降调                                                              |
| `--ckpt`        | 否   | None | 指定 checkpoint 步数。不提供时使用最新 checkpoint                                               |
| `--device`      | 否   | None | 推理设备（如 `cuda`、`cpu`、`cuda:1`）。不提供时自动检测                                             |

### 示例

```bash
# 基本用法
python scripts/infer.py \
  --config configs/config_quality.yaml \
  --exp_name lather_quality \
  --input input.wav \
  --output output.wav \
  --speaker alice

# 升调 3 个半音
python scripts/infer.py \
  --config configs/config_quality.yaml \
  --exp_name lather_quality \
  --input input.wav \
  --output output_up3.wav \
  --speaker alice \
  --pitch_shift 3

# 使用指定 checkpoint
python scripts/infer.py \
  --config configs/config_quality.yaml \
  --exp_name lather_quality \
  --input input.wav \
  --output output.wav \
  --speaker alice \
  --ckpt 100000

# 速度优先模式
python scripts/infer.py \
  --config configs/config_fast.yaml \
  --exp_name lather_fast \
  --input input.wav \
  --output output_fast.wav \
  --speaker alice
```

---

## WebUI

### 启动

```bash
python webui.py \
  --quality_config ckpt/lather_quality/config.yaml \
  --quality_exp_name lather_quality \
  --port 7860
```

可以同时加载音质优先和速度优先两个模型：

```bash
python webui.py \
  --quality_config ckpt/lather_quality/config.yaml \
  --quality_exp_name lather_quality \
  --fast_config ckpt/lather_fast/config.yaml \
  --fast_exp_name lather_fast \
  --host 0.0.0.0 \
  --port 7860
```

### WebUI 参数

| 参数                   | 默认值         | 说明                      |
| -------------------- | ----------- | ----------------------- |
| `--quality_config`   | `''`        | 音质优先模式的配置文件路径           |
| `--quality_exp_name` | `''`        | 音质优先模式的实验名称             |
| `--fast_config`      | `''`        | 速度优先模式的配置文件路径           |
| `--fast_exp_name`    | `''`        | 速度优先模式的实验名称             |
| `--host`             | `127.0.0.1` | 监听地址。设为 `0.0.0.0` 可外部访问 |
| `--port`             | 7860        | 监听端口                    |

至少需要提供 `--quality_config` 或 `--fast_config` 之一。

### 使用方法

1. 打开浏览器访问 `http://127.0.0.1:7860`
2. 上传源音频文件
3. 选择 Mode（quality 或 fast）
4. 选择 Speaker（下拉列表会自动加载可用说话人）
5. 调整 Pitch Shift 滑块（-24 到 +24 半音）
6. 点击 "Run Inference"
7. 等待处理完成后试听和下载

---

## Pitch Shift 使用说明

Pitch shift 通过缩放 F0 实现音高偏移：

```
f0_shifted = f0 * 2^(pitch_shift / 12)
```

| pitch_shift | 效果           |
| ----------- | ------------ |
| 0           | 原调           |
| 12          | 升一个八度        |
| -12         | 降一个八度        |
| 3           | 升 3 个半音（小三度） |
| -5          | 降 5 个半音（纯四度） |

**注意事项**：

- 大幅度 pitch shift（超过 ±12 半音）可能导致音质下降
- 源音频的音域与目标说话人音域差距过大时，建议使用 pitch shift 做适当补偿
- 男声转女声通常需要 +4 到 +8 半音；女声转男声通常需要 -4 到 -8 半音

---

## 双模式推理的区别与选择

|      | 音质优先        | 速度优先        |
| ---- | ----------- | ----------- |
| 采样率  | 44.1kHz     | 24kHz       |
| 采样步数 | 20 步 Euler  | 4 步 Euler   |
| 有效带宽 | 全频段（~20kHz） | 受限（~12kHz）  |
| 推理速度 | 较慢          | 约 5 倍快（理论上） |
| 音质   | 更好，细节更丰富    | 略有损失，高频弱    |
| 适用场景 | 最终成品输出      | 实时预览、快速试听   |

**选择建议**：

- 制作最终作品 → 音质优先
- 快速试音/选人/调参 → 速度优先
- 显存紧张 → 速度优先（24kHz 模型显存占用更小）

---

## 推理效果不佳时的排查

### 效果归因表

| 问题现象         | 可能原因                        | 对应模块                   | 排查方向                                 |
| ------------ | --------------------------- | ---------------------- | ------------------------------------ |
| 内容（歌词/音素）不清晰 | ContentVec 提取质量不佳           | ContentVecExtractor    | 检查输入音频质量，确认 ContentVec 权重正确          |
| 音高不准/跑调      | F0 提取错误                     | RMVPE                  | 检查 RMVPE 权重，或输入音频中有大段无声/噪声           |
| 音色不像目标说话人    | Speaker embedding 不足        | LatherSVC (spk_embed)  | 增加训练数据或训练步数，确认说话人 ID 正确              |
| 唱法生硬/不自然     | Variance predictor 预测不准     | SVCVariancePredictor   | 检查训练时 var_loss 是否收敛，增加训练步数           |
| 整体模糊 / 缺少细节  | Rectified Flow 采样不足         | RectifiedFlow          | 增加 `sampling_steps`（如 20→30）         |
| 底噪/嘶嘶声       | Aux decoder 质量差或 vocoder 问题 | ConvNeXt / NSF-HiFiGAN | 检查 aux_mel_loss 收敛情况，确认 vocoder 权重匹配 |
| 高频缺失         | 速度优先模式的固有限制                 | Vocoder mel 插值         | 改用音质优先模式                             |
| 电音/金属质感      | 数据中混入伴奏或有混响                 | 数据质量                   | 清理训练数据，做更彻底的人声分离                     |

### 通用排查步骤

1. **确认 checkpoint 选择合理**：尝试不同步数的 checkpoint，有时最新的不是最好的
2. **检查输入音频**：确保是干声，无伴奏/混响/底噪
3. **检查说话人 ID**：确认 `--speaker` 参数与训练时一致
4. **尝试不同 pitch_shift**：如果源音频与目标说话人音域差距大，适当调整
5. **查看训练日志**：确认 mel_loss、aux_mel_loss、var_loss 都正常收敛
6. **对比 validation 输出**：TensorBoard 中的 diff_* 音频是否正常
