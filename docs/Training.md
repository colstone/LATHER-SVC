# 训练指南

## 数据准备

### 音频要求

| 项目   | 要求                    |
| ---- | --------------------- |
| 格式   | WAV（单声道）              |
| 采样率  | 任意（预处理时自动重采样）         |
| 内容   | 干声（建议提前做人声分离，去除伴奏和混响） |
| 切片长度 | 建议 5-15 秒             |
| 数据量  | 每个说话人建议 30 分钟以上（更多更好） |

### 数据目录结构

每个说话人一个文件夹，内部放置 wav 文件。支持子目录递归扫描。

```
data/
├── speaker_A/
│   ├── song1_seg001.wav
│   ├── song1_seg002.wav
│   └── ...
└── speaker_B/
    ├── track1_001.wav
    └── ...
```

### 配置数据集

在 `configs/svc_base.yaml` 的 `datasets` 段中注册每个说话人：

```yaml
datasets:
  - raw_data_dir: /path/to/speaker_A
    speaker: speaker_A
    spk_id: 0
    language: zh
    artifact_level: 0
    test_prefixes:
      - song1_seg003

  - raw_data_dir: /path/to/speaker_B
    speaker: speaker_B
    spk_id: 1
    language: en
    artifact_level: 0
    test_prefixes:
      - track1_005
```

各字段说明：

- `raw_data_dir`：wav 文件所在目录的绝对路径
- `speaker`：说话人名称（任意字符串，用于标识）
- `spk_id`：说话人 ID，从 0 开始递增。全局 `num_spk` 必须大于最大 `spk_id`
- `language`：语言标记，目前仅作为元数据记录
- `artifact_level`：0 = 高质量干声，1 = 有瑕疵/拼接痕迹的数据
- `test_prefixes`：用于验证集的文件名前缀。匹配到的文件会被划入验证集，其余用于训练

同时修改 `configs/config_quality.yaml`（或 `config_fast.yaml`）中的 `binary_data_dir`：

```yaml
binary_data_dir: data/my_project/binary
```

---

## 预处理

### 命令

```bash
python scripts/binarize.py --config configs/config_quality.yaml
```

速度优先模式需要单独预处理（因为采样率和 hop size 不同）：

```bash
python scripts/binarize.py --config configs/config_fast.yaml
```

### 预处理产物

预处理完成后，`binary_data_dir` 下会生成：

```
data/my_project/binary/
├── train.data          # 训练集二进制数据
├── train.idx           # 训练集索引
├── train.meta          # 训练集元信息（lengths, spk_ids 等）
├── valid.data          # 验证集二进制数据
├── valid.idx           # 验证集索引
├── valid.meta          # 验证集元信息
├── spk_map.json        # 说话人名称 → ID 映射
└── lang_map.json       # 语言映射（SVC 模式下为空）
```

### 预处理做了什么

对每个 wav 文件依次执行：

1. **Mel 提取**：使用 STFT + mel filterbank 提取 mel spectrogram
2. **F0 提取**：使用 RMVPE 提取基频和 unvoiced flag
3. **谐波-噪声分离**：使用 WORLD/VR 算法将波形分解为谐波和噪声部分
4. **Variance 参数提取**：
   - breathiness：噪声部分的帧能量（dB）
   - voicing：谐波部分的帧能量（dB）
   - tension：高频谐波与基频谐波的能量比（logit）
   - 三个参数都做正弦窗平滑（窗宽 0.12 秒）
5. **ContentVec 特征缓存**：加载 ContentVec-768，提取第 7-12 层 hidden states 并保存
6. **打包**：将所有特征写入 IndexedDataset 二进制格式

> 注意：ContentVec 特征在预处理阶段就已缓存，训练时不需要加载 ContentVec 模型，节省显存。

---

## 训练

### 命令

```bash
python scripts/train.py --config configs/config_quality.yaml --exp_name my_experiment
```

`--exp_name` 指定实验名称，checkpoint 和日志会保存到 `checkpoints/my_experiment/`。

### 关键训练参数

| 参数                     | 默认值        | 说明                 |
| ---------------------- | ---------- | ------------------ |
| `max_updates`          | 160000     | 总训练步数              |
| `max_batch_frames`     | 48000      | 每 batch 最大帧数（控制显存） |
| `max_batch_size`       | 48         | 每 batch 最大样本数      |
| `val_check_interval`   | 2000       | 验证间隔步数             |
| `pl_trainer_precision` | `16-mixed` | 混合精度训练             |
| `pl_trainer_devices`   | `auto`     | 使用的 GPU            |

显存不足时，降低 `max_batch_frames` 和 `max_batch_size`。

### 监控训练

**启动 TensorBoard**：

```bash
tensorboard --logdir checkpoints/my_experiment/lightning_logs
```

**关键监控指标**：

| 指标             | 含义                                    | 期望趋势          |
| -------------- | ------------------------------------- | ------------- |
| `mel_loss`     | Rectified Flow velocity matching loss | 持续下降，最终收敛     |
| `aux_mel_loss` | ConvNeXt aux decoder 的 mel L1 loss    | 前期快速下降，后期缓慢收敛 |
| `var_loss`     | Variance predictor MSE loss           | 持续下降          |

**Validation 输出**：

- `gt_*`：Ground truth 音频（vocoder 合成）
- `aux_*`：Aux decoder 粗略 mel 的合成音频
- `diff_*`：Rectified Flow 精修后的合成音频
- `auxmel_*` / `diffmel_*`：mel spectrogram 对比图（差异 + GT + 预测）

对比 `diff_*` 和 `gt_*` 音频可以直接评估当前训练效果。`aux_*` 一般比 `diff_*` 模糊，这是正常的。

### 训练时长参考

以单张 RTX 3090 (24GB) 为例：

| 数据量           | 建议步数            | 大约时长     |
| ------------- | --------------- | -------- |
| 1 说话人，30 分钟   | 80,000-100,000  | 6-10 小时  |
| 5 说话人，各 30 分钟 | 120,000-160,000 | 12-20 小时 |
| 10+ 说话人       | 160,000+        | 20+ 小时   |

实际时长取决于 batch size、数据切片长度和 GPU 性能。

---

## 双模式训练

音质优先和速度优先是两个独立的训练流程：

|                 | 音质优先                          | 速度优先                       |
| --------------- | ----------------------------- | -------------------------- |
| 配置文件            | `configs/config_quality.yaml` | `configs/config_fast.yaml` |
| 采样率             | 44.1kHz                       | 24kHz                      |
| Hop size        | 512                           | 300                        |
| 采样步数            | 20                            | 4                          |
| 预处理             | 需要单独预处理                       | 需要单独预处理                    |
| 训练              | 独立训练                          | 独立训练                       |
| binary_data_dir | 必须不同                          | 必须不同                       |

两个模式共享代码和配置继承链，但生成的二进制数据和训练权重完全独立。

```bash
# 音质优先
python scripts/binarize.py --config configs/config_quality.yaml
python scripts/train.py --config configs/config_quality.yaml --exp_name quality_exp

# 速度优先
python scripts/binarize.py --config configs/config_fast.yaml
python scripts/train.py --config configs/config_fast.yaml --exp_name fast_exp
```

---

## Checkpoint 管理

### 保留策略

| 参数                        | 默认值   | 说明                         |
| ------------------------- | ----- | -------------------------- |
| `num_ckpt_keep`           | 5     | 保留最近 N 个 checkpoint，旧的自动删除 |
| `permanent_ckpt_start`    | 60000 | 从第 60000 步开始标记为永久保留        |
| `permanent_ckpt_interval` | 10000 | 每 10000 步保存一个永久 checkpoint |

永久 checkpoint 不受 `num_ckpt_keep` 限制，不会被自动删除。适合用于后续选取最佳模型。

### Checkpoint 内容

每个 checkpoint 包含：

- 模型权重（`model` 前缀）
- 优化器状态
- 学习率调度器状态
- 当前训练步数

---

## 常见问题排查

### 显存不足 (OOM)

- 降低 `max_batch_frames`（如从 48000 → 24000）
- 降低 `max_batch_size`（如从 48 → 16）
- 确认使用了 `16-mixed` 精度
- 检查是否有其他进程占用显存

### Loss 不降

- 检查数据质量：是否有大量静音、噪声或录音问题
- 检查 F0 提取是否正常：查看预处理日志中是否有大量 "empty f0" 跳过
- 检查学习率是否过大或过小
- 确认 ContentVec 和 RMVPE 权重已正确放置

### 说话人没有分开

- 确认 `use_spk_id: true` 且 `num_spk` 正确
- 检查 `spk_map.json` 中的说话人映射是否正确
- 增加训练步数
- 确保每个说话人数据量充足

### 验证音频失真或嘶嘶声

- 检查 vocoder 权重是否正确加载
- 确认 vocoder 的采样率和 hop size 与 acoustic model 配置匹配
- 检查数据中是否混入了伴奏/混响

### 预处理报错

- `fairseq` 加载失败：参见 README 中的 fairseq 安装说明，或改用 `transformers` 后端
- RMVPE 加载失败：确认 `pe_ckpt` 路径正确
- 磁盘空间不足：二进制数据集可能较大，确保有足够存储空间
