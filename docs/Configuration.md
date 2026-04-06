# 配置文件说明

## 配置文件层次结构

LATHER-SVC 使用 YAML 配置文件，支持 `base_config` 继承机制。子配置中的值会覆盖父配置中的同名项。

继承链：

```
configs/original/base.yaml          # 基础默认值（训练框架、优化器、硬件等）
  └─ configs/svc_base.yaml          # SVC 公共配置（模型结构、ContentVec、variance、vocoder 等）
       ├─ configs/config_quality.yaml  # 音质优先模式（44.1kHz, 20步）
       └─ configs/config_fast.yaml     # 速度优先模式（24kHz, 4步）
```

使用时只需指定最终配置文件（如 `configs/config_quality.yaml`），系统会自动加载整条继承链。

---

## 关键配置项说明

### 音频参数

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `audio_sample_rate` | int | 44100 | 音频采样率。音质优先 44100，速度优先 24000 | 一般不需要 |
| `hop_size` | int | 512 | STFT 跳步长度。音质优先 512，速度优先 300 | 一般不需要 |
| `fft_size` | int | 2048 | FFT 窗口大小 | 不需要 |
| `win_size` | int | 2048 | 窗口大小 | 不需要 |
| `fmin` | int | 40 | mel 滤波器最低频率 (Hz) | 不需要 |
| `fmax` | int | 16000 | mel 滤波器最高频率 (Hz)。速度优先模式为 12000 | 一般不需要 |
| `audio_num_mel_bins` | int | 128 | mel 频段数 | 不需要 |
| `mel_base` | str | `'e'` | mel 对数底数 | 不需要 |
| `spec_min` | list | `[-12]` | mel 归一化最小值 | 不需要 |
| `spec_max` | list | `[0]` | mel 归一化最大值 | 不需要 |

### ContentVec 配置

在 `contentvec:` 块下配置。

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `checkpoint_path` | str | `ckpt/contentvec/checkpoint_best_legacy_500.pt` | ContentVec 模型权重路径 | **需要**（确认路径正确） |
| `layer_mode` | str | `attention` | 层融合模式。`attention`: 可学习加权融合；`single`: 使用单层 | 不需要 |
| `target_layer` | int | 12 | `single` 模式下使用的层号 | 不需要 |
| `layer_range` | list[int] | `[7, 12]` | `attention` 模式下融合的层范围（闭区间） | 不需要 |
| `freeze` | bool | true | 是否冻结 ContentVec 参数 | 不需要 |
| `input_dim` | int | 768 | ContentVec 输出维度 | 不需要 |

### 模型参数

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `hidden_size` | int | 256 | Condition Refiner 隐藏维度 | 不需要 |
| `enc_layers` | int | 4 | Condition Refiner Transformer 层数 | 不需要 |
| `num_heads` | int | 2 | 注意力头数 | 不需要 |
| `enc_ffn_kernel_size` | int | 3 | FFN 卷积核大小 | 不需要 |
| `ffn_act` | str | `gelu` | FFN 激活函数 | 不需要 |
| `dropout` | float | 0.1 | Dropout 比率 | 不需要 |
| `use_pos_embed` | bool | true | 是否使用位置编码 | 不需要 |
| `use_rope` | bool | true | 是否使用 RoPE（旋转位置编码） | 不需要 |
| `rel_pos` | bool | false | 是否使用相对位置编码 | 不需要 |

### Variance Predictor 配置

在 `variance_predictor:` 块下配置。

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `backbone_layers` | int | 5 | Conv1d backbone 层数 | 不需要 |
| `backbone_channels` | int | 256 | backbone 通道数 | 不需要 |
| `kernel_size` | int | 5 | 卷积核大小 | 不需要 |
| `dropout` | float | 0.1 | Dropout 比率 | 不需要 |

Variance 相关全局参数：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `lambda_var_loss` | float | 0.5 | variance loss 权重系数 |
| `breathiness_db_min` / `breathiness_db_max` | float | -96.0 / -20.0 | breathiness 值域 (dB) |
| `voicing_db_min` / `voicing_db_max` | float | -96.0 / -12.0 | voicing 值域 (dB) |
| `tension_logit_min` / `tension_logit_max` | float | -10.0 / 10.0 | tension 值域 (logit) |
| `breathiness_smooth_width` | float | 0.12 | breathiness 平滑窗口 (秒) |
| `voicing_smooth_width` | float | 0.12 | voicing 平滑窗口 (秒) |
| `tension_smooth_width` | float | 0.12 | tension 平滑窗口 (秒) |

### Diffusion 配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `diffusion_type` | str | `reflow` | 扩散类型，LATHER-SVC 仅支持 `reflow` | 不要修改 |
| `time_scale_factor` | int | 1000 | 时间步缩放因子 | 不需要 |
| `T_start` | float | 0.4 | 训练时 Rectified Flow 的起始时间（shallow diffusion） | 不需要 |
| `T_start_infer` | float | 0.4 | 推理时的起始时间 | 不需要 |
| `sampling_algorithm` | str | `euler` | 采样算法。支持 `euler`/`rk2`/`rk4`/`rk5` | 不需要 |
| `sampling_steps` | int | 20 | 采样步数。音质优先 20，速度优先 4 | 一般不需要 |
| `K_step` | int | 400 | Shallow diffusion 步数参数 | 不需要 |
| `backbone_type` | str | `lynxnet2` | 扩散 backbone 类型 | 不需要 |

`backbone_args:` 块：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `num_channels` | int | 1024 | backbone 通道数 |
| `num_layers` | int | 6 | backbone 层数 |
| `kernel_size` | int | 31 | 卷积核大小 |
| `dropout_rate` | float | 0.0 | Dropout 比率 |

### Shallow Diffusion 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_shallow_diffusion` | bool | true | 是否启用 shallow diffusion |
| `lambda_aux_mel_loss` | float | 0.2 | Aux decoder mel loss 权重 |
| `main_loss_type` | str | `l2` | 主 loss 类型 |
| `main_loss_log_norm` | bool | false | 是否对 loss 使用 log-normal 加权 |

`shallow_diffusion_args:` 块：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `train_aux_decoder` | bool | true | 是否训练 aux decoder |
| `train_diffusion` | bool | true | 是否训练 diffusion |
| `val_gt_start` | bool | false | 验证时是否使用 GT mel 作为起点 |
| `aux_decoder_arch` | str | `convnext` | Aux decoder 架构 |
| `aux_decoder_grad` | float | 0.1 | Aux decoder 梯度缩放系数 |

`aux_decoder_args:` 块：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `num_channels` | int | 512 | ConvNeXt 通道数 |
| `num_layers` | int | 6 | ConvNeXt 层数 |
| `kernel_size` | int | 7 | 卷积核大小 |
| `dropout_rate` | float | 0.1 | Dropout 比率 |

### 训练配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `max_updates` | int | 160000 | 总训练步数 | 可根据需要调整 |
| `max_batch_frames` | int | 48000 | 每 batch 最大帧数 | 根据显存调整 |
| `max_batch_size` | int | 48 | 每 batch 最大样本数 | 根据显存调整 |
| `val_check_interval` | int | 2000 | 每多少步做一次验证 | 可调整 |
| `num_valid_plots` | int | 10 | 验证时生成多少个对比图/音频 | 可调整 |
| `val_with_vocoder` | bool | true | 验证时是否通过 vocoder 合成音频 | 不需要 |
| `num_ckpt_keep` | int | 5 | 保留最近几个 checkpoint | 可调整 |
| `permanent_ckpt_start` | int | 60000 | 从第几步开始保存永久 checkpoint | 可调整 |
| `permanent_ckpt_interval` | int | 10000 | 永久 checkpoint 的保存间隔 | 可调整 |
| `accumulate_grad_batches` | int | 1 | 梯度累积步数 | 可调整 |
| `log_interval` | int | 100 | 日志打印间隔 | 不需要 |

### 优化器配置

`optimizer_args:` 块：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `optimizer_cls` | str | `modules.optimizer.muon.Muon_AdamW` | 优化器类 |
| `lr` | float | 0.0006 | 学习率 |
| `muon_args.weight_decay` | float | 0.1 | Muon 部分的权重衰减 |
| `adamw_args.weight_decay` | float | 0.0 | AdamW 部分的权重衰减 |

`lr_scheduler_args:` 块：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `scheduler_cls` | str | `torch.optim.lr_scheduler.StepLR` | 学习率调度器 |
| `step_size` | int | 5000 | 每多少步衰减 |
| `gamma` | float | 0.8 | 衰减系数 |

### 数据配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `binary_data_dir` | str | - | 二进制数据集存放路径 | **需要** |
| `dataset_size_key` | str | `lengths` | 用于 batch sampler 的数据长度字段 | 不需要 |

`datasets` 列表中每个条目的结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `raw_data_dir` | str | 原始 wav 文件目录路径 |
| `speaker` | str | 说话人名称 |
| `spk_id` | int | 说话人 ID（从 0 开始） |
| `language` | str | 语言标记（目前仅作为元数据，不影响模型） |
| `artifact_level` | int | 数据质量等级。0 = 高质量，1 = 拼接/有瑕疵 |
| `test_prefixes` | list[str] | 用于验证集的文件名前缀列表 |

`binarization_args:` 块：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `shuffle` | bool | true | 是否打乱数据顺序 |
| `num_workers` | int | 16 | 预处理并行工作进程数 |

### 说话人配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `use_spk_id` | bool | true | 是否启用说话人 ID | 不需要 |
| `num_spk` | int | 15 | 说话人总数，必须 ≥ 实际说话人数 | **需要**（匹配你的数据集） |

### Variance Embed 开关

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_breathiness_embed` | bool | true | 是否将 breathiness 嵌入回条件 |
| `use_voicing_embed` | bool | true | 是否将 voicing 嵌入回条件 |
| `use_tension_embed` | bool | true | 是否将 tension 嵌入回条件 |

### Vocoder 配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `vocoder` | str | `NsfHifiGAN` | 声码器类型 | 不需要 |
| `vocoder_ckpt` | str | `ckpt/nsf-hifigan/...` | 声码器权重路径 | **需要**（确认路径正确） |
| `vocoder_sample_rate` | int | 44100 | 声码器采样率 | 不需要 |
| `vocoder_hop_size` | int | 512 | 声码器 hop size | 不需要 |

### F0 / HNSep 配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `pe` | str | `rmvpe` | F0 提取算法 | 不需要 |
| `pe_ckpt` | str | `ckpt/rmvpe/RMVPE.pt` | F0 提取模型路径 | **需要**（确认路径正确） |
| `hnsep` | str | `world` | 谐波-噪声分离算法 | 不需要 |
| `hnsep_ckpt` | str | `ckpt/vr/model.pt` | HN 分离模型路径 | **需要**（确认路径正确） |
| `f0_min` | int | 40 | F0 最低频率 (Hz) | 不需要 |
| `f0_max` | int | 2000 | F0 最高频率 (Hz) | 不需要 |

### 硬件配置

| 配置项 | 类型 | 默认值 | 说明 | 是否需要修改 |
|--------|------|--------|------|-------------|
| `pl_trainer_devices` | str | `auto` | PyTorch Lightning 训练设备 | 可指定 GPU 编号，如 `[0]` |
| `pl_trainer_precision` | str | `16-mixed` | 训练精度 | 不需要 |
| `pl_trainer_accelerator` | str | `auto` | 加速器类型 | 不需要 |
| `pl_trainer_num_nodes` | int | 1 | 节点数 | 多机训练时修改 |

---

## 新建数据集的完整配置示例

假设你有两个说话人 `alice` 和 `bob`，要用音质优先模式训练：

**1. 修改 `configs/svc_base.yaml` 中的 `datasets` 段：**

```yaml
datasets:
  - raw_data_dir: /data/svc/alice      # alice 的干声 wav 目录
    speaker: alice
    spk_id: 0
    language: zh
    artifact_level: 0                   # 高质量数据
    test_prefixes:
      - song_001_seg003                 # 用于验证集的文件名前缀

  - raw_data_dir: /data/svc/bob
    speaker: bob
    spk_id: 1
    language: en
    artifact_level: 0
    test_prefixes:
      - track_005_seg001
```

**2. 修改 `configs/svc_base.yaml` 中的全局参数：**

```yaml
num_spk: 2                             # 说话人总数

# 确认以下路径指向正确的预训练模型
contentvec:
  checkpoint_path: ckpt/contentvec/checkpoint_best_legacy_500.pt

vocoder_ckpt: ckpt/nsf-hifigan/your_vocoder.ckpt
pe_ckpt: ckpt/rmvpe/RMVPE.pt
hnsep_ckpt: ckpt/vr/model.pt
```

**3. 修改 `configs/config_quality.yaml` 中的数据路径：**

```yaml
binary_data_dir: data/my_project/binary   # 二进制数据集输出路径
```

**4. 执行预处理和训练：**

```bash
python scripts/binarize.py --config configs/config_quality.yaml
python scripts/train.py --config configs/config_quality.yaml --exp_name my_project
```
