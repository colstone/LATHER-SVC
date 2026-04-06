# 模型架构

## 系统总览

LATHER-SVC 的前向流程如下：

```
源音频 (wav)
  │
  ├──[16kHz 重采样]──→ ContentVec-768 ──→ Layer Attention (layers 7-12) ──→ Linear projection ──→ content [B, T, 256]
  │                                                                                                    │
  ├──[原始采样率]────→ RMVPE ──→ F0 ──→ mel 频率转换 ──→ Linear ──→ pitch_embed [B, T, 256]            │
  │                                                                                                    │ (+)
  └──[speaker id]───→ Embedding lookup ──→ spk_embed [B, 1, 256] ─────────────────────────────────→ (+)│
                                                                                                       ↓
                                                                                          condition [B, T, 256]
                                                                                                       │
                                                                                         Condition Refiner
                                                                                     (4-layer Transformer + RoPE)
                                                                                                       │
                                                                                          refined condition
                                                                                               │
                                                              ┌────────────────────────────────┤
                                                              ↓                                ↓
                                                    Variance Predictor               embed variances (+)
                                                    (5-layer Conv1d +                        │
                                                     3 heads: B/V/T)                         ↓
                                                                                   final condition
                                                                                        │
                                                              ┌─────────────────────────┤
                                                              ↓                         ↓
                                                   ConvNeXt Aux Decoder        Rectified Flow (LYNXNet2)
                                                   → coarse mel                t: 0.4 → 1.0
                                                   (shallow diffusion 起点)     → refined mel
                                                                                        │
                                                                                        ↓
                                                                               NSF-HiFiGAN vocoder
                                                                                        │
                                                                                        ↓
                                                                              output wav (44.1kHz)
```

---

## 各模块详细说明

### ContentVecExtractor

**源文件**：`modules/content_encoder.py`

ContentVecExtractor 负责从音频中提取内容特征（说话内容信息），去除说话人身份信息。

**加载方式**：

- 优先尝试通过 `fairseq` 的 `checkpoint_utils.load_model_ensemble_and_task` 加载
- 如果 fairseq 不可用，回退到 `transformers` 库的 `HubertModel.from_pretrained`
- 加载后冻结参数（`freeze: true`），不参与训练梯度更新

**Layer Attention 机制**：

- ContentVec（HuBERT）有 12 层 Transformer encoder
- 不同层捕获不同抽象级别的特征：底层偏声学细节，高层偏语义
- `layer_range: [7, 12]` 选取第 7-12 层的 hidden states（共 6 层）
- `LayerAttention` 模块对这 6 层做可学习的 softmax 加权求和
- 替代方案：`layer_mode: single` 只使用 `target_layer` 指定的单层

**帧率对齐**：

- ContentVec 以 16kHz 音频为输入，输出帧率为 50Hz（每帧 20ms）
- mel spectrogram 帧率取决于 `audio_sample_rate / hop_size`：
  - 音质优先：44100 / 512 ≈ 86Hz
  - 速度优先：24000 / 300 = 80Hz
- `align_frame_rate()` 使用线性插值将 50Hz 的 ContentVec 特征上采样到 mel 帧率

**两种使用模式**：

- **训练时**：预处理阶段（binarizer）提前提取并缓存各层 hidden states `[L, T, 768]`，训练时直接从缓存加载，经过 `from_cached_features()` 做 Layer Attention + Linear 投影
- **推理时**：实时从音频提取，经过完整的 `forward()` 流程

**输出**：`[B, T_mel, hidden_size]`（默认 hidden_size=256）

---

### ConditionRefiner

**源文件**：`modules/condition_refiner.py`

ConditionRefiner 继承自 DSRX 的 `FastSpeech2Encoder`，是一个标准的多层 Transformer encoder。

**与 DSRX FastSpeech2Encoder 的区别**：

- 在 DSRX 中，FastSpeech2Encoder 处理的是音素序列 + 时长信息
- 在 LATHER-SVC 中，输入变为 content + pitch + speaker 的叠加向量，不涉及音素
- 重写了 `forward_embedding()`，接受 `condition` 张量而非音素 embedding

**RoPE 的作用**：

- 旋转位置编码（Rotary Position Embedding）提供相对位置信息
- 使模型能够感知帧间距离，对于建模连续音频信号的时间依赖关系至关重要
- 相比绝对位置编码，RoPE 对变长序列有更好的泛化能力

**为什么需要跨帧上下文建模**：

- ContentVec 的每帧特征主要反映局部声学信息
- F0 和 speaker embedding 是独立注入的
- Condition Refiner 让这些信息在时间维度上互相融合，形成全局一致的条件表示
- 这对于 variance prediction 和 diffusion 生成都很重要：模型需要理解前后文才能预测自然的唱法变化

**结构**：4 层 Transformer encoder，hidden_size=256，2 头注意力，GELU 激活

---

### SVCVariancePredictor

**源文件**：`modules/variance_predictor.py`

**设计**：共享 backbone + 独立 head

- **共享 backbone**：5 层 Conv1d（channels=256, kernel_size=5），每层后接 ReLU + LayerNorm + Dropout
- **独立 head**：3 个 `nn.Linear(256, 1)`，分别输出 breathiness、voicing、tension

**共享 backbone 的设计理由**：

- 三个唱法参数在物理上有关联（气声大的时候通常 tension 低、voicing 弱）
- 共享底层特征提取可以学到这些跨参数的关联
- 独立 head 保证每个参数有自己的预测空间

**三个参数的物理含义和值域**：

| 参数          | 物理含义               | 值域         | 单位    |
| ----------- | ------------------ | ---------- | ----- |
| breathiness | 气声程度。噪声分量的能量大小     | [-96, -20] | dB    |
| voicing     | 发声程度。谐波分量的能量大小     | [-96, -12] | dB    |
| tension     | 声带张力。高频谐波相对基频谐波的比率 | [-10, 10]  | logit |

**训练 vs 推理**：

- **训练时**：使用预处理阶段从真实音频提取的 ground truth variance 值来计算 loss，同时用 GT 值嵌入回条件（teacher forcing）
- **推理时**：使用 predictor 的预测值，经过 `clamp()` 限制在合法值域内后嵌入回条件

---

### AuxDecoderAdaptor (ConvNeXt)

**源文件**：`modules/aux_decoder/convnext.py`，`modules/aux_decoder/__init__.py`

ConvNeXt Aux Decoder 负责从条件特征直接预测一个粗略的 mel spectrogram。

**结构**：

- 输入卷积：`Conv1d(hidden_size, 512, kernel_size=7)`
- 6 层 ConvNeXtBlock：depthwise conv + LayerNorm + pointwise conv + GELU + residual
- 输出卷积：`Conv1d(512, mel_bins, kernel_size=7)`

**Coarse mel 预测的作用**：

- 提供 shallow diffusion 的起点
- 相比从纯噪声开始（t=0），从 coarse mel 开始（t=0.4）可以大幅减少所需的采样步数
- Coarse mel 本身不够精细，但提供了正确的大致轮廓

**梯度控制**：

- `aux_decoder_grad: 0.1` 表示从 diffusion loss 反传到 aux decoder 的梯度乘以 0.1
- 这是为了避免 diffusion 的训练目标过度影响 aux decoder，让它专注于自己的 L1 mel loss

---

### RectifiedFlow

**源文件**：`modules/core/reflow.py`

Rectified Flow 是 LATHER-SVC 的核心生成模型，负责将 coarse mel 精修为高质量 mel spectrogram。

**Rectified Flow vs DDPM**：

- DDPM 使用固定的 noise schedule 逐步去噪，需要数百步
- Rectified Flow 学习从噪声到数据的直线路径（velocity field），理论上一步即可完成
- 实际中 20 步（音质优先）或 4 步（速度优先）即可得到高质量结果

**t_start=0.4 Shallow Diffusion 的含义**：

- 标准 Rectified Flow 从 t=0（纯噪声）积分到 t=1（数据）
- Shallow diffusion 将起点从 t=0 改为 t=0.4
- 起点 x(0.4) 由 ConvNeXt Aux Decoder 的 coarse mel 预测和噪声混合得到：`x = 0.4 * coarse_mel + 0.6 * noise`
- 训练时也只在 t ∈ [0.4, 1.0] 范围内采样
- 效果：保留了 coarse mel 的大致结构，只需精修细节，显著减少采样步数

**LYNXNet2 Backbone**：

- LATHER-SVC 使用 LYNXNet2 作为 velocity function 的 backbone
- 结构参数：1024 通道，6 层，kernel_size=31
- 输入：噪声/中间态 mel `[B, 1, M, T]` + 时间步 + 条件 `[B, H, T]`
- 输出：预测的 velocity `[B, 1, M, T]`

**Euler 采样**：

- 默认使用 Euler 方法从 t_start 积分到 t=1
- `dt = (1.0 - t_start) / sampling_steps`
- 每一步：`x += velocity_fn(x, t) * dt`
- 也支持 RK2、RK4、RK5 高阶方法

---

### Vocoder 对接

**Mel 帧率插值**：

- 速度优先模式下，acoustic model 输出 80Hz mel（24kHz / 300 hop）
- NSF-HiFiGAN 期望 86Hz mel（44.1kHz / 512 hop）
- `_prepare_vocoder_inputs()` 使用线性插值将 mel 和 F0 从源帧率重采样到目标帧率

**NSF-HiFiGAN 的高频激励机制**：

- NSF（Neural Source Filter）在标准 HiFiGAN 基础上增加了正弦激励信号
- 使用 F0 生成基频的正弦波作为声源
- 这使得生成的音频在谐波结构上更准确，特别是高频部分
- 采样率统一为 44.1kHz

**24kHz 模式的带宽限制**：

- 速度优先模式的 acoustic model 工作在 24kHz（`fmax: 12000`）
- 通过 mel 插值上采样到 44.1kHz 帧率后送入 vocoder
- 虽然 vocoder 输出 44.1kHz 波形，但有效带宽受限于 12kHz
- 因此速度优先模式的高频细节会弱于音质优先模式

---

## 训练流程

### Loss 函数组成

训练阶段的总 loss 由三部分组成：

```
L_total = L_reflow + λ_aux * L_aux + λ_var * L_var
```

| Loss     | 计算方式                                     | 权重          | 说明                                                |
| -------- | ---------------------------------------- | ----------- | ------------------------------------------------- |
| L_reflow | `RectifiedFlowLoss`：L2 velocity matching | 1.0         | 主 loss，比较预测 velocity 和目标 velocity                 |
| L_aux    | L1 mel loss                              | λ_aux = 0.2 | ConvNeXt aux decoder 的粗略 mel 与 GT mel 的 L1 距离     |
| L_var    | MSE variance loss                        | λ_var = 0.5 | Variance predictor 预测值与 GT variance 的 MSE，三个参数取平均 |

**RectifiedFlowLoss 细节**：

- 在 [t_start, 1.0] 均匀采样时间步 t
- 构造插值：`x_t = (1-t) * noise + t * gt_mel`
- velocity function 预测 `v_pred`
- 目标 velocity：`v_gt = gt_mel - noise`
- Loss = MSE(v_pred, v_gt)，经 non-padding mask 处理

### 数据预处理 Pipeline

`SVCBinarizer.process_item()` 对每个 wav 文件执行：

1. `librosa.load()` 加载音频到目标采样率
2. `get_mel_torch()` 提取 mel spectrogram
3. RMVPE 提取 F0 和 unvoiced flag
4. `DecomposedWaveform` 做谐波-噪声分离
5. 从分离结果中提取 breathiness（噪声能量 dB）、voicing（谐波能量 dB）、tension（高频谐波比 logit）
6. 对三个 variance 参数做正弦窗平滑（smooth_width=0.12s）
7. ContentVec 提取并缓存 hidden states（layers 7-12）
8. 打包为 IndexedDataset（二进制格式，支持随机访问）

---

## 与 DSRX/DiffSinger 的关系

### 从 DSRX 保留的模块

| 模块                     | 路径                                  | 用途                            |
| ---------------------- | ----------------------------------- | ----------------------------- |
| FastSpeech2Encoder     | `modules/fastspeech/tts_modules.py` | 被 ConditionRefiner 继承         |
| RectifiedFlow          | `modules/core/reflow.py`            | 核心 diffusion                  |
| AuxDecoderAdaptor      | `modules/aux_decoder/`              | Shallow diffusion aux decoder |
| LYNXNet2               | `modules/backbones/lynxnet2.py`     | Diffusion backbone            |
| NSF-HiFiGAN            | `modules/nsf_hifigan/`              | Vocoder                       |
| RMVPE                  | `modules/pe/rmvpe/`                 | F0 提取                         |
| BaseTask / BaseDataset | `basics/`                           | 训练框架基础类                       |
| IndexedDataset         | `utils/indexed_datasets.py`         | 二进制数据集                        |
| PyTorch Lightning 训练循环 | `basics/base_task.py`               | 训练/验证/日志                      |
| Muon_AdamW 优化器         | `modules/optimizer/`                | 优化器                           |

### 从 DSRX 删除的模块

| 模块                                     | 原因                       |
| -------------------------------------- | ------------------------ |
| 音素前端（phoneme dictionary, text encoder） | SVC 不需要文本输入              |
| Duration predictor                     | SVC 不需要时长预测              |
| Variance encoder（DSRX 的 SVS 版本）        | 替换为 SVCVariancePredictor |
| DS acoustic / DS variance inference    | 替换为 SVCInfer             |
| Batch inference backend                | 不需要                      |
| ONNX 导出                                | 暂不支持                     |
| LoRA 微调工具                              | 暂未集成到 SVC 流程             |
| Finetune templates                     | 不适用                      |

### LATHER-SVC 新建的模块

| 模块                   | 路径                               | 说明                              |
| -------------------- | -------------------------------- | ------------------------------- |
| ContentVecExtractor  | `modules/content_encoder.py`     | ContentVec 提取 + Layer Attention |
| ConditionRefiner     | `modules/condition_refiner.py`   | 条件精炼 Transformer                |
| SVCVariancePredictor | `modules/variance_predictor.py`  | 唱法参数预测                          |
| LatherSVC            | `modules/toplevel_svc.py`        | 顶层 acoustic model               |
| SVCBinarizer         | `preprocessing/svc_binarizer.py` | SVC 数据预处理                       |
| SVCTask              | `training/svc_task.py`           | SVC 训练任务                        |
| SVCInfer             | `inference/svc_infer.py`         | SVC 推理                          |
| WebUI                | `webui.py`                       | Gradio 推理界面                     |
