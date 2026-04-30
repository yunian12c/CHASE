# CHASE

本仓库包含 CHASE 训练/评估代码（基于 GOP + dur/energy + Whisper 对齐特征），并支持多 seed 运行与结果汇总。

## 目录结构

```text
CHASE/
  src/                # 训练脚本、模型、运行脚本、汇总脚本
  data/               # 数据与（可选）Whisper 对齐特征
  exp/                # 训练输出（result.csv、train.log、模型等）
  figure/             # 画图/可视化（如有）
```

## 环境与依赖

建议使用 Python 3.8+。

安装依赖（最小集合）：

```bash
pip install -r requirements.txt
```

说明：
- 需要 `torch`（CUDA 版本按你的机器自行安装/配置）
- 需要 `scikit-learn`（用于 PCA）

## 数据准备

训练脚本默认从项目内 `data/` 读取。

### 1) GOP/标签/对齐后的 numpy 数据

根据 `--am` 选择不同子目录（默认 `librispeech`）：

```text
data/
  seq_data_librispeech/
    tr_feat.npy
    te_feat.npy
    tr_label_phn.npy
    te_label_phn.npy
    tr_label_utt.npy
    te_label_utt.npy
    tr_label_word.npy
    te_label_word.npy
    tr_word_id.npy
    te_word_id.npy
```

若你训练时用 `--am paiia` 或 `--am paiib`，则需要对应的 `data/seq_data_paiia/` 或 `data/seq_data_paiib/`。

### 2) raw_kaldi_gop 的 key 文件（用于对齐 Whisper 特征文件名）

```text
data/
  raw_kaldi_gop/
    librispeech/
      tr_keys_phn.csv
      te_keys_phn.csv
```

### 3) Whisper 对齐特征（可选）

默认位置（也可通过 `--whisper-feat-root` 覆盖）：

```text
data/
  whisper_feature/
    feature_aligned/
      whisper_block25_features/
        train/*.npy
        test/*.npy
```

每个 `.npy` 通常是 `<sample_id>.npy`，形状期望为 `[T, 1280]`，脚本会按 GOP 序列长度 pad/截断。

### 4) Whisper 逐维归一化统计

默认文件（也可通过 `--whisper-stat-path` 覆盖）：

```text
data/whisper_feature.npz
```

需包含 `mean`、`std`（可选 `count`）。

## 训练

### 单次训练

在项目根目录：

```bash
python src/train.py --exp-dir exp/seed3 --seed 3 --am librispeech
```

输出会写入：
- `exp/seed3/train.log`
- `exp/seed3/result.csv`
- `exp/seed3/best_audio_model.pth`

### 多 seed 批量训练（推荐）

脚本：`src/run.sh`

默认不带参数时：
- seeds：`3 312 712 644 867`
- GPU：`0`

```bash
bash src/run.sh
```

指定 GPU：

```bash
bash src/run.sh --gpus 0,1,2,3
```

指定 seeds：

```bash
bash src/run.sh 3 312 712 --gpus 0,1
```

## 汇总 5 次 seed 最佳结果并取平均

脚本：`src/collect_summary.py`

规则：
- 每个 seed：在 `exp/seed{seed}/result.csv` 中选取 `phone_test_mse` 最低的 epoch 作为 best
- 对 5 个 seed 的 best 行做 mean/std

默认直接运行：

```bash
python src/collect_summary.py
```

输出（写到 `exp/` 下）：
- `exp/result_best_by_phone_test_mse.csv`
- `exp/result_best_mean_std.csv`

## 备注（上传 GitHub 建议）

通常不建议把 `data/` 与 `exp/`（模型/特征/日志等大文件）直接提交到 GitHub。
建议只提交代码，并在 README 说明数据如何获取/生成。

