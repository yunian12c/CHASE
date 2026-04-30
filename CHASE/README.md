# CHASE

本仓库包含 CHASE 的训练与评估代码。该项目基于 GOP、duration、energy 以及可选的 Whisper 对齐特征进行自动发音评估实验，并支持多 seed 训练与结果汇总。

## 目录结构

```text
CHASE/
  src/                # 训练脚本、模型文件、运行脚本、结果汇总脚本
  data/               # GOP 特征、标签文件与对齐数据
  exp/                # 训练输出目录，如 result.csv、train.log、模型权重等
  figure/             # 图表或可视化结果
```

## 环境与依赖

建议使用 Python 3.8+。

安装依赖：

```bash
pip install -r requirements.txt
```

说明：

- 需要安装 `torch`，CUDA 版本请根据本机 GPU 环境自行配置。
- 需要安装 `scikit-learn`，用于 PCA 等处理步骤。

## 数据准备

训练脚本默认从项目内的 `data/` 目录读取数据。本仓库已提供 GOP 相关特征、标签文件和对齐后的 numpy 数据。由于文件体积限制，Whisper 对齐特征未随仓库上传。

默认数据目录如下：

```text
data/
  raw_kaldi_gop/              # GOP 原始 CSV 文件及样本 key 文件
  seq_data_librispeech/       # 训练/测试特征、标签、词级 ID、duration 与 energy 特征
```

其中，`seq_data_librispeech/` 为默认使用的数据目录，对应训练参数：

```bash
--am librispeech
```

如果使用其他声学模型名称，例如 `--am paiia` 或 `--am paiib`，则需要准备对应的数据目录：

```text
data/seq_data_paiia/
data/seq_data_paiib/
```

## Whisper 对齐特征（可选）

Whisper 对齐特征未随本仓库上传。实验代码默认从以下路径读取 Whisper 相关文件：

```text
data/whisper_feature.npz
```

如需完整的 Whisper 对齐特征文件以复现实验结果，请通过邮件联系作者获取。

Contact: `your_email@example.com`

## 训练

### 单次训练

在项目根目录下运行：

```bash
python src/train.py --exp-dir exp/seed3 --seed 3 --am librispeech
```

训练输出将保存到：

```text
exp/seed3/
  train.log
  result.csv
  best_audio_model.pth
```

### 多 seed 批量训练

项目提供了多 seed 批量运行脚本：

```bash
bash src/run.sh
```

默认设置为：

```text
seeds: 3 312 712 644 867
GPU: 0
```

指定 GPU：

```bash
bash src/run.sh --gpus 0,1,2,3
```

指定 seeds：

```bash
bash src/run.sh 3 312 712 --gpus 0,1
```

## 结果汇总

项目提供了结果汇总脚本：

```bash
python src/collect_summary.py
```

默认规则为：

- 对每个 seed，在 `exp/seed{seed}/result.csv` 中选择 `phone_test_mse` 最低的 epoch 作为最佳结果。
- 对多个 seed 的最佳结果计算 mean 和 std。

输出文件保存到 `exp/` 目录下：

```text
exp/result_best_by_phone_test_mse.csv
exp/result_best_mean_std.csv
```

## GitHub 文件说明

本仓库上传了以下数据目录：

```text
data/raw_kaldi_gop/
data/seq_data_librispeech/
```

以下 Whisper 相关文件未随仓库上传：

```text
data/whisper_feature.npz
```

如需 Whisper 对齐特征或完整复现实验所需的附加文件，请通过邮件联系作者获取。

## License

This project is released under the MIT License.
