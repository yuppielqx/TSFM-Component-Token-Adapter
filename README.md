# Component-Token Adapter for Frozen ChronosBolt

本仓库当前的主方法是 **Component-Token Adapter**：在冻结的 ChronosBolt-base 时间序列基础模型上，引入由 trend/seasonal 成分生成的离散时间成分 token，通过轻量级 gated residual adapter 完成下游多变量预测适配。

本 README 面向需要复现实验、继续训练、运行 rollout 或扩展消融实验的合作者。仓库早期的 TS-RAG 检索、PatchTST、few-shot、imputation、reconstruction 和 next-token 代码已经移除；当前代码、checkpoint 和 results 均围绕 Component-Token Adapter 组织。

## 1. 方法概览

给定长度为 `seq_len=512` 的多变量历史窗口，方法包含两条路径：

```text
原始历史序列
    ├── 原始序列 patch 化
    │      └── 冻结 ChronosBolt input embedding + encoder
    │             └── 基础 patch hidden states
    │
    └── trend / seasonal decomposition
           └── component patch 化
                  └── 冻结 encoder
                         └── 冻结 component RQ-VAE codebook
                                └── 离散 component codes
                                       └── 可训练 code/component/stage/position embeddings
                                              └── patch-level token aggregator
                                                     └── token condition

基础 patch hidden states + token condition
    └── gated residual fusion
           └── 冻结 ChronosBolt decoder
                  └── 多分位数预测
                         └── inverse normalization
                                └── 最终预测
```

当前最佳配置是：

- components：`trend seasonal`；
- residual component：关闭（`noresid`）；
- `seq_len=512`；
- adapter 基础预测长度：`pred_len=64`；
- 长 horizon：通过 recursive rollout 得到 `96 192 336 720`；
- backbone：冻结 ChronosBolt-base；
- component codebook：冻结的两阶段 RQ-VAE codebook；
- 训练目标：multi-quantile Pinball loss；
- MSE/MAE：使用中位数 quantile `q=0.5` 的预测计算，仅作为评估指标。

训练时只更新 token embeddings、token aggregator、projection 和 fusion gate；ChronosBolt backbone、输入/输出 patch embedding 和 component codebook 均冻结。

## 2. 服务器环境

当前实验服务器信息：

```text
Host: 10.30.58.200
Project: /data/qingxiangliu/TS-RAG
Python: /home/qingxiangliu/miniforge3/envs/tsrag/bin/python
```

进入项目并设置模块搜索路径：

```bash
cd /data/qingxiangliu/TS-RAG
export PYTHONPATH=.
```

建议使用已有环境：

```bash
/home/qingxiangliu/miniforge3/envs/tsrag/bin/python --version
```

当前代码依赖 PyTorch、Transformers、ChronosBolt 所需的 Hugging Face 组件、NumPy、Pandas、tqdm、safetensors 等。若在新环境运行，应先确认这些包已安装；不要直接删除或替换 `checkpoints/base/`，因为它是冻结 backbone。

依赖清单位于 `requirements.txt`。在已经准备好 CUDA 对应 PyTorch 的环境中，可以执行：

```bash
cd /data/qingxiangliu/TS-RAG
pip install -r requirements.txt
```

`requirements.txt` 不强制指定 CUDA wheel 来源；如果新环境尚未安装 PyTorch，应先按照机器 CUDA 版本从 PyTorch 官方渠道安装匹配的 PyTorch，再安装其余依赖。当前服务器环境验证版本为 PyTorch 2.6.0+cu124、Transformers 4.49.0、NumPy 2.0.2、Pandas 2.2.3、PyArrow 21.0.0。

## 3. 数据集

数据位于：

```text
/data/qingxiangliu/TS-RAG/datasets/
```

当前八个目标数据集为：

| Dataset | 文件 | 频率 | period |
|---|---|---:|---:|
| ETTh1 | `ETT-small/ETTh1.csv` | `h` | 24 |
| ETTh2 | `ETT-small/ETTh2.csv` | `h` | 24 |
| ETTm1 | `ETT-small/ETTm1.csv` | `t` | 96 |
| ETTm2 | `ETT-small/ETTm2.csv` | `t` | 96 |
| Exchange-rate | `exchange_rate/exchange_rate.csv` | `h` | 24 |
| Weather | `weather/weather.csv` | `t` | 144 |
| Electricity | `electricity/electricity.csv` | `t` | 96 |
| Traffic | `traffic/traffic.csv` | `t` | 96 |

数据加载由 `data_provider/data_factory.py` 和 `data_provider/data_loader.py` 完成。当前实验使用全量下游数据：`percent=100`。

## 4. Checkpoint 结构

当前保留的 checkpoint：

```text
checkpoints/
├── base/
│   ├── config.json
│   └── model.safetensors
├── component_rqvae/
│   ├── trend_prototypes_rqvae.pt
│   ├── seasonal_prototypes_rqvae.pt
│   └── residual_prototypes_rqvae.pt
├── component_token_adapter_full_p100_noresid/
├── component_token_adapter_ETTh2_p100_noresid/
├── component_token_adapter_ETTm1_p100_noresid/
├── component_token_adapter_ETTm2_p100_noresid/
├── component_token_adapter_exchange_rate_p100_noresid/
├── component_token_adapter_weather_p100_noresid/
├── component_token_adapter_electricity_p100_noresid/
└── component_token_adapter_traffic_p100_noresid/
```

`component_token_adapter_full_p100_noresid/` 对应 ETTh1 的 full-data adapter。`component_rqvae/` 中的 residual codebook 当前不用于 `noresid` 主实验，但保留用于 residual-on 消融。

## 5. 代码入口

### 5.1 主模型

```text
models/ComponentTemporalTokenAdapter.py
```

该文件实现：

- trend/seasonal 分解后的 component patch 编码；
- 冻结 RQ-VAE quantization；
- code/component/stage/patch-position embeddings；
- patch token aggregation；
- gated residual fusion；
- quantile prediction 和 Pinball loss。

依赖：

- `models/ChronosBolt.py`：ChronosBolt 预测 backbone；
- `models/ChronosBoltPrototype.py`：`ResidualQuantizer` 和相关 RQ-VAE 组件；
- `utils/decomposition.py`：trend/seasonal/residual 选择与分解。

### 5.2 训练入口

当前服务器长作业使用：

```text
train_component_token_adapter.py
```

根目录 train_component_token_adapter.py 是当前唯一推荐的训练入口。现有旧作业使用的 remote_patches/ 路径通过临时兼容软链接映射到根目录，待旧作业结束后可移除该软链接。

### 5.3 Rollout 评估入口

当前服务器长 horizon 评估使用：

```text
eval_component_token_adapter_rollout.py
```

根目录 eval_component_token_adapter_rollout.py 是当前唯一推荐的 rollout 入口。现有旧作业使用的 remote_patches/ 路径通过临时兼容软链接映射到根目录。

## 6. 一键脚本

为方便合作者复现实验，服务器 `script/` 目录新增两个 shell 脚本：

```text
script/train_component_codebooks.sh
script/run_component_token_adapter_all_datasets.sh
```

两个脚本都假定从项目根目录执行，并默认使用服务器上的 Python 环境：

```bash
cd /data/qingxiangliu/TS-RAG
bash script/train_component_codebooks.sh
bash script/run_component_token_adapter_all_datasets.sh
```

也可以通过环境变量覆盖 Python 路径和 GPU：

```bash
PYTHON_BIN=/path/to/python GPU_ID=1 bash script/train_component_codebooks.sh
PYTHON_BIN=/path/to/python GPU_ID=2 DATASET=ETTh1 \
  bash script/run_component_token_adapter_all_datasets.sh
```

`run_component_token_adapter_all_datasets.sh` 默认顺序处理八个数据集，并且每个数据集都执行“训练 64-step adapter → 检查 checkpoint → recursive rollout”三步。由于 Electricity 和 Traffic 很慢，建议先使用 `DATASET=ETTh1` 等单数据集模式验证环境。

## 7. 训练 RQ-VAE component codebook

### 7.1 Codebook 的作用

Component adapter 不直接学习一个新的离散化空间，而是使用冻结的 component RQ-VAE codebook 将 trend、seasonal（以及可选 residual）表示映射为离散 codes。codebook 训练阶段使用冻结的 ChronosBolt encoder，训练 RQ-VAE 的 residual codebook 和 reconstruction heads。

训练目标是 component patch 的重建损失与 commitment loss 的组合。训练、验证、测试按原始时序顺序划分为 70%/10%/20%，避免随机划分造成时间泄漏。

### 7.2 运行 codebook 训练

```bash
cd /data/qingxiangliu/TS-RAG
bash script/train_component_codebooks.sh
```

脚本实际调用：

```bash
python train_component_prototypes.py \
  --components "trend seasonal residual" \
  --pretrained_model_path ./checkpoints/base/ \
  --parquet_path ./datasets/pretrain/retrieval_database_512.parquet \
  --output_dir ./checkpoints/component_rqvae/ \
  --checkpoint_dir ./checkpoints/component_proto_ckpt/
```

完整超参数在 `script/train_component_codebooks.sh` 中，包括：`n_proto=256`、`n_rq_stages=2`、`beta=0.25`、`mask_ratio=0.1`、`train_epochs=20`、`batch_size=128` 和 early stopping `patience=3`。

训练结束后，主要输出为：

```text
checkpoints/component_rqvae/trend_prototypes_rqvae.pt
checkpoints/component_rqvae/seasonal_prototypes_rqvae.pt
checkpoints/component_rqvae/residual_prototypes_rqvae.pt
```

主实验使用 `trend seasonal`，因此 residual codebook 只在 residual-on 消融时需要。训练 codebook 时必须保留 `checkpoints/base/`；adapter 训练和 rollout 也必须使用同一组 codebook。

### 7.3 Codebook 训练数据

默认训练数据为：

```text
datasets/pretrain/retrieval_database_512.parquet
```

该 parquet 文件包含 `x` 列，每一行是长度为 512 的预训练窗口。虽然文件路径历史上包含 `retrieval_database` 字样，但当前 codebook 训练只把它作为预训练窗口数据源；当前 adapter 预测流程不需要外部 retrieval database。

## 8. 训练和预测八个数据集

### 8.1 推荐的完整流程

```text
准备 base checkpoint 和 component codebook
    ↓
为每个数据集训练 64-step adapter
    ↓
检查 checkpoints/<dataset_adapter>/adapter_checkpoint.pt
    ↓
使用同一 adapter 运行 horizons="96 192 336 720" 的 recursive rollout
    ↓
检查 results/<rollout_dir>/*.txt 中的 data_path、MSE、MAE
```

### 8.2 一键运行

```bash
cd /data/qingxiangliu/TS-RAG
bash script/run_component_token_adapter_all_datasets.sh
```

该脚本按 ETTh1、ETTh2、ETTm1、ETTm2、Exchange-rate、Weather、Electricity、Traffic 的顺序执行。每个数据集的 adapter 训练参数为：

```text
percent=100
seq_len=512
pred_len=64
label_len=48
components="trend seasonal"
train_epochs=5
batch_size=64
seed=2021
```

每个数据集的 `freq` 和 `period` 由脚本显式设置。rollout 使用 `pred_len=720` 作为最大预测长度，并通过 `--horizons "96 192 336 720"` 输出四个 horizon 的指标。

### 8.3 单独运行一个数据集

推荐先进行单数据集检查：

```bash
DATASET=ETTh1 GPU_ID=0 \
  bash script/run_component_token_adapter_all_datasets.sh
```

可用的 `DATASET` 值为：

```text
ETTh1 ETTh2 ETTm1 ETTm2 Exchange-rate Weather Electricity Traffic
```

注意，脚本是串行执行的。若某个数据集已经有可靠 checkpoint，但只需要重新评估，应直接调用第 7 节的 rollout 命令，而不是重复训练。

## 9. 训练一个数据集

下面是 ETTh1 的示例。其他数据集只需修改 `root_path`、`data_path`、`freq` 和 `period`。

```bash
cd /data/qingxiangliu/TS-RAG
export PYTHONPATH=.

/home/qingxiangliu/miniforge3/envs/tsrag/bin/python \
  train_component_token_adapter.py \
  --model_id ETTh1_component_token_adapter \
  --data custom \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --freq h \
  --percent 100 \
  --seq_len 512 \
  --pred_len 64 \
  --label_len 48 \
  --pretrained_model_path ./checkpoints/base/ \
  --component_codebook_dir ./checkpoints/component_rqvae/ \
  --components "trend seasonal" \
  --period 24 \
  --train_epochs 5 \
  --batch_size 64 \
  --num_workers 4 \
  --gpu_loc 0 \
  --seed 2021 \
  --checkpoint_dir ./checkpoints/component_token_adapter_full_p100_noresid/ \
  --results_dir ./results/component_token_adapter_full_p100_noresid/
```

训练完成后，最佳验证 Pinball loss 对应的 adapter 会写入：

```text
checkpoints/<adapter_name>/adapter_checkpoint.pt
```

训练日志中：

- `loss` 是反向传播使用的 multi-quantile Pinball loss；
- `val_mse`、`val_mae` 是中位数 quantile 的点预测指标；
- checkpoint 选择依据是 validation Pinball loss，而不是 MSE/MAE。

## 10. Rolling evaluation

使用已经训练好的 64-step adapter 对多个 horizon 递归预测：

```bash
cd /data/qingxiangliu/TS-RAG
export PYTHONPATH=.

/home/qingxiangliu/miniforge3/envs/tsrag/bin/python \
  eval_component_token_adapter_rollout.py \
  --model_id ETTh1_component_token_adapter \
  --data custom \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --freq h \
  --percent 100 \
  --seq_len 512 \
  --pred_len 720 \
  --label_len 48 \
  --horizons "96 192 336 720" \
  --pretrained_model_path ./checkpoints/base/ \
  --component_codebook_dir ./checkpoints/component_rqvae/ \
  --adapter_checkpoint ./checkpoints/component_token_adapter_full_p100_noresid/adapter_checkpoint.pt \
  --components "trend seasonal" \
  --period 24 \
  --batch_size 64 \
  --num_workers 4 \
  --gpu_loc 0 \
  --seed 2021 \
  --results_dir ./results/component_token_adapter_rollout_full_noresid/
```

重要：`--horizons` 必须使用带引号的空格分隔字符串：

```bash
--horizons "96 192 336 720"
```

不要使用：

```bash
--horizons 96,192,336,720
```

后者会被当前 evaluator 拒绝。

## 11. 八个数据集的运行参数

建议为每个数据集单独指定 checkpoint 和 results 目录：

```text
ETTh1       root=./datasets/ETT-small/  data=ETTh1.csv       freq=h  period=24
ETTh2       root=./datasets/ETT-small/  data=ETTh2.csv       freq=h  period=24
ETTm1       root=./datasets/ETT-small/  data=ETTm1.csv       freq=t  period=96
ETTm2       root=./datasets/ETT-small/  data=ETTm2.csv       freq=t  period=96
Exchange    root=./datasets/exchange_rate/ data=exchange_rate.csv freq=h period=24
Weather     root=./datasets/weather/    data=weather.csv     freq=t  period=144
Electricity root=./datasets/electricity/ data=electricity.csv freq=t  period=96
Traffic     root=./datasets/traffic/   data=traffic.csv     freq=t  period=96
```

主实验统一使用：

```text
percent=100
seq_len=512
pred_len=64       # train
pred_len=720      # rollout evaluator upper bound
components="trend seasonal"
horizons="96 192 336 720"
seed=2021
```

## 12. Results 目录和文件命名

当前正式结果使用以下目录：

```text
results/
├── component_token_adapter_full_p100_noresid/              # ETTh1 direct 64-step metrics
├── component_token_adapter_rollout_full_noresid/           # ETTh1 rollout
├── component_token_adapter_ETTh2_p100_noresid/             # ETTh2 direct metrics
├── component_token_adapter_rollout_ETTh2_noresid/          # ETTh2 rollout
├── component_token_adapter_ETTm1_p100_noresid/
├── component_token_adapter_rollout_ETTm1_noresid/
├── component_token_adapter_ETTm2_p100_noresid/
├── component_token_adapter_rollout_ETTm2_noresid/
├── component_token_adapter_exchange_rate_p100_noresid/
├── component_token_adapter_weather_p100_noresid/
└── component_token_adapter_electricity_p100_noresid/
```

部分 rollout 文件名沿用了旧的 `ETTh1_component_token_adapter_rollout.txt`，不能仅凭文件名判断数据集。必须检查文件内部：

```bash
grep '^data_path=' results/<result_dir>/*.txt
```

例如：

```text
ETTh1_component_token_adapter_rollout.txt
data_path=weather.csv
```

这实际上是 Weather 结果。

当前结果表中的 MSE/MAE 是：

- MSE：预测值与真实值的平方误差平均；
- MAE：预测值与真实值的绝对误差平均；
- 两者均从 `q=0.5` median quantile prediction 计算；
- 越低越好。

## 13. 当前已完成结果

已整理的正式 rollout 结果如下（MSE / MAE）：

| Dataset | 96 | 192 | 336 | 720 | Average |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.357013 / 0.372016 | 0.397802 / 0.396354 | 0.420826 / 0.413071 | 0.445639 / 0.440846 | 0.405320 / 0.405572 |
| ETTh2 | 0.242403 / 0.298166 | 0.281186 / 0.331008 | 0.315700 / 0.363287 | 0.407208 / 0.430065 | 0.311624 / 0.355631 |
| ETTm1 | 0.285282 / 0.330095 | 0.334421 / 0.362801 | 0.378452 / 0.391400 | 0.448889 / 0.435485 | 0.361761 / 0.379945 |
| ETTm2 | 0.173468 / 0.241765 | 0.240243 / 0.286406 | 0.297482 / 0.324600 | 0.378593 / 0.377977 | 0.272447 / 0.307687 |
| Exchange-rate | 0.091002 / 0.204051 | 0.188629 / 0.298947 | 0.340773 / 0.417975 | 0.782561 / 0.671657 | 0.350741 / 0.398158 |
| Weather | 0.148809 / 0.182224 | 0.194762 / 0.228600 | 0.247937 / 0.270195 | 0.321267 / 0.320728 | 0.228193 / 0.250437 |

Electricity 和 Traffic 的正式 rollout 结果需要在对应作业完成后补入。

## 14. 复现实验时的注意事项

1. 从项目根目录运行脚本，并设置 `PYTHONPATH=.`。
2. 不要删除或覆盖 `checkpoints/base/` 和 `checkpoints/component_rqvae/`。
3. `components` 参数必须和 checkpoint 中的 adapter 配置一致；当前主结果使用 `trend seasonal`。
4. `period` 必须根据数据集频率设置，错误的 period 会改变 decomposition 和 token assignment。
5. rollout 的 horizon 使用空格分隔，并整体加引号。
6. 结果文件名可能是硬编码的旧名字，读取时检查 `data_path=`。
7. Electricity 有 321 个变量，rolling evaluation 的 batch 数量很大，运行时间显著高于 ETT 数据集。
8. 不要因为训练日志中的 `missing keys` 警告就直接认为 checkpoint 无效；ChronosBolt checkpoint 与当前 adapter 的可训练模块不同，需结合最终 loss、预测输出和结果文件判断。
9. 当前长 horizon 是 recursive rollout，不等同于为每个 horizon 单独训练一个 adapter。

## 15. 消融实验建议

当前方法的主要消融开关包括：

- `components="trend"`：trend only；
- `components="seasonal"`：seasonal only；
- `components="trend seasonal residual"`：residual token on；
- 移除 component embedding；
- 移除 stage embedding；
- 移除 patch-position embedding；
- 移除 codebook conditioning；
- 移除 gated fusion 或比较无 gate 的 residual injection；
- 比较连续 component features 与离散 component tokens。

每个消融都应保持数据划分、随机种子、`seq_len`、训练步数、batch size 和评估流程一致。

## 16. 与 TS-Memory 的比较说明

当前已有结果与 TS-Memory 的 ChronosBolt-base + TS-Memory 数值进行了比较。在 5 个共同数据集、4 个 horizon、MSE/MAE 两项指标的比较中，本文方法取得：

- MSE：12/20 项更低；
- MAE：14/20 项更低。

但该比较不是完全严格的 apples-to-apples protocol：本文使用一个 64-step adapter 递归 rollout 到长 horizon，而 TS-Memory 为不同 horizon 训练独立 adapter。因此正式论文中应表述为“在当前协议下具有竞争力”，并进一步完成 horizon-specific、validation-based alpha selection 和统一数据划分的对齐实验。

## 17. 推荐的协作流程

### 新数据集

1. 准备数据文件并确认 `root_path/data_path/freq/period`；
2. 用 `train_component_token_adapter.py` 训练 64-step adapter；
3. 检查 `adapter_checkpoint.pt` 是否生成；
4. 用 `eval_component_token_adapter_rollout.py` 运行 rollout；
5. 检查结果文件中的 `data_path=` 和各 horizon 指标；
6. 将结果加入统一 benchmark 表。

### 修改模型

1. 先修改 `models/ComponentTemporalTokenAdapter.py`；
2. 做一个小数据 smoke run，确认 forward、loss 和 checkpoint 保存正常；
3. 再提交完整训练；
4. 记录配置、GPU、seed、训练时间和结果目录；
5. 修改 fusion/token 结构后必须重新训练 adapter，不能复用旧 checkpoint。

## 18. Citation

如果工作基于 ChronosBolt，请同时引用对应 Chronos 工作；如果论文继续讨论 TS-RAG 或 TS-Memory，请在正文中明确区分：

- TS-RAG：实例级检索增强；
- TS-Memory：外部时间序列记忆增强；
- 本方法：component-level discrete temporal token adaptation。

## 19. Moirai-2 official-path Component-Token Adapter

本节描述 Moirai-2 版本的实现。它与前文的 ChronosBolt 实验是两套独立
backbone 实现：共享 Component-Token Adapter 的方法思想，但不共享预测
wrapper、checkpoint 或 rollout 代码。

### 19.1 本次 Moirai-2 改动

核心文件如下：

```text
models/moirai2/Moirai2ComponentTokenAdapter.py
models/moirai2/uni2ts/model/moirai2/forecast.py
models/moirai2/uni2ts/model/moirai2/__init__.py
train_moirai2_component_token_adapter.py
eval_moirai2_component_token_adapter.py
train_moirai2_component_prototypes.py
script/train_moirai2_component_codebooks.sh
```

主要改动包括：

1. 引入 `Moirai2Forecast`，由官方 forecast wrapper 负责输入打包、mask、
   patch token 构造、递归 token assignment 和 quantile 预测；
2. 将 adapter 从原先的手工 future-hidden 解码方式改为 official-path hook
   注入，不再自行复制 Moirai-2 的长预测逻辑；
3. evaluator 新增 `--rollout_mode official`，并将其设为默认模式；
4. evaluator 新增 `--disable_adapter`，用于在完全相同的代码路径中运行
   官方 zero-shot 对照；
5. code embedding table 使用对应 RQ-VAE prototype 初始化，而不是随机初始化；
6. residual projection 使用零初始化，使未训练 adapter 从严格 identity 开始；
7. 预训练 codebook 允许跨数据周期共享。codebook metadata 中的训练周期仅
   产生 warning，不再阻止 period=7/144 等下游数据；
8. `script/train_moirai2_component_codebooks.sh` 的 `PYTHONPATH` 已对齐当前
   `models/moirai2` 目录结构。

### 19.2 为什么称为 official-path

这里的 “official” 指 Moirai-2 的预测过程，而不是指 Component-Token
Adapter 由 Moirai-2 官方提供。Adapter 是本仓库的方法，backbone forecast
路径来自 Moirai-2/Uni2TS 的官方实现。

官方路径负责：

- 根据 `context_length` 和 `prediction_length` 构造过去与未来 patch；
- 构造 `past_observed_target`、`past_is_pad` 和 prediction mask；
- 保留多条 quantile trajectory 进行长预测；
- 按官方顺序递归赋值未来 token，并重新计算后续 quantiles；
- 输出完整的 `(batch, quantile, horizon)` 预测张量。

因此 H=96/192/336/720 都直接交给 `Moirai2Forecast` 完成。代码不会先生成
一条 median point forecast，再把该点预测手工拼回上下文。`rollout_chunk=64`
对应 `num_predict_token * patch_size`，只是官方模型一次可赋值的最大未来
长度，不改变最终 horizon。

两种实验的唯一区别为是否注册 adapter hooks：

```text
official zero-shot = Moirai2Forecast + frozen Moirai2Module + adapter disabled
ours               = Moirai2Forecast + frozen Moirai2Module + adapter enabled
```

当调用 `model.set_adapter_enabled(False)` 或 evaluator 使用
`--disable_adapter` 时，代码不会注册 pre-hook/encoder-hook，输出就是未修改的
官方 `Moirai2Forecast` 输出。开发验证中，disabled-adapter 在 H=64 和 H=720
均与直接调用官方 wrapper 逐元素一致（`max_abs_diff=0`）。

### 19.3 Adapter 如何注入官方预测过程

Moirai-2 backbone 和 component RQ-VAE codebook 始终冻结。一次 official
module invocation 中，adapter 执行以下步骤：

```text
official packed target + prediction mask
    -> 找到每个样本最后一个已知 patch
    -> 重建最近 512 步 raw context
    -> trend / seasonal decomposition
    -> frozen Moirai-2 component encoder
    -> frozen two-stage RQ quantization
    -> code + component + stage + position embeddings
    -> patch-level token aggregator
    -> component condition tokens

official encoder source hidden state
    -> cross-attention(query=source hidden, key/value=component tokens)
    -> gated residual fusion
    -> 写回同一个 source hidden state
    -> official output head and recursive quantile forecast
```

pre-hook 从官方 module 的 `target` 和 `prediction_mask` 中恢复最近的已知上下文。
在长预测的每次递归调用中，这个上下文都会随已生成未来值更新，因此 component
condition 也会重新计算。encoder-hook 只修改官方 decoder 实际读取的最后一个
已知 source state，不改写 future token 的构造与赋值规则。

设 source hidden 为 $h$，component cross-attention 输出为 $a$，注入形式为：

```text
g  = sigmoid(W_g [h; a] + b_g)
h' = h + g * W_p(a)
```

`W_p` 使用零初始化，所以初始化时严格满足 `h' = h`。训练只更新 component
token embeddings、token aggregator、cross-attention、projection 和 fusion
gate；Moirai-2 backbone 与 RQ codebook 权重不更新。

### 19.4 共享预训练 codebook 与数据周期

codebook 表示预训练得到的 component prototype space，不为每个下游周期重新
训练。下游 `period` 只控制 trend/seasonal decomposition；分解后的 component
仍映射到同一组冻结 prototypes。加载时继续严格检查 backbone、`d_model`、
`patch_size`、prototype 数量和 tensor shape，但不要求 codebook metadata 的
训练周期等于下游周期。

当前 Moirai-2 实验周期为：

| Dataset | `freq` | `period` |
|---|---:|---:|
| ETTh1 / ETTh2 | `h` | 24 |
| ETTm1 / ETTm2 | `t` | 96 |
| Exchange-rate | `d` | 7 |
| Weather | `t` | 144 |
| Electricity | `h` | 24 |
| Traffic | `h` | 24 |

注意：该表是 Moirai-2 实验的实际配置；不要直接套用前文 ChronosBolt 脚本中的
历史周期设置。

### 19.5 训练与官方可比评测协议

当前严格可比协议为：

```text
seq_len=512
train pred_len=64
components="trend seasonal"
train_epochs=20
patience=3
seed=2021
evaluation horizons=96 192 336 720
rollout_mode=official
rollout_chunk=64
```

每个数据集使用训练 split 训练一个 64-step adapter，再分别对四个 horizon
运行完整测试 split。评测 batch size 只影响吞吐量，不改变窗口、预测路径或指标。
MSE/MAE 均由官方 quantile 输出中最接近 0.5 的 central quantile 计算。

训练示例：

```bash
python train_moirai2_component_token_adapter.py \
  --data custom \
  --root_path ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --features M \
  --freq d \
  --seq_len 512 \
  --pred_len 64 \
  --component_codebook_dir ./checkpoints/moirai2_component_rqvae \
  --components "trend seasonal" \
  --period 7 \
  --train_epochs 20 \
  --patience 3 \
  --batch_size 32 \
  --checkpoint_dir ./checkpoints/moirai2_exchange_adapter
```

Adapter 官方路径评测示例：

```bash
python eval_moirai2_component_token_adapter.py \
  --data custom \
  --root_path ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --features M \
  --freq d \
  --seq_len 512 \
  --pred_len 720 \
  --horizons 720 \
  --rollout_mode official \
  --rollout_chunk 64 \
  --component_codebook_dir ./checkpoints/moirai2_component_rqvae \
  --adapter_checkpoint ./checkpoints/moirai2_exchange_adapter/adapter_checkpoint.pt \
  --period 7 \
  --batch_size 128
```

在同一 evaluator 中运行官方 zero-shot 时，保持数据与 horizon 参数不变，并
增加：

```bash
--disable_adapter
```

为了保持 apples-to-apples comparison，zero-shot 与 adapter 必须使用相同的
数据 split、标准化器、`seq_len`、测试窗口、horizon、central quantile 和完整
测试集；不能将手工 point rollout 的结果与 official quantile rollout 混在同一
张主结果表中。
# TinyTimeMixer 官方预测流程 Adapter

仓库同时提供基于 IBM TinyTimeMixer (TTM) 的 component codebook 与 adapter
实现：

- `models/tinytimemixer/`：TTM 模型、RQ-VAE codebook 与
  `TTMComponentTokenAdapter`；
- `train_ttm_component_codebooks.py`：冻结 TTM backbone，训练通用 component
  codebook；
- `train_ttm_component_token_adapter.py`：冻结 TTM 与 codebook，只训练
  trend/seasonal component-token adapter；
- `script/run_ttm_ett_adapter.sh` 和 `script/run_ttm_other_adapter.sh`：使用
  正确数据周期与 TTM frequency token 的单任务入口。

TTM adapter 保留官方预测链路：backbone 后插入 component-token fusion，随后
继续使用官方 decoder、forecast head 和 inverse scaler。96、192、336、720
分别加载官方 `512-{horizon}-ft-r2` checkpoint 并训练独立 adapter，不使用
recursive rollout；四个 horizon 共享同一个冻结的通用 codebook。

参考：[TTM 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf)、
[IBM 官方实现](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer)、
[TTM R2 权重](https://huggingface.co/ibm-research/ttm-research-r2)。
