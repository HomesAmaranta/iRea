# iRea

iRea 是 SID 生成式推荐实验代码，包含 TIGER、HSTU、SASRec 和共享的 RQVAE。

- RQVAE 独立训练，将物品文本 embedding 量化为语义 ID（SID）。
- 三个推荐模型共用 `trainers.general_trainer`，数据加载、训练目标和评估流程一致。
- 每个模型使用独立的 `config/模型名.gin`，不再设置配置子目录。

## 1. 环境准备

以下命令都在项目根目录执行：

```bash
cd /opt/tiger/project/iRea
python -m pip install -e . pyarrow
```

`pyarrow` 用于物品 embedding 的 Parquet 缓存。请使用与本机 CUDA 兼容的 PyTorch；推荐单卡 GPU 运行。当前 `general_trainer` 是单进程训练入口，不要用 `torchrun` 启动多个进程。

文本编码器默认在 `config/base.gin` 中配置为：

```text
/opt/tiger/project/hf_models/sentence-t5-xl
```

换机器后需要修改该路径，或在命令后追加：

```bash
--gin "MODEL_HUB_SENTENCE_T5_XL='/你的模型目录/sentence-t5-xl'"
```

下面的启动示例关闭 W&B，避免首次运行需要登录。使用 W&B 时去掉 `--gin "train.wandb_logging=False"`，并先执行 `wandb login`。

**当前环境注意**：本机检查时，RQVAE 入口在导入 W&B 时出现 `cannot import name 'Deprecated' from 'wandb.proto.wandb_telemetry_pb2'`。RQVAE trainer 在模块顶层导入 W&B，关闭日志不能绕过这个错误。需要先在干净的虚拟环境中安装依赖，或修复当前 W&B 安装，再执行 RQVAE 命令；本文没有修改训练代码或依赖环境。

## 2. 数据准备

当前配置使用 Amazon 2014 5-core 数据。通过 `--split` 选择类别，例如 `beauty`、`sports`、`toys`、`clothing`、`home`；不传时默认为 `beauty`。

默认数据根目录是 `dataset/amazon`。以 Beauty 为例：

```text
dataset/amazon/
  raw/beauty/
    reviews_Beauty_5.json.gz
    meta_Beauty.json.gz
  processed/beauty/
    item_emb_sentence-t5-xl.parquet
```

首次运行时，数据加载器会尝试下载缺失的评论和元数据，并生成物品 embedding。已有缓存会直接复用；没有网络时，应提前准备好原始数据和编码器。即使已有 embedding 缓存，加载器仍会检查原始文件是否存在。

要使用其他目录中的已有数据，在 **RQVAE 和推荐模型的每条启动命令** 后追加同一参数，并替换为实际的数据根目录：

```bash
--gin "train.dataset_folder='/path/to/dataset/amazon'"
```

确保缓存来自相同数据、相同编码器，并保持物品映射一致。更换编码器或数据后，不要混用旧缓存和旧 RQVAE 检查点。

## 3. 先训练共享 RQVAE

同一数据类别只需训练一次，三个推荐模型共用其检查点：

```bash
CUDA_VISIBLE_DEVICES=0 python -m trainers.rqvae_trainer config/rqvae.gin \
  --split beauty \
  --gin "train.wandb_logging=False"
```

默认训练 5000 个 epoch，每 50 个 epoch 及最后一个 epoch 保存检查点。epoch 从 0 编号，最终文件为：

```text
out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt
```

虽然配置已独立为 `config/rqvae.gin`，检查点目录仍沿用上面的旧路径，三个推荐模型的默认配置都指向它。

已有同类别、相同结构的 RQVAE 检查点时，可以跳过此步骤。在推荐模型命令后指定：

```bash
--gin "train.pretrained_rqvae_path='/你的检查点目录/checkpoint_epoch_4999.pt'"
```

默认 RQVAE 输入维度为 768，隐层为 `[512, 256, 128, 64]`，量化维度为 32，使用 3 层、每层 256 个码字。更改结构后，要同步修改推荐模型配置中的 `AmazonSeqDataset.rqvae_*`；改变码本大小或 SID 长度时，还要同步 `train.codebook_size` 和 `train.sem_id_dim`。

如果缩短 RQVAE 训练轮数，推荐模型必须改为实际生成的检查点路径。例如训练 10 个 epoch 后使用 `checkpoint_epoch_9.pt`，不能继续读取默认的 `4999`。

## 4. 选择推荐模型运行

完成 RQVAE 训练后，下面三条命令按需选择，不需要依次运行。

### TIGER

```bash
CUDA_VISIBLE_DEVICES=5 python -m trainers.general_trainer config/tiger.gin \
  --split beauty \
  --gin "train.wandb_logging=False"
```

### HSTU

```bash
CUDA_VISIBLE_DEVICES=3 python -m trainers.general_trainer config/hstu.gin \
  --split beauty \
  --gin "train.wandb_logging=False"
```

### SASRec

```bash
CUDA_VISIBLE_DEVICES=4 python -m trainers.general_trainer config/sasrec.gin \
  --split beauty \
  --gin "train.wandb_logging=False"
```

`--split` 会替换配置中的 `{split}`，包括数据类别和检查点路径。切换数据类别时，需要对应类别的 RQVAE，不能直接使用 Beauty 的检查点。

HSTU、SASRec 默认使用 BF16，TIGER 默认不启用混合精度。不支持 BF16 的设备可追加 `--gin "train.mixed_precision='no'"`；该参数属于 `general_trainer`，不要直接用于 RQVAE trainer。

## 5. 常用参数覆盖

`--gin` 可以重复传入。下面示例在已有 RQVAE 的前提下，用 HSTU 跑一个完整 epoch 检查流程；它仍会加载全量数据并执行验证、测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -m trainers.general_trainer config/hstu.gin \
  --split beauty \
  --gin "train.epochs=1" \
  --gin "train.batch_size=32" \
  --gin "train.infer_batch_size=16" \
  --gin "train.num_workers=0" \
  --gin "train.mixed_precision='no'" \
  --gin "train.save_dir_root='out/smoke/hstu/beauty'" \
  --gin "train.wandb_logging=False"
```

| 参数 | 作用 |
| --- | --- |
| `train.learning_rate` | 学习率 |
| `train.max_seq_len` | 最多保留的历史物品数，默认 20；每个物品展开为 3 个 SID token，即最多 60 个历史 token |
| `train.beam_size` | 自回归搜索宽度，默认 30，不能小于最大的评估 K |
| `train.top_ks` | 评估截断位置，默认 `(5, 10)` |
| `train.constrain_sid` | 是否将每一步限制在对应层的码本词表，默认 `False` |
| `train.early_stop_patience` | 验证指标连续不提升的次数阈值 |
| `train.save_dir_root` | 输出目录；重复使用同一目录会覆盖最佳权重和结果文件 |
| `train.seed` | 推荐模型训练随机种子，默认 42 |

## 6. 训练与推理约定

- **数据**：使用 TIGER 的滑窗方式，在训练序列内生成“历史物品 SID → 下一物品 SID”样本。每位用户最后一个物品用于测试，倒数第二个用于验证，不加入训练目标。
- **编码**：各层 SID 使用独立 offset，padding 为 0。默认 SID 长度为 3，当前配置不添加消歧后缀。
- **目标**：三个推荐模型都使用 teacher forcing 和目标 SID 上的 NTP 交叉熵，不计算历史 token 的 loss，不使用 BCE 或负采样。
- **模型**：TIGER 保留 T5 encoder-decoder；SASRec、HSTU 保留各自骨干，输出 logits 直接用最后一层 hidden 乘 SID embedding 词表，不额外按 `d_model` 缩放。
- **HSTU 位置偏置**：不使用时间戳；距离为 `query位置 − key位置`。query=2 对 keys `[0, 1, 2, 3]` 的位置桶为 `[2, 1, 0, 0]`，未来 key=3 仍被因果掩码屏蔽。
- **推理**：逐 token 自回归 beam search，固定生成一个完整 SID，不读取目标 label。默认不使用物品前缀树约束；`constrain_sid=True` 也只是按码本层限制词表，不保证生成结果对应真实物品。
- **评估**：按完整 SID 匹配计算 Recall、NDCG，不是原始 item ID 的全物品打分。不同物品 SID 相同时无法区分，结果需结合 RQVAE 碰撞率解读。

统一接口为 `model(inp)`，返回 `loss`、`logits`、`last_hidden_state`；推理调用 `model.generate(inp, num_beams=...)`。

## 7. 输出文件

| 模型 | Beauty 默认输出目录 |
| --- | --- |
| RQVAE | `out/tiger/amazon/beauty/rqvae/` |
| TIGER | `out/tiger/amazon/beauty/` |
| HSTU | `out/hstu/amazon/beauty/sid/` |
| SASRec | `out/sasrec/amazon/beauty/sid/` |

推荐模型目录中包含 `best_model.pt`、`results.json` 和训练日志。默认按验证集 `Recall@10` 选最佳模型，训练结束会重新加载最佳权重并计算测试指标。`best_model.pt` 只保存模型权重，当前 `general_trainer` 没有独立的断点续训或仅评估命令行入口。

## 8. 目录与运行检查

```text
iRea/
  config/
    base.gin           # 文本编码器路径
    rqvae.gin          # 共享 RQVAE 配置
    tiger.gin
    hstu.gin
    sasrec.gin
  data/                # 数据加载、滑窗样本和 SID batch
  models/              # 四个模型及 SID 公共接口
  modules/             # 公共组件
  trainers/
    general_trainer.py
    rqvae_trainer.py
    trainer_utils.py
```

查看命令行帮助，不会开始训练或下载数据：

```bash
python -m trainers.general_trainer --help
python -m trainers.rqvae_trainer --help
```

当前项目没有 `tests/` 目录，不提供失效的测试命令。推荐模型入口的帮助命令已验证可用；RQVAE 入口需要先解决上面的 W&B 导入问题。本文未启动真实数据训练，不包含未经验证的性能结果。

本项目使用 MIT 许可证，详见 [LICENSE](LICENSE)。
