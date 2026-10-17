# iRea

iRea 是 SID 生成式推荐实验代码，包含 TIGER、HSTU、SASRec、LCRec、OneRec，以及独立的 RQVAE 和 RQ-KMeans 训练入口。

- RQVAE 独立训练，将物品文本 embedding 量化为语义 ID（SID）。
- TIGER/HSTU/SASRec 共用 `trainers.general_trainer`；LCRec/OneRec 共用独立的 `trainers.qwen3_trainer`，后者支持单卡和多进程训练。
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

### OneRec RQ-KMeans

从 `genrec` 移植的 OneRec RQ 阶段使用残差 K-means，不训练神经网络 encoder/decoder。配置独立放在 `config/rqkmeans.gin`，无需依赖或导入旁边的 `genrec` 项目：

```bash
CUDA_VISIBLE_DEVICES=0 python -m trainers.rqkmeans_trainer config/rqkmeans.gin \
  --split beauty
```

默认配方与源项目一致：Qwen3-Embedding-8B 物品向量、3 层量化、每层 256 个码字、`random_state=42`、`max_iter=100`、`n_init=10`，开启空簇重置。`backend="auto"` 在物品数乘码本大小小于 `5e7` 时使用完整 KMeans，否则使用 MiniBatchKMeans；`batch_size=4096` 是后者的聚类批大小，不是文本编码批大小。聚类在 CPU 上运行，GPU 仅可能用于首次生成 embedding。

编码器路径由 `config/base.gin` 的 `MODEL_HUB_QWEN3_EMBEDDING_8B` 指定，默认 `/opt/tiger/project/hf_models/Qwen3-Embedding-8B`。已有 `dataset/amazon/processed/beauty/item_emb_Qwen3-Embedding-8B.parquet` 时直接复用；否则走现有 AmazonItemDataset 的文本编码流程。缓存与源项目格式相同，可指定源项目数据目录，前提是数据、编码器和物品映射一致：

```bash
python -m trainers.rqkmeans_trainer config/rqkmeans.gin --split beauty \
  --gin "fit_rqkmeans.dataset_folder='/opt/tiger/project/genrec/dataset/amazon'"
```

输出为 `out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt` 和 `results.json`，后者包含 SID 碰撞率、唯一 SID 数、物品数和最大碰撞组大小。权重保持源项目的 centroid 格式，SID 分配接口为：

```python
from models.rqkmeans import RqKmeans

quantizer = RqKmeans.load("out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt")
sem_ids = quantizer.assign(embeddings).sem_ids  # [N, 3], each code in [0, 255]
```

仅加载可信的 checkpoint，兼容格式包含 NumPy pickle 数据。此入口只负责 RQ 训练，OneRec SFT 使用下方的 `config/onerec.gin`。不会自动将现有三个推荐模型的数据量化器切换为 RQ-KMeans；不要把 `rqkmeans.pt` 传给 `train.pretrained_rqvae_path`。现有 RQVAE 配置和训练流程不变。

### LCRec / OneRec 的 LLM 训练

两者使用独立的 `trainers/qwen3_trainer.py`，参考 `NTR4GR/LC-Rec-backbone` 的 Qwen3 + LoRA 训练流程，不需要导入该项目或预先导出另一套交互数据。`config/onerec.gin` 继承 `config/lcrec.gin`，仅改变模型名称、SID 量化器和物品 embedding 来源：

- LCRec：Sentence-T5-XL + 与 TIGER 相同的 RQVAE checkpoint，3 层 SID，无额外消歧码。
- OneRec：Qwen3-Embedding-8B + `rqkmeans.pt`，3 层 SID；其余主干、prompt、LoRA、训练与评估参数与 LCRec 相同。这里的 OneRec 指 LCRec 主干的 RQ-KMeans SID 变体，不是额外移植另一套推荐网络。
- 两者的推荐主干均为 `/opt/tiger/project/hf_models/Qwen3-1.7B`。不要把用于物品编码的 Qwen3-Embedding-8B 当作推荐主干。

在 `iRea` 根目录运行；先完成对应的 RQVAE 或 RQ-KMeans 训练：

```bash
python -m trainers.qwen3_trainer config/lcrec.gin --split beauty
python -m trainers.qwen3_trainer config/onerec.gin --split beauty

# 四卡脚本入口（上面的普通 python 命令是单进程，不会自动启动四卡）
bash scripts/train_lcrec.sh beauty   # 脚本内指定 GPU 0,1,2,3
bash scripts/train_onerec.sh beauty  # 脚本内指定 GPU 4,5,6,7

# 通用入口的 lcrec/onerec 分支也会转到对应的四卡脚本
bash scripts/train.sh lcrec beauty
bash scripts/train.sh onerec beauty

# 等价四卡命令；每张卡持有完整基础模型及一份独立的最优模型副本
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m trainers.qwen3_trainer config/lcrec.gin --split beauty
```

GPU 编号直接写在各脚本的 `export CUDA_VISIBLE_DEVICES` 中：LCRec 为 `0,1,2,3`，OneRec 为 `4,5,6,7`，不继承外部 GPU 设置；换卡时直接修改该行。可见 GPU 数不是 4 时会报错退出。类别默认 `beauty`，后续 `--gin` 参数原样传给 trainer，`PYTHON` 可指定解释器路径。两个任务同时运行时应指定不同的 GPU 组及输出目录；`--standalone` 自动选择通信端口，不共用固定端口。

新 trainer 的参数前缀是 **`train_lcrec.`**，不是 `train.`。例如：

```bash
python -m trainers.qwen3_trainer config/lcrec.gin --split toys \
  --gin "train_lcrec.batch_size=8" \
  --gin "train_lcrec.gradient_accumulate_every=16"
```

默认 LoRA rank=8、alpha=32、dropout=0.05，训练 `q_proj,v_proj,o_proj,up_proj,down_proj`，同时训练并保存 `embed_tokens,lm_head`。采用 AdamW、lr=`1e-3`、cosine scheduler、200 warmup steps、BF16、梯度检查点、每卡 microbatch=128、梯度累积=1；有效 batch 为 `128 * 1 * GPU 数量`，默认四卡为 512。最大输入长度为 1024 token，推理 beam 数为 20。无 BF16 环境可追加 `--gin "train_lcrec.mixed_precision='no'"`。

数据直接复用 Amazon 的滑窗和 leave-one-out 划分，同一次运行只量化一次物品。历史最大 20 个物品，prompt 为 `What would user be likely to purchase next after buying items {history} ?|start_of_answer|`，SID 注册为 `<a_0><b_1><c_2>` 形式的独立 token。训练只监督目标 SID 的 3 个位置，不计算 EOS、prompt/padding 的 loss；长度超限只删除完整的最早历史物品，不截断目标。推理只传历史 prompt，生成 3 个 SID token 后达到长度上限停止；默认使用真实 SID 目录的前缀约束 beam search，`constrain_sid=False` 则仅约束每层码本，不约束目录中的组合。

默认最多训练 10 个 epoch，每个 epoch 验证一次，连续 2 次验证 CE 无严格降低则早停。验证 CE 只包含目标 SID，排除 EOS 和辅助对齐损失；`--gin "train_lcrec.early_stop_metric='hr'"` 切回 Recall@最大 K。仅在验证指标刷新最优时，用 `copy.deepcopy` 保留的内存最优模型测试一次；验证无改善不测试，训练结束也不再测试、不从磁盘重载。最终结果复用最后一次验证刷新最优时的测试指标，测试表现不参与选模或早停。SID 碰撞不会自动消歧，Recall/NDCG 与现有三个模型一样按完整 SID 匹配，不是唯一 ItemID 匹配。

辅助对齐默认关闭，通过 `train_lcrec.mse_loss_weight` 开启。`align_target` 支持 `item/quantized`，LCRec 额外支持 `latent`；`align_item` 支持 `current/next`。`align_source` 支持 `last_inp`（答案分隔符前一位置）、`last_out`（分隔符位置）、`mean_out`（预测 SID 各 token 的位置均值），损失支持 `cos/gram/cosgram`。cos 保留 `Linear → GELU → Linear` MLP，将 hidden 映射到目标维度；Gram 仅使用一层无 bias 的 `Linear(hidden_dim, hidden_dim, bias=False)`，维度不变、无激活函数。Gram 对线性映射后的表示归一化，在本卡 microbatch 内计算余弦相似度矩阵的 MSE，目标向量维度无需与 hidden 相同。`cosgram` 使用独立的 MLP 和 Linear 分支，仍取 `0.5*cos + 0.5*gram`；纯 gram 不创建 MLP。此前带 bias 的 Linear 实验需与当前版本分开记录，旧 checkpoint 含额外 bias 参数，不能直接严格加载。默认与参考一样为 `quantized/current/last_inp/cos`。

TIGER/SASRec/HSTU 的 `train.align_loss_type` 同样使用上述损失。Gram 先后有原始 hidden、MLP、等维 Linear 三种实现，不可视为同一实验配置；分别保留在 `out`、`out_mlpgram`、`out_lineargram`。当前 `scripts/train_grid.sh` 输出到 `out_lineargram`；Qwen 网格的 gram/cosgram 也使用该目录，cos/base 保持原目录。`out/extract_results.py` 自动读取存在的三个目录，在原页签中分区并列显示。旧 hidden/MLP Gram checkpoint 缺少新 Linear 头，不能直接作为新 Gram 对齐训练的完整 checkpoint 加载。

输出到 `out/lcrec/amazon/<split>/sft/` 或 `out/onerec/amazon/<split>/sft/`，启用对齐会追加配置后缀。`best_model/` 保存 LoRA adapter、可训练 embedding/output head 和 tokenizer，启用 cos 分支时另存 `alignment.pt`，启用 gram 分支时另存 `gram_alignment.pt`，cosgram 保存两个头；基础模型仍从 `base_model` 加载。目录另有 `sid_catalogue.json`、`run_config.json`、`results.json` 和日志，结果中的 `extra` 记录最优轮次及验证值。当前入口是从基础模型开始训练，不包含优化器断点续训接口。

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

### 脚本与批量提交

`scripts/train.sh` 接收模型名、数据集名，并透传后续 `--gin` 参数；默认关闭 W&B。脚本不设置或覆盖 GPU 可见性，由 `mytask` 分配设备。

```bash
cd /opt/tiger/project/iRea/scripts
bash train_grid.sh
```

三个模型分别读取各自的 gin 结构参数；这里遍历的是三种模型，不额外搜索层数或维度。每个数据集需要先准备好对应的 RQVAE 检查点，批量脚本不会启动 RQVAE 训练。

使用本机自动分卡时，先自行执行 `mytask workers local`；已有远程 worker 配置时按现有配置调度。脚本不修改调度器配置。重复提交会使用相同的模型输出目录，请避免相同模型、数据集的任务并发覆盖结果。

两个脚本都直接写 `cd /opt/tiger/project/iRea`，不依赖启动位置或 `mytask` 副本所在目录。没有 help 或 dry-run 分支，执行 `bash train_grid.sh` 就会提交全部 9 个任务；模型和数据集列表直接在脚本里修改。

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
| `train.early_stop_metric` | 默认 `"ce"`：验证 CE 越低越好；`"hr"`：验证 `Recall@max(top_ks)` 越高越好 |
| `train.save_dir_root` | 输出目录；重复使用同一目录会覆盖最佳权重和结果文件 |
| `train.seed` | 推荐模型训练随机种子，默认 42 |

## 6. 训练与推理约定

- **数据**：使用 TIGER 的滑窗方式，在训练序列内生成“历史物品 SID → 下一物品 SID”样本。每位用户最后一个物品用于测试，倒数第二个用于验证，不加入训练目标。
- **编码**：各层 SID 使用独立 offset，padding 为 0。默认 SID 长度为 3，当前配置不添加消歧后缀。
- **目标**：三个推荐模型都使用 teacher forcing 和目标 SID 上的 NTP 交叉熵，不计算历史 token 的 loss，不使用 BCE 或负采样。
- **模型**：TIGER 保留 T5 encoder-decoder；HSTU 保留原骨干和不缩放的 SID embedding 输出头。
- **SASRec 主干**：恢复为 `genrec/genrec/models/sasrec.py` 的结构，保留 `hidden=128`、`FFN=512`、4 个 block、4 个 head、dropout=0.1。使用绝对位置 embedding、`padding_idx=0` 和 Xavier 初始化；CE 模式的输入 embedding 乘 `sqrt(embed_dim)`。每个 block 的 Q 来自 LayerNorm 后输入，K/V 来自原输入，Q/K/V 带 bias、无 O 投影，分数除以 `sqrt(head_dim)`，注意力残差使用归一化后的 Q 输入。FFN 为带 bias 的 ReLU 两层映射，残差使用 FFN 前的原输入。softmax 后屏蔽 padding query，输入和每个 block 后清零 padding；末尾只执行一次 `LayerNorm(eps=1e-8)`，输出为 `hidden @ embedding.T`。`train.d_kv` 仅兼容 `embed_dim/num_heads`，`train.feed_forward_proj` 仅支持 `"relu"`。SID 接口和目标 NTP CE 不变，20 个历史物品、3 层 SID 对应 62 个位置（含目标前缀）。此前 RMSNorm/相对位置版本 checkpoint 不能直接加载，重跑需使用独立输出目录，避免与 SASRec2 旧结果混合。
- **HSTU 位置偏置**：不使用时间戳；距离为 `query位置 − key位置`。query=2 对 keys `[0, 1, 2, 3]` 的位置桶为 `[2, 1, 0, 0]`，未来 key=3 仍被因果掩码屏蔽。
- **推理**：逐 token 自回归 beam search，固定生成一个完整 SID，不读取目标 label。默认不使用物品前缀树约束；`constrain_sid=True` 也只是按码本层限制词表，不保证生成结果对应真实物品。
- **评估**：按完整 SID 匹配计算 Recall、NDCG，不是原始 item ID 的全物品打分。不同物品 SID 相同时无法区分，结果需结合 RQVAE 碰撞率解读。

统一接口为 `model(inp)`，返回 `loss`、`logits`、`last_hidden_state`；推理调用 `model.generate(inp, num_beams=...)`。

SASRec 和 TIGER 在 `eval()` 模式的 SID 生成中默认启用 KV cache，启动命令不用改：

- SASRec 首步计算历史并缓存每层原输入投影得到的 K/V，之后只输入新生成的一个 token；绝对位置索引和因果 mask 的 query 偏移延续完整历史长度（包括左侧 padding），不从零重新编号。
- TIGER 的 encoder 只计算一次，decoder 缓存 self-attention 和 cross-attention K/V，之后只输入新增 token。
- beam search 每步选择父 beam 后会同步复制、重排缓存；缓存仅属于本次调用，不写入检查点。
- HSTU 仍使用完整前缀重算；训练前向不使用缓存。BF16 下缓存与重算可能有小幅浮点差异，接近同分的 beam 顺序可能变化。

对照时可调用 `model.generate(inp, num_beams=30, use_cache=False)` 关闭 SID 生成缓存。直接调用前需要 `model.eval()`；`general_trainer` 的评估已自动处理。

### 可选 Align Loss

默认 `train.mse_loss_weight=0`，只训练 NTP。开启后总损失为：
`NTP + mse_loss_weight * (1 - mean(cos(MLP(hidden), target)))`。
沿用参考实现的参数名 `mse_loss_weight`，实际只支持 cosine，不是 MSE。

- `train.align_target='item'`：生成 SID 所用的原始文本 embedding。
- `train.align_target='latent'`：同一个冻结 RQVAE 的 encoder 输出。
- `train.align_target='quantized'`：该 RQVAE 各层选中码字向量之和，不含消歧后缀。
- `train.align_item='current'`：对齐最后一个历史物品 b；`'next'`：对齐目标物品 s。原参考实现的 `pre` 在这里改名为 `current`。
- 只支持 `train.mse_loss_mode='last'`、`train.align_loss_type='cos'`。输入 a1–a3、b1–b3，目标 s1–s3 时，SASRec/HSTU 取 b3 的最后一层 hidden；TIGER 取 decoder 起始 `pad` 的最后一层 hidden，即预测 s1 的位置，**不是 encoder hidden**。

三类 target 在训练数据初始化时准备，按原始物品索引选择，不从可能碰撞的 SID 反查。目标不回传梯度；投影头结构为 `Linear → GELU → Linear(bias=False)`，参与优化并保存到带配置后缀的 checkpoint。评估和推理不使用对齐目标。

例如运行 TIGER 的 quantized/current 对齐：

```bash
python -m trainers.general_trainer config/tiger.gin --split beauty \
  --gin "train.align_target='quantized'" \
  --gin "train.align_item='current'" \
  --gin "train.mse_loss_weight=5" \
  --gin "train.mse_loss_mode='last'" \
  --gin "train.align_loss_type='cos'" \
  --gin "train.save_dir_root='out/tiger/amazon/{split}/align_quantized_current_w5'" \
  --gin "train.wandb_logging=False"
```

日志分别记录 `train/loss`、`train/ntp_loss`、未加权的 `train/align_loss`，对齐配置写入 `results.json`。不同配置应使用不同输出目录。手动加载启用对齐的检查点时，先调用 `configure_alignment(model, target_dim)` 注册同维度投影头，再加载模型权重。

启用对齐时，checkpoint 和日志文件名均包含 target、方向、权重及模式，例如 `best_model_quantized_current_cos5_last.pt`、`tiger_quantized_current_cos5_last_20261016_120000.log`。关闭对齐时仍使用 `best_model.pt` 和原来的日志命名。

## 7. 输出文件

| 模型 | Beauty 默认输出目录 |
| --- | --- |
| RQVAE | `out/tiger/amazon/beauty/rqvae/` |
| TIGER | `out/tiger/amazon/beauty/` |
| HSTU | `out/hstu/amazon/beauty/sid/` |
| SASRec | `out/sasrec/amazon/beauty/sid/` |
| LCRec | `out/lcrec/amazon/beauty/sft/` |
| OneRec | `out/onerec/amazon/beauty/sft/` |

TIGER/HSTU/SASRec 目录中包含 `best_model.pt`、`results.json` 和训练日志。三个推荐模型默认按验证集 CE 最低值选最佳模型，连续 10 次验证无严格改善则早停。CE 验证使用 teacher forcing，只计算目标 SID 的平均交叉熵，不包含 alignment loss，不运行 beam search。追加 `--gin "train.early_stop_metric='hr'"` 可切回验证集 `Recall@max(top_ks)`（默认 `Recall@10`）最高值选模；每个样本只有一个目标 SID，因此此处 HR 与 Recall 相同。LCRec/OneRec 的 adapter 输出格式及参数前缀见上节。

训练期间不评估测试集。验证指标刷新最优时，通过 `best_model = copy.deepcopy(model).eval()` 保留独立的完整模型副本，设备与训练模型相同，因此会额外占用一份模型的内存或显存。训练结束直接使用 `best_model`，仅计算一次测试集 Recall/NDCG，不再从磁盘加载。仍保存 `best_model.pt` 权重文件供后续使用。`results.json` 记录 `early_stop_metric`、`best_valid_metric`、`best_valid_value`；未计算的 `best_valid_ce` 或 `best_valid_recall` 为 `null`。`best_model.pt` 只保存模型权重，当前 `general_trainer` 没有独立的断点续训或仅评估命令行入口。

## 8. 目录与运行检查

```text
iRea/
  config/
    base.gin           # 文本编码器路径
    rqvae.gin          # 共享 RQVAE 配置
    rqkmeans.gin       # OneRec RQ-KMeans 配置
    tiger.gin
    hstu.gin
    sasrec.gin
    lcrec.gin          # Qwen3 + LoRA，复用 TIGER SID
    onerec.gin         # 继承 LCRec，仅更换 SID 配方
  data/                # 数据加载、滑窗样本和 SID batch
  models/              # 推荐模型、量化器及 SID 公共接口
  modules/             # 公共组件
  trainers/
    general_trainer.py
    rqvae_trainer.py
    rqkmeans_trainer.py
    qwen3_trainer.py
    trainer_utils.py
  scripts/
    train.sh           # 单任务启动
    train_lcrec.sh     # LCRec 四卡训练
    train_onerec.sh    # OneRec 四卡训练
    train_grid.sh      # 9 个任务的 mytask 批量提交
  tests/
    test_kv_cache.py    # 缓存与完整前向的等价性测试
    test_alignment.py  # 对齐来源、hidden 位置、梯度及训练流程
```

查看命令行帮助，不会开始训练或下载数据：

```bash
python -m trainers.general_trainer --help
python -m trainers.rqvae_trainer --help
python -m trainers.rqkmeans_trainer --help
python -m trainers.qwen3_trainer --help
```

运行 KV cache 和对齐回归测试，不需要检查点或下载数据；有可用 GPU 时额外验证缓存的 FP32/BF16：

```bash
python -m unittest discover -s tests -v
```

测试覆盖逐步 logits、生成序列与分数、左侧 padding、父 beam 复制重排、连续调用和训练反向传播。测试通过不等于已完成真实数据指标验证。

本项目使用 MIT 许可证，详见 [LICENSE](LICENSE)。
