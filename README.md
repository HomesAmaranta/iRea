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

辅助对齐默认关闭，通过 `train_lcrec.mse_loss_weight` 开启。`align_target` 支持 `item/quantized`，LCRec 额外支持 `latent`；`align_item` 支持 `current/next`。`align_source` 支持 `last_inp`（答案分隔符前一位置）、`last_out`（分隔符位置）、`mean_out`（预测 SID 各 token 的位置均值），损失支持 `cos/gram/cosgram`。cos 保留 `Linear → GELU → Linear` MLP，将 hidden 映射到目标维度；Gram 仅使用一层无 bias 的 `Linear(hidden_dim, hidden_dim, bias=False)`，维度不变、无激活函数。Gram 对线性映射后的表示归一化，在本卡 microbatch 内计算余弦相似度矩阵的 MSE，目标向量维度无需与 hidden 相同。`cosgram` 使用独立的 MLP 和 Linear 分支，取 `cos_loss_weight*cos + (1-cos_loss_weight)*gram`。占比范围为 `[0,1]`，默认 `0.5`，只影响 cosgram；总损失仍为 `NTP + mse_loss_weight * align_loss`。纯 gram 不创建 MLP。此前带 bias 的 Linear 实验需与当前版本分开记录，旧 checkpoint 含额外 bias 参数，不能直接严格加载。默认与参考一样为 `quantized/current/last_inp/cos`。

TIGER/SASRec/HSTU 的 `train.align_loss_type` 同样使用上述损失。原始 hidden、MLP、带 bias Linear、无 bias Linear 实验分别保留在 `out`、`out_mlpgram`、`out_lineargram`、`out_linear2gram`，不可视为同一配置。当前 `scripts/train_grid.sh` 输出到 `out_linear2gram`；Qwen 网格的 gram/cosgram 也使用该目录，cos/base 保持原目录。两套网格的 `cos_loss_weights=(0.5)` 可改成例如 `(0.3 0.5 0.8)`，只展开 cosgram。单次训练可传 `--gin "train.cos_loss_weight=0.8"`（Qwen 用 `train_lcrec.cos_loss_weight`）。非默认占比自动在输出目录和日志名追加 `_cw0.8` 等后缀，默认 0.5 保留旧路径。`out/extract_results.py` 自动读取存在的四个目录，在原页签内按批次、损失和 cos 占比分区并列显示，保持分隔和原列宽；旧记录未注明占比时按 0.5 处理。旧 hidden/MLP Gram checkpoint 缺少新 Linear 头，不能直接作为新 Gram 对齐训练的完整 checkpoint 加载。

输出到 `out/lcrec/amazon/<split>/sft/` 或 `out/onerec/amazon/<split>/sft/`，启用对齐会追加配置后缀。Qwen3 训练器不再写入 `best_model/`、LoRA adapter、embedding/output head、tokenizer 或对齐头权重，只保留 `sid_catalogue.json`、`run_config.json`、`results.json` 和日志；结果中的 `extra` 记录最优轮次及验证值。内存中的最佳模型副本和验证刷新最优时的测试流程保持不变，因此仍有该副本的显存开销。已有磁盘 checkpoint 不会自动删除，新实验结束后也没有新权重可供离线加载。当前入口是从基础模型开始训练，不包含优化器断点续训接口；本变更不影响 RQVAE 和 `general_trainer` 的权重保存。

### LCRec 定步数物品探针与序列几何

`qwen3_trainer` 对 LCRec 默认开启 `train_lcrec.linear_probe`，OneRec 默认关闭。每 50 个成功的优化器更新 step 计算一次 CKA/acc（`train_lcrec.linear_probe_every_steps=50`），跨 epoch 累计，梯度累积的 microbatch 不单独计数。epoch 末尾不足 50 步不补测；验证 CE、推荐测试和早停仍按原来的 epoch 调度，不受探针频率影响。探针冻结当前模型，直接复用训练 collator 和 `prompt_template`，历史只放一个商品的完整 3 层 SID，默认输入为 `What would user be likely to purchase next after buying items <a_*><b_*><c_*> ?|start_of_answer|`。取答案分隔符前一 token 的最后层 decoder hidden，即默认模板的问号位置，与训练 `align_source=last_inp` 一致（不随训练的其他 align_source 切换）。不向模型输入目标答案，不额外添加 BOS/EOS；padding、position IDs 与训练一致。直接调用保留 LoRA 的 decoder，不计算词表 logits，不经过对齐头。这里只固定单物品历史，并不复现多物品训练上下文。新记录以 `probe_input_format=single_item_prompt_last_inp_v1` 区分旧版裸 SID 探针，不能当成同一测量口径拼接曲线。

- CKA：与 Sentence-T5 全量商品向量按 ItemID 对齐，按 arXiv:2605.09969 的作者实现做 q95 裁剪、逐行 L2 归一化和 debiased CKA；不采样、不平均 batch CKA，也不额外训练回归映射。
- 同时新增 **raw SID CKA**：每个物品只输入注册的三个 SID token，不经过 prompt collator、不加 BOS/EOS/答案分隔符、不含历史或下一物品。保留 LoRA 的 decoder 最后一层取最后一个 SID 位置的 hidden，不经过投影，计算同样的论文去偏 CKA 和 genrec linear CKA，与全量 Sentence-T5 按 ItemID 对齐。此分支只计算两个 CKA，不重复拟合分类或回归探针。默认随 `linear_probe` 同步每 50 个成功更新 step 执行，rank zero 计算，提取时 eval/no_grad 并恢复模式及 RNG。W&B 指标为 `raw_sid/cka`、`raw_sid/cka_genrec`；日志前缀 `[RawSIDCKA]`；同一个 `metrics.jsonl` 每条记录的 `raw_sid` 字段保存数值、状态和 `probe_input_format=single_item_raw_sid_last_v1`，config 同样说明取值方式。最终 `extra.linear_probe_latest.raw_sid` 随原有 latest/best 机制保存；原 prompt 分支的字段、指标名与统计口径不变。旧裸 SID 的普通 centered cosine-Gram CKA 不能与本分支的去偏 CKA 直接拼接。
- acc：沿用二级类别探针，固定 seed=42 的商品 8:1:1 划分，一个带 bias 的线性分类器、普通 CE，不加类别权重或正则。固定 Adam lr=0.01，全批量，每次更新后计算 valid CE，连续 20 轮无严格降低则早停，最多 1000 轮；恢复 valid CE 最低的分类器，再计算 test acc。Beauty 为 40 个二级类别加缺失二级类，小类别不删除。
- 回归：同一次全量物品 hidden 复用 general trainer 的 `fit_probe`，独立训练 `Linear(hidden, Sentence-T5, bias=True)`，损失为 `MSE + (1-cos)`；与分类共用固定 8:1:1 划分，但不共享拟合权重或优化器。默认 Adam lr=1e-3、最多 3000 轮、验证总损失连续 50 轮不改善早停；恢复最佳权重后计算测试 cos、MSE，以及测试物品查询全量 Sentence-T5 库的 Recall@10/20/50/100。每个探针间隔重新拟合，频率仍由 `linear_probe_every_steps` 控制，不是每 epoch 补测。参数为 `train_lcrec.linear_probe_learning_rate`、`linear_probe_max_epochs`、`linear_probe_patience`。
- 回归结果记录于 `metrics.jsonl` 的 `splits`、`test`、`retrieval`、验证选模历史等字段；`test` 汇总 cos/MSE/acc/Recall。W&B 与 general trainer 同名：`linear_probe/{train,valid,test}_{mse,cosine,loss}`、`linear_probe/test_Recall@{10,20,50,100}`、`linear_probe/best_epoch` 和 `linear_probe/best_valid_loss`。原有论文 CKA、genrec CKA、物品有效秩/cos 都继续使用映射前 hidden，不使用回归预测。Qwen3 仍不保存主干或探针权重。
- 达到最大轮数也是正常结束，不再检查梯度收敛阈值。仅在分类 CE 出现非有限值时，LCRec 记录 `category.status=nonfinite` 和失败轮次，acc 显示 `NA`；保留该步 CKA 并继续训练。数据错误、OOM 等其他异常仍报错。
- hidden、CKA 和分类器数值计算都在 rank zero 的主干设备上运行（GPU 训练时为 GPU）；其他 rank 等待，不重复拟合、不重复写文件。开启探针时分布式通信超时放宽到 60 分钟，以容纳大维度分类拟合。CPU 仅处理标签、划分和文件读写。
- 输出：日志中的 `[ItemProbe]`，以及 `item_probe/<run_id>/config.json`、`category_labels.csv`、`metrics.jsonl`。每条记录含 `epoch` 和 `global_step`；不保存探针或主干权重。最终 `results.json` 的 `extra.linear_probe_latest` 记录最近一次探针；只有探针恰好落在推荐最佳模型的 `best_step` 时才填写 `extra.linear_probe_best`，否则为 `null`，不拿更早的探针冒充最佳模型指标。未达到首个 50 步时 latest 也为 `null`，不按 test acc 选推荐模型。

默认提取 batch 为 32，参考缓存为数据目录下 `processed/<split>/item_emb_sentence-t5-xl.parquet`。关闭用 `--gin "train_lcrec.linear_probe=False"`；调小提取批大小用 `--gin "train_lcrec.linear_probe_batch_size=16"`。也可设置 `train_lcrec.linear_probe_reference_path`、`train_lcrec.linear_probe_seed`。旧 `category_probe_cs` 参数已删除。新增诊断会增加耗时和显存，不影响已启动进程，新进程才生效。

Qwen3 训练默认独立开启 `train_lcrec.sequence_geometry=True`（LCRec/OneRec 均适用）。每个实际训练 microbatch 从同一次前向的最终 decoder norm 输出中，截取预测三个 SID 的 `[答案分隔符, SID0, SID1]` hidden，形状 `[B,3,D]`；这与 general trainer 的有效预测位置口径一致，不是整个 prompt，也不是物品探针使用的问号位置。仅复制并 detach 这三个位置，不额外 forward，不因诊断开启全部层 hidden 输出。与原实现一样，有效秩取全部有效预测位置，cos 跨样本均匀有放回抽 token 对，默认每 batch 4096 对。

序列指标每 epoch 末只报告一次：每卡计算本卡实际 microbatch 的谱秩/cos，再跨卡归约标量和及有效 batch 数，取等权均值；不拼接跨卡 hidden，不平均各卡均值，包含分布式 sampler 实际产生的重复样本。独立 RNG 种子为 `sequence_geometry_seed + epoch * world_size + rank`。日志为 `[SequenceGeometry]`，W&B 为 `geometry/sequence/{effective_rank,avg_cosine}`，文件为 `sequence_geometry_<run_id>.jsonl`，最终结果在 `extra.sequence_geometry_latest` / `extra.sequence_geometry_path`。可用 `train_lcrec.sequence_geometry_cosine_pairs`、`train_lcrec.sequence_geometry_seed` 调整；关闭所有诊断需同时设置 `linear_probe=False` 和 `sequence_geometry=False`。大维度 hidden 的每 batch 谱分解和定步数回归拟合会增加耗时，公式不做近似降维。

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

### 每个 Epoch 的单物品线性探针

`general_trainer` 默认启用 `train.linear_probe=True`，TIGER/HSTU/SASRec 每个 epoch 完成后都对当前模型计算一次。它是独立的诊断任务，不加入推荐训练 loss，不参与推荐模型选模和早停，也不修改主干参数或梯度。此前的 `grad_conflict_every` 入口已移除。

- 所有探针数值计算跟随推荐主干设备。GPU 训练时，hidden 不搬回 CPU：回归训练、MSE/cos、全库检索、全量 CKA、分类器拟合及 accuracy/F1/混淆矩阵均在同一张 GPU 上运行。Sentence-T5 参考向量及 CKA 统计在首次评价时缓存到该 GPU；回归 Linear 在 CPU 隔离 RNG 后初始化，再移到主干设备训练。没有 OOM 后自动退回 CPU 的逻辑；主干显式设为 CPU 时才在 CPU 上计算。
- 全量商品逐个作为独立序列输入，每行只有该商品的 3 个 SID，按现有规则加 offset；可以批量计算，但不同物品不会拼成同一序列。不传 labels，不附加下一商品或历史上下文。
- 原有分支直接调用正常 `model(inp)`，取 `output.last_hidden_state[:, -1, :]`。HSTU/SASRec 对应最后一个输入 SID；TIGER 对应 decoder 起始位，保留不变。提取时临时使用 `eval()`、`no_grad()`，结束后恢复训练状态和随机状态。
- TIGER 默认额外开启 `train.tiger_encoder_probe=True`：同一 epoch 末只运行 encoder，取最后一层三个 SID token 的原始 hidden，转 FP32 后沿 token 维求 mean。先求均值，再按各指标独立预处理；不先归一化，不经过 decoder 或对齐头。用这个 `encoder_mean` 物品矩阵计算两种 CKA、物品有效秩/平均 cos、独立回归探针的测试 MSE/cos/Recall@10/20/50/100，以及独立分类探针的 acc。两分支共享物品划分和训练超参数，但不共享拟合权重。
- 目标固定为 `dataset/amazon/processed/<split>/item_emb_sentence-t5-xl.parquet`，按 `ItemID` 与 SID 行匹配。只读现有缓存，不自动下载或重新编码。即使推荐模型换了 SID 编码器，探针目标也不会自动跟着改变。
- 固定 seed=42，按商品 ID 做 8:1:1 的 train/valid/test 划分，并在全部 epoch 中复用。Beauty 的 12,101 个商品划分为 9,680/1,210/1,211。这是探针的商品划分，不代表推荐模型训练时没见过测试商品；SID 碰撞的商品仍按独立 ItemID 保留。
- 每轮重新训练一层 `Linear(hidden_dim, Sentence-T5_dim, bias=True)`，没有 MLP、没有激活函数。回归目标为 `MSE(prediction, target) + 1 - mean(cosine(prediction, target))`，两项权重均为 1，不标准化 hidden，不预先归一化 target，不加 weight decay。默认 Adam lr=1e-3、FP32、全 train split 更新，最多 3000 轮；连续 50 轮验证总损失无严格改善则早停。恢复 valid `MSE + (1-cos)` 最低的权重后才评估测试集，不合并 train/valid 重训。协议为 `raw_hidden_affine_cosine_plus_mse_v1`，与旧岭回归记录分开。
- 分别记录 train/valid/test 的 MSE、平均 cosine similarity 和总损失。MSE 对所有商品及向量坐标取平均；cosine 越大越好。test 不用于拟合或选模。回归和二级分类复用固定 8:1:1 商品划分，但独立初始化、独立优化器、独立验证选模、独立推理；分类直接读取 raw hidden，不读取回归预测或回归权重。分类的缺标签过滤策略保持不变，过滤后的实际划分计数见 config。
- 测试集映射向量作为 query，在**全量 Sentence-T5 商品库**中按 cosine 检索，按确切 ItemID 计算 `Recall@10/20/50/100`（一 query 一个正确商品），候选包含 train/valid/test 商品，不只在测试子库检索。相同分数按 ItemID 升序打破并列；按 query 分批计算，不分批平均 Recall。k 大于候选数时等价于取全部候选。这里不是推荐任务的下一商品 Recall，也不是分类 top-k accuracy。
- CKA **只算映射前的 hidden**。按 arXiv:2605.09969 作者实现，对 hidden 和 Sentence-T5 各自计算“每行绝对值 q95 的平均值”作为裁剪阈值，再逐行 L2 归一化，计算基于无偏 HSIC 的 debiased CKA，分母加 `1e-6`，保留负值。使用与 U-centering 等价的 FP64 特征空间公式，不构造全量 Gram、不采样、不平均 minibatch CKA。常量表示返回 0；非有限结果记为 `null/NA`。旧 biased CKA 记录不能与新协议直接拼接。
- 另加 **GRHSTU/genrec 口径的 item CKA**，对应 `genrec/modules/cka.py::linear_cka`：复用全量单物品 last hidden 与 ItemID 对齐的 Sentence-T5，不经过回归映射。两矩阵转 FP32、按列减均值，计算 `||Xc.T @ Yc||_F^2 / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)`；关闭 autocast，不裁剪、不做逐行 L2 归一化、不去偏、不加 epsilon、不 clamp。全量计算，不是 batch CKA 均值，也不只在探针测试集计算。零方差等非有限结果从原函数的 NaN 转为 JSON `null`，不向 W&B 写假数值。W&B 名为 `linear_probe/cka_genrec`，日志为 `cka_genrec`，JSON 字段为 `cka_genrec`、`cka_genrec_status`、`cka_genrec_method`，config 中记录 `cka_genrec` 方法。原 `linear_probe/cka` 保持论文版口径。Qwen3 item 探针同步记录新指标，但保留其原有 hidden 提取及评估调度；不新增逐层 CKA。
- 默认同时开启 `train.category_probe=True`：复用同一次提取的**映射前 hidden**，每轮重新训练带 bias 的线性 LogisticRegression 分类器。支持 Amazon 2014 Beauty、Sports、Toys 的根类别下**二级类别**；标签保留父类，例如 `Skin Care > Face` 与 `Makeup > Face` 分开。Beauty 为 40 个二级类别加 `__missing_second_level__` 缺失类（104 个商品），共 41 类，全部 12,101 个商品参与。样本仅 2–6 个的类别也保留，不合并、不按频次过滤、不重采样。按 ItemID 对齐标签，复用相同 8:1:1 划分；训练集未出现的类仍参与测试统计，记入 `unseen_train_classes`，不能预测到时计错。确有多个不同二级标签或完全缺失元数据的商品仍从分类任务排除，不改变回归探针或重新划分。未知数据集或标签不足时明确记录 `skipped`，不伪造 accuracy。
- 分类器仅用 train 拟合标准化及参数，使用每个样本等权的普通交叉熵（`class_weight=None`），不加 L2 或 weight decay。固定 Adam lr=0.01，每轮做一次全批量更新，再计算 valid CE；连续 20 轮无严格降低则早停，最多 1000 轮。恢复 valid CE 最低的权重后才计算 test acc，不搜索 C，不用 accuracy 或 Macro-F1 选模。输出层类别取 train/valid 标签的并集，保证仅在 valid 出现的稀有类也有定义良好的 CE；test 标签不决定输出层，仅在 test 出现的类别保留在统计中并计错。各划分的 Macro-F1、混淆矩阵仅作附加报告，accuracy 为 0 到 1 的比例。标签沿用 `processed/<split>/item_category_probe_v2.json` 缓存与原文件校验，类别、划分、过滤规则不变。
- 新分类协议标记为 `protocol=affine_ce_valid_early_stop_v1`，记录 `best_epoch`（分类器轮次，从 1 开始）、`best_valid_ce`、`epochs_trained`、`stop_reason` 和 CE 历史；历史 train CE 是该轮更新前值，valid CE 是更新后值。旧 C 搜索、一级分类或 `balanced` 结果不会自动重算，不能与新协议直接拼接曲线。独立脚本 `scripts/evaluate_category_probe.py` 也复用同一实现，默认在 CUDA 上拟合。

控制台和原训练 `.log` 每轮输出 `[ItemProbe]`。每次运行还在实验输出目录创建独立的 `item_probe/<run_id>/`：

- `config.json`：参考缓存路径、校验值、SID 校验值、划分的商品 ID、回归学习率/训练预算与检索统计口径。
- `metrics.jsonl`：每轮的三组 MSE/cos/loss、回归验证选模历史、全量原始 hidden CKA、`geometry`、`category` 分类结果和探针耗时。统一 `test` 字段包含 `mse`、`cosine`、`acc`、`Recall@10/20/50/100`；分类不可用时 acc 为 `null`，原因见 `category`。
- `latest_linear.pt`：最近一轮回归探针的验证最优 weight/bias、协议和最佳轮次，独立于推荐模型及分类器 checkpoint。
- `latest_test_ranks.csv`：最近一轮每个测试 ItemID 在全量候选库中的排名。
- `category_labels.csv`：按探针目录排序的类别标签及有效性；`config.json` 中保存分类的有效商品划分、类别数量及多数类预测的 accuracy 基线。
- `latest_category_linear.pt`：最近一次探针中 valid CE 最优分类器的 weight/bias、类别顺序、协议和最佳分类轮次/valid CE，标准化已折入参数，可直接 `F.linear(hidden, weight, bias).argmax(-1)`，不属于推荐模型。

W&B 开启时同步记录 `linear_probe/train_mse`、`linear_probe/valid_mse`、`linear_probe/test_mse`、对应的 `*_cosine`/`*_loss`、`linear_probe/test_Recall@10`（以及 20/50/100）、`linear_probe/cka`、`linear_probe/best_epoch` 和 `linear_probe/best_valid_loss`。分类 acc 单独位于 `category_probe/test_acc`。`results.json` 的 `linear_probe_latest` 是最后一轮诊断，不一定对应推荐模型的最佳轮次。

TIGER encoder-mean 分支的同名 W&B 指标统一增加 `encoder_mean/` 前缀，例如 `encoder_mean/linear_probe/cka`、`encoder_mean/category_probe/test_acc`、`encoder_mean/geometry/item/avg_cosine`。日志标记为 `[ItemProbe:encoder_mean]`，数据和两套探针权重独立保存于 `item_probe_encoder_mean/<run_id>/`，最终结果通过 `encoder_probe_latest` / `encoder_probe_dir` 指向本次运行，config 和每轮记录均含 `hidden_source=encoder_mean`。仅关闭此分支可追加 `--gin "train.tiger_encoder_probe=False"`；关闭 `train.linear_probe` 会同时关闭两分支。它只增加物品级诊断，不改变原来的训练 batch 序列级统计。

重新完成八组 Beauty 训练后，运行 `python debug/fig1_acc_cka_new/summarize.py` 汇总。图表固定为 HSTU、TIGER、TIGER-encoder 三行，仍是八组推荐训练、十二组表示诊断，TIGER 两行引用同一个主模型。只输出 PDF；没有 encoder 序列数据的格子明确标记未计算，不复制 decoder 指标。脚本会拒绝缺少 encoder-mean 数据的旧 TIGER 运行，避免新旧结果混用。

分类任务在同一条 `[ItemProbe]` 日志中追加 `category_level=2`、`category_test_acc`、`category_test_macro_f1`、`category_valid_ce` 和 `category_best_epoch`，并打印实际计算 `device`；W&B 对应 `category_probe/test_acc`、`category_probe/test_macro_f1`、`category_probe/best_valid_ce`、`category_probe/best_epoch`，也记录 train/valid 指标。其开关依赖 `linear_probe`，可追加 `--gin "train.category_probe=False"` 仅关闭分类；旧 `train.category_probe_cs` 参数已删除。不修改推荐训练、选模或早停标准；分类复用同一份 hidden，不增加主干 forward。分类拟合使用 PyTorch Adam、FP64、零初始化，避免消耗训练 RNG，结果记录 `backend=torch_adam` 及实际设备。

例如调整探针提取批大小或关闭诊断：

```bash
python -m trainers.general_trainer config/tiger.gin --split beauty \
  --gin "train.linear_probe_batch_size=256" \
  --gin "train.wandb_logging=False"

# 在原训练命令后追加即可关闭：
# --gin "train.linear_probe=False"
```

可用 `train.linear_probe_seed`、`train.linear_probe_learning_rate=1e-3`、`train.linear_probe_max_epochs=3000`、`train.linear_probe_patience=50` 调整回归探针，或用 `train.linear_probe_reference_path='/path/to/item_emb_sentence-t5-xl.parquet'` 指定参考缓存。旧 `train.linear_probe_alphas` 已移除；旧的 `fit_linear_probe` 仅保留给独立岭回归基线脚本。`scripts/evaluate_checkpoint_retrieval.py` 复用同一 MSE+cos 实现和 Recall@10/20/50/100。每轮梯度训练比闭式岭回归更耗时，可通过训练预算控制。这些任务衡量 Sentence-T5 向量、关系结构和已输入类别信息的保留，不是未知属性推断。正在运行的进程不会自动加载新探针，旧日志也不会自动补齐结果。

### 物品级与序列级几何指标

两组指标都在投影前的 hidden 上计算，不进入 loss，不影响主干梯度、随机状态或早停。有效秩按 NITP 附录 D / Roy-Vetterli 定义：对原始 hidden 按列去均值，取协方差特征值 `lambda`，归一化 `p=lambda/sum(lambda)`，计算 `exp(-sum(p*log(p)))`。等价 SVD 实现必须使用奇异值的平方；不预先做 L2 归一化或 CKA 的 q95 裁剪。

- **物品级**：复用每轮探针提取的全量单物品 `last_hidden_state[:, -1, :]`。cos 是所有不同物品对的精确均值，只做 L2 归一化，排除自身，不中心化。用向量和的等价公式避免构建物品两两矩阵。W&B 名为 `geometry/item/effective_rank`、`geometry/item/avg_cosine`，JSON 位于原探针 `metrics.jsonl` 每条记录的 `geometry` 字段。依赖 `train.linear_probe=True`；Qwen3 物品探针也复用这组计算，沿用其原有 step 调度。
- **序列级**：`general_trainer` 默认 `train.sequence_geometry=True`，复用每个实际训练 batch 的 `output.last_hidden_state`，不额外 forward、不切换 eval，也不积攒整轮 hidden。统计返回的所有有效预测位置；当前 TIGER/HSTU/SASRec 均是 `[B, 3, D]`，不是完整历史序列。TIGER 仍是 decoder 的 `[起始位, SID0, SID1]`，起始位是有效预测位置；用对应目标 labels 的有效性判断，不用历史 attention mask。
- 每个 batch 的有效秩使用上述全部有效位置；cos 从不同序列的非零 token hidden 中均匀、有放回抽取固定数量的 token 对，默认 `train.sequence_geometry_cosine_pairs=4096`。该数量是项目默认值，不是论文公布的参数。使用独立 RNG，种子为 `train.sequence_geometry_seed + epoch`，默认基础种子 42。
- **序列级只在 epoch 末尾报告一次**：对有效的 batch 指标作等权算术平均，不按 token 数加权，也不把全 epoch hidden 混起来重算有效秩。W&B 名为 `geometry/sequence/effective_rank`、`geometry/sequence/avg_cosine`；控制台前缀 `[SequenceGeometry]`；输出目录的 `sequence_geometry_<run_id>.jsonl`（启用 alignment 时还带配置后缀）每轮写一行。`results.json` 保存配置、文件路径及 `sequence_geometry_latest`。
- 中心化后全零的有效秩为 `null`；cos 排除零向量，少于两个非零物品或不足两条有效序列时为 `null`。序列 epoch 汇总记录每项指标的 `valid_batches`、`skipped_batches`，不把无定义值填成 0。若整轮均无定义，不向 W&B 写该数值。数值计算留在主干设备上，使用 FP64。

序列级开关独立于线性探针：关闭所有上述诊断需要同时追加 `--gin "train.linear_probe=False" --gin "train.sequence_geometry=False"`。每 batch 谱分解有额外计算成本；这里只记录训练过程中 batch 几何的均值，不代表 epoch 末固定模型的整体几何。

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

训练期间不评估推荐任务的测试集；每轮探针的商品 test 划分是独立诊断，不参与推荐选模。验证指标刷新最优时，通过 `best_model = copy.deepcopy(model).eval()` 保留独立的完整模型副本，设备与训练模型相同，因此会额外占用一份模型的内存或显存。训练结束直接使用 `best_model`，仅计算一次测试集 Recall/NDCG，不再从磁盘加载。仍保存 `best_model.pt` 权重文件供后续使用。`results.json` 记录 `early_stop_metric`、`best_valid_metric`、`best_valid_value`；未计算的 `best_valid_ce` 或 `best_valid_recall` 为 `null`。`best_model.pt` 只保存模型权重，当前 `general_trainer` 没有独立的断点续训或仅评估命令行入口。

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
