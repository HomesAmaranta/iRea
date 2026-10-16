"""LCRec prompts over the same Amazon sliding-window samples as TIGER."""

import copy
import string

import torch

from data.amazon import AmazonItemDataset, AmazonSeqDataset
from models.rqkmeans import RqKmeans


ANSWER_TOKEN = "|start_of_answer|"
DEFAULT_PROMPT = "What would user be likely to purchase next after buying items {history} ?"


class AmazonKMeansSeqDataset(AmazonSeqDataset):
    """Change only SID assignment; reuse Amazon sequence splitting and sampling."""

    def __init__(self, root, split, max_seq_len, encoder_model_name,
                 pretrained_rqkmeans_path, codebook_size=256, n_layers=3,
                 align_target=None, align_item="current"):
        if align_target not in (None, "item", "quantized"):
            raise ValueError("RQ-KMeans has no encoder latent; use item or quantized")
        if align_item not in ("current", "next"):
            raise ValueError("align_item must be current or next")
        self.root, self.split = root, split.lower()
        self.train_test_split = "train"
        self._max_seq_len = max_seq_len
        self.add_disambiguation = False
        self.align_item = align_item
        self.alignment_embeddings = None
        quantizer = RqKmeans.load(pretrained_rqkmeans_path.format(split=self.split))
        if (quantizer.codebook_size, quantizer.n_layers) != (codebook_size, n_layers):
            raise ValueError("RQ-KMeans checkpoint and configured SID dimensions differ")
        items = AmazonItemDataset(
            root=root, split=self.split, train_test_split="all",
            encoder_model_name=encoder_model_name,
        )
        self.item_embeddings = torch.as_tensor(items.embeddings, dtype=torch.float32)
        codes = quantizer.assign(items.embeddings).sem_ids
        self.sem_ids_list = codes.tolist()
        if align_target == "item":
            self.alignment_embeddings = self.item_embeddings
        elif align_target == "quantized":
            self.alignment_embeddings = torch.from_numpy(sum(
                centroids[codes[:, layer]]
                for layer, centroids in enumerate(quantizer.centroids)
            ))
        self._load_sequences()
        self._generate_samples()


def load_lm_datasets(root, split, max_seq_len, quantizer, encoder_model_name,
                     pretrained_rqvae_path, pretrained_rqkmeans_path,
                     codebook_size=256, sem_id_dim=3, rqvae_config=None,
                     align_target=None, align_item="current"):
    """Assign SIDs once, then share immutable item tables between data splits."""
    if quantizer == "rqvae":
        config = dict(rqvae_config or {})
        train = AmazonSeqDataset(
            root=root, split=split, train_test_split="train", max_seq_len=max_seq_len,
            encoder_model_name=encoder_model_name,
            pretrained_rqvae_path=pretrained_rqvae_path,
            add_disambiguation=False, rqvae_codebook_size=codebook_size,
            rqvae_n_layers=sem_id_dim, align_target=align_target, align_item=align_item,
            **config,
        )
    elif quantizer == "rqkmeans":
        train = AmazonKMeansSeqDataset(
            root=root, split=split, max_seq_len=max_seq_len,
            encoder_model_name=encoder_model_name,
            pretrained_rqkmeans_path=pretrained_rqkmeans_path,
            codebook_size=codebook_size, n_layers=sem_id_dim,
            align_target=align_target, align_item=align_item,
        )
    else:
        raise ValueError("quantizer must be rqvae or rqkmeans")
    datasets = {"train": train}
    for name in ("valid", "test"):
        value = copy.copy(train)
        value.train_test_split = name
        value.alignment_embeddings = None
        value._generate_samples()
        datasets[name] = value
    return datasets


def sid_token(layer, code):
    if not 0 <= layer < len(string.ascii_lowercase):
        raise ValueError("SID token names support 1 to 26 layers")
    return f"<{string.ascii_lowercase[layer]}_{code}>"


def register_sid_tokens(tokenizer, codebook_size, sem_id_dim):
    """Use ordinary added tokens so decoded SIDs are not skipped as special tokens."""
    if min(codebook_size, sem_id_dim) <= 0:
        raise ValueError("SID dimensions must be positive")
    tokens = [sid_token(layer, code) for layer in range(sem_id_dim)
              for code in range(codebook_size)]
    tokenizer.add_tokens(tokens + [ANSWER_TOKEN])
    encoded = [tokenizer.encode(token, add_special_tokens=False)
               for token in tokens + [ANSWER_TOKEN]]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError("Each SID token and answer separator must encode as one token")
    flat_ids = [ids[0] for ids in encoded]
    if len(set(flat_ids)) != len(flat_ids):
        raise ValueError("SID tokens and answer separator must have distinct token IDs")
    return torch.tensor(flat_ids[:-1], dtype=torch.long).view(sem_id_dim, codebook_size)


class LCSIDCollator:
    """Left-pad prompts; keep target SIDs separate so generation cannot see labels."""

    def __init__(self, tokenizer, sid_token_ids, max_seq_len=20, model_max_length=512,
                 prompt_template=DEFAULT_PROMPT):
        self.tokenizer = tokenizer
        self.sid_token_ids = sid_token_ids
        self.sem_id_dim, self.codebook_size = sid_token_ids.shape
        self.max_seq_len = max_seq_len
        self.model_max_length = model_max_length
        self.prompt_template = prompt_template
        fields = [field for _, field, _, _ in string.Formatter().parse(prompt_template)
                  if field is not None]
        if fields != ["history"] or ANSWER_TOKEN in prompt_template:
            raise ValueError("prompt_template must contain only one {history} field")
        if max_seq_len <= 0 or model_max_length <= self.sem_id_dim + 1:
            raise ValueError("Invalid history limit or model_max_length")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires PAD and EOS token IDs")

    def __call__(self, samples):
        prompts, targets, embeddings = [], [], []
        for sample in samples:
            history = torch.as_tensor(sample.item_ids, dtype=torch.long)
            target = torch.as_tensor(sample.target_ids, dtype=torch.long)
            if history.numel() == 0 or history.numel() % self.sem_id_dim:
                raise ValueError("History must contain complete nonempty SIDs")
            if target.shape != (self.sem_id_dim,):
                raise ValueError("Target length differs from sem_id_dim")
            if ((history < 0) | (history >= self.codebook_size)).any() or (
                (target < 0) | (target >= self.codebook_size)
            ).any():
                raise ValueError("SID code outside the configured codebook")
            history = history.view(-1, self.sem_id_dim)[-self.max_seq_len:].tolist()
            # Remove only whole old items; never truncate the answer or instruction.
            while history:
                text = " ".join(
                    "".join(sid_token(layer, code) for layer, code in enumerate(item))
                    for item in history
                )
                prompt = self.prompt_template.format(history=text) + ANSWER_TOKEN
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) + self.sem_id_dim + 1 <= self.model_max_length:
                    break
                history = history[1:]
            else:
                raise ValueError("model_max_length cannot fit prompt, one item and target")
            prompts.append(ids)
            targets.append(self.sid_token_ids[torch.arange(self.sem_id_dim), target])
            embeddings.append(getattr(sample, "align_target_emb", None))

        width = max(map(len, prompts))
        input_ids = torch.full((len(samples), width), self.tokenizer.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(prompts):
            input_ids[row, -len(ids):] = torch.tensor(ids)
            attention_mask[row, -len(ids):] = 1
        batch = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "labels": torch.stack(targets)}
        if any(value is not None for value in embeddings):
            if any(value is None for value in embeddings):
                raise ValueError("Alignment targets must be present for every sample")
            batch["align_target_emb"] = torch.stack([
                torch.as_tensor(value, dtype=torch.float32) for value in embeddings
            ])
        return batch
