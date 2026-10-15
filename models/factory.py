"""Build the three SID recommenders with their original backbones."""

from models.hstu import HSTU
from models.sasrec import SASRec
from models.tiger import Tiger


def build_sid_model(
    model_name, codebook_size=256, sem_id_dim=3, max_seq_len=20, model_config=None
):
    config = dict(model_config or {})
    if min(codebook_size, sem_id_dim, max_seq_len) <= 0:
        raise ValueError("SID dimensions and history length must be positive")
    vocab_size = codebook_size * sem_id_dim + 1
    history_tokens = max_seq_len * sem_id_dim
    if model_name == "tiger":
        config.update(
            vocab_size=vocab_size, codebook_size=codebook_size, sem_id_dim=sem_id_dim,
            max_history_tokens=history_tokens, pad_token_id=0, eos_token_id=0,
        )
        return Tiger(config)
    if model_name not in ("sasrec", "hstu"):
        raise ValueError("model_name must be tiger, sasrec or hstu")
    config.update(
        num_items=vocab_size - 1,
        # Reserve positions for the target prefix without truncating history.
        max_seq_len=history_tokens + sem_id_dim - 1,
        codebook_size=codebook_size,
        sem_id_dim=sem_id_dim,
    )
    if model_name == "sasrec":
        config["loss_type"] = "ce"
        return SASRec(**config)
    return HSTU(**config)
