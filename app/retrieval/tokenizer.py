"""Load only the embedding model tokenizer; no weights or embedding calls."""
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


@lru_cache(maxsize=4)
def load_tokenizer(model: str, revision: str, cache_dir: str, local_files_only: bool = False) -> tuple[Tokenizer, Path]:
    path = Path(hf_hub_download(repo_id=model, filename="tokenizer.json", revision=revision,
                               cache_dir=cache_dir, local_files_only=local_files_only))
    tokenizer = Tokenizer.from_file(str(path))
    # Measure the entire input. Never silently truncate before splitting.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, path
