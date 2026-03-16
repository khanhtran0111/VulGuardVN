from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel

from pipeline_common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase E: export embeddings for retrieval")
    parser.add_argument("--features-dir", type=Path, default=Path("data/artifacts/features"))
    parser.add_argument("--model-dir", type=Path, default=Path("model/final_model"))
    parser.add_argument("--out-dir", type=Path, default=Path("index"))
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(str(args.model_dir))
    model.to(device)
    model.eval()

    for split in ["train", "valid", "test"]:
        input_ids = np.load(args.features_dir / f"{split}.input_ids.npy")
        attention_mask = np.load(args.features_dir / f"{split}.attention_mask.npy")

        dataset = TensorDataset(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        vectors = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Embedding {split}"):
                ids, mask = [x.to(device) for x in batch]
                outputs = model(input_ids=ids, attention_mask=mask)
                pooled = mean_pool(outputs.last_hidden_state, mask)
                vectors.append(pooled.cpu().numpy())

        emb = np.concatenate(vectors, axis=0)
        np.save(args.out_dir / f"{split}.emb.npy", emb)

    with (args.out_dir / "embedding_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_model": str(args.model_dir),
                "splits": ["train", "valid", "test"],
                "pooling": "mean_pool_last_hidden_state",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Phase E done.")
    print(f"Embeddings saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
