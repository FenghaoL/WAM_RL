"""Train an RLT-inspired reader on explicitly exported frozen token features."""
import argparse
from pathlib import Path

import torch
from ahawam.models.wan22.rl_token_readout import RLTokenReadout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Trusted .pt: tokens [N,M,D], valid [N,M], feature_spec dict")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    torch.manual_seed(42)
    data = torch.load(args.features, map_location="cpu", weights_only=True)
    tokens, valid = data["tokens"], data["valid"]
    spec = data["feature_spec"]
    for key in ("base_checkpoint", "normalization", "token_source", "cache_protocol"):
        if not spec.get(key):
            raise ValueError(f"feature_spec requires {key}")
    if tokens.ndim != 3 or len(tokens) == 0 or valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("Invalid token feature dataset")
    model = RLTokenReadout(tokens.shape[-1], args.hidden_dim, max_tokens=tokens.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(args.steps):
        indices = torch.randint(len(tokens), (min(args.batch_size, len(tokens)),))
        loss, _, _ = model(tokens[indices].to(args.device, torch.float32), valid[indices].to(args.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step={step + 1} reconstruction_mse={loss.item():.6g}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=model.config, reader=model.state_dict(), feature_spec=spec, steps=args.steps), args.output)


if __name__ == "__main__":
    main()
