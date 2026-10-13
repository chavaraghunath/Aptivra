"""CLI script to export Aptivra 200M model to GGUF format."""

import argparse
from pathlib import Path

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.gguf_exporter import export_model_to_gguf
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Aptivra 200M model to GGUF format")
    parser.add_argument("--output-dir", type=str, default="scratch/hf_export/gguf")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument(
        "--quant-type", choices=["fp32", "fp16", "q8_0"], default="fp16", help="Target quantization"
    )
    args = parser.parse_args()

    model_config = ModelConfig(
        vocab_size=256,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=4096,
        device="cpu",
    )

    print(f"Initializing Aptivra 200M model with {args.experts} experts...")
    model = BackboneTransformer(model_config)
    router = GatingRouter(d_model=model_config.d_model, initial_num_experts=args.experts)
    experts = [ExpertModule(expert_id=i, d_model=model_config.d_model) for i in range(args.experts)]

    out_file = Path(args.output_dir) / f"aptivra-200m-{args.quant_type}.gguf"
    export_model_to_gguf(
        model=model,
        router=router,
        experts=experts,
        model_config=model_config,
        output_path=out_file,
        quant_type=args.quant_type,
    )


if __name__ == "__main__":
    main()
