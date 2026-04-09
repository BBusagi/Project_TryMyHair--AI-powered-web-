from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prepared Stable-Hair request config.")
    parser.add_argument("--config", required=True, help="Absolute path to generated hair_transfer.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_dir = Path(__file__).resolve().parents[1]
    repo_dir = project_dir / "external_models/Stable-Hair"

    sys.path.insert(0, str(repo_dir))

    import numpy as np
    from omegaconf import OmegaConf

    from infer_full import StableHair, concatenate_images

    model = StableHair(config=str(config_path))
    kwargs = OmegaConf.to_container(model.config.inference_kwargs)
    source_image, transfer_result, source_image_bald, reference_image = model.Hair_Transfer(**kwargs)

    output_dir = Path(model.config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / model.config.save_name
    concatenate_images(
        [
            source_image,
            source_image_bald,
            reference_image,
            (transfer_result * 255.0).astype(np.uint8),
        ],
        output_file=str(output_file),
        type="np",
    )
    print(output_file)


if __name__ == "__main__":
    main()
