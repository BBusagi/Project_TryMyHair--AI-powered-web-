from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXTERNAL_MODELS_DIR = PROJECT_DIR / "external_models"


@dataclass(frozen=True)
class TransferInputs:
    source_face: Path
    hair_shape_reference: Path
    hair_color_reference: Path | None = None
    output_dir: Path = PROJECT_DIR / "outputs"
    result_path: Path | None = None


class ModelRepoMissingError(RuntimeError):
    pass


class HairTransferAdapter:
    repo_name: str

    @property
    def repo_dir(self) -> Path:
        return EXTERNAL_MODELS_DIR / self.repo_name

    def require_repo(self) -> Path:
        if not self.repo_dir.exists():
            raise ModelRepoMissingError(
                f"{self.repo_name} repo is missing. Run scripts/clone_model_repos.sh first."
            )

        return self.repo_dir

    def build_command(self, inputs: TransferInputs) -> list[str]:
        raise NotImplementedError

    def run(self, inputs: TransferInputs) -> subprocess.CompletedProcess[str]:
        repo_dir = self.require_repo()
        command = self.build_command(inputs)
        return subprocess.run(
            command,
            cwd=repo_dir,
            text=True,
            capture_output=True,
            check=False,
        )


class BarbershopAdapter(HairTransferAdapter):
    repo_name = "Barbershop"

    def build_command(self, inputs: TransferInputs) -> list[str]:
        color_reference = inputs.hair_color_reference or inputs.hair_shape_reference
        return [
            "python",
            "main.py",
            "--im_path1",
            str(inputs.source_face),
            "--im_path2",
            str(inputs.hair_shape_reference),
            "--im_path3",
            str(color_reference),
            "--sign",
            "hairdesigner_poc",
        ]


class HairFastGANAdapter(HairTransferAdapter):
    repo_name = "HairFastGAN"

    def build_command(self, inputs: TransferInputs) -> list[str]:
        color_reference = inputs.hair_color_reference or inputs.hair_shape_reference
        result_path = inputs.result_path or inputs.output_dir / "hairfastgan_result.png"
        return [
            "python",
            "main.py",
            "--input_dir",
            "",
            "--output_dir",
            str(inputs.output_dir),
            "--face_path",
            str(inputs.source_face),
            "--shape_path",
            str(inputs.hair_shape_reference),
            "--color_path",
            str(color_reference),
            "--result_path",
            str(result_path),
        ]


class StableHairAdapter(HairTransferAdapter):
    repo_name = "Stable-Hair"

    def build_command(self, inputs: TransferInputs) -> list[str]:
        # Upstream infer_full.py reads configs/hair_transfer.yaml instead of CLI image paths.
        # Generate a per-request config before enabling this adapter in a server route.
        return [
            "python",
            "infer_full.py",
        ]


MODEL_ADAPTERS = {
    "barbershop": BarbershopAdapter(),
    "hairfastgan": HairFastGANAdapter(),
    "stable-hair": StableHairAdapter(),
}
