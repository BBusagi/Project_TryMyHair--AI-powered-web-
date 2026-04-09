from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


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

    @property
    def required_weight_paths(self) -> list[Path]:
        return [
            self.repo_dir / "models/stage1/pytorch_model.bin",
            self.repo_dir / "models/stage2/pytorch_model.bin",
            self.repo_dir / "models/stage2/pytorch_model_1.bin",
            self.repo_dir / "models/stage2/pytorch_model_2.bin",
        ]

    @property
    def runner_path(self) -> Path:
        return PROJECT_DIR / "backend/stable_hair_infer.py"

    def python_executable(self) -> str:
        env_python = os.environ.get("STABLE_HAIR_PYTHON")
        if env_python:
            return env_python

        repo_venv_python = self.repo_dir / ".venv/bin/python"
        if repo_venv_python.exists():
            return str(repo_venv_python)

        return sys.executable

    def inspect_python_dependencies(self) -> dict[str, Any]:
        command = [
            self.python_executable(),
            "-c",
            "import torch, diffusers, omegaconf; print('torch=' + torch.__version__)",
        ]

        try:
            process = subprocess.run(
                command,
                cwd=self.repo_dir if self.repo_dir.exists() else PROJECT_DIR,
                text=True,
                capture_output=True,
                check=False,
                timeout=15,
            )
        except Exception as error:
            return {
                "ok": False,
                "detail": str(error),
            }

        return {
            "ok": process.returncode == 0,
            "detail": (process.stdout or process.stderr).strip(),
        }

    def inspect(self) -> dict[str, Any]:
        required_weights = [str(path) for path in self.required_weight_paths]
        missing_weights = [str(path) for path in self.required_weight_paths if not path.exists()]
        repo_exists = self.repo_dir.exists()
        inference_script = self.repo_dir / "infer_full.py"
        config_template = self.repo_dir / "configs/hair_transfer.yaml"
        dependency_status = self.inspect_python_dependencies()

        return {
            "repoName": self.repo_name,
            "repoDir": str(self.repo_dir),
            "repoExists": repo_exists,
            "inferenceScriptExists": inference_script.exists(),
            "configTemplateExists": config_template.exists(),
            "runnerExists": self.runner_path.exists(),
            "pythonExecutable": self.python_executable(),
            "pythonDependenciesOk": dependency_status["ok"],
            "pythonDependenciesDetail": dependency_status["detail"],
            "requiredWeights": required_weights,
            "missingWeights": missing_weights,
            "ready": repo_exists
            and inference_script.exists()
            and config_template.exists()
            and self.runner_path.exists()
            and dependency_status["ok"]
            and len(missing_weights) == 0,
            "setupHint": (
                "Download Stable-Hair pretrained weights from the official Google Drive, "
                "place stage1/pytorch_model.bin and stage2/pytorch_model*.bin under "
                "external_models/Stable-Hair/models/, then install Stable-Hair requirements "
                "in a dedicated environment and set STABLE_HAIR_PYTHON if needed."
            ),
        }

    def write_request_config(
        self,
        *,
        inputs: TransferInputs,
        request_dir: Path,
        step: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        hair_encoder_scale: float,
        size: int,
        seed: int,
    ) -> Path:
        request_dir.mkdir(parents=True, exist_ok=True)
        output_dir = inputs.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        save_name = (inputs.result_path or output_dir / "stable_hair_result.jpg").name
        config_path = request_dir / "hair_transfer.yaml"
        config_path.write_text(
            "\n".join(
                [
                    'pretrained_model_path: "runwayml/stable-diffusion-v1-5"',
                    "",
                    f'pretrained_folder: "{self.repo_dir / "models/stage2"}"',
                    'encoder_path: "pytorch_model.bin"',
                    'adapter_path: "pytorch_model_1.bin"',
                    'controlnet_path: "pytorch_model_2.bin"',
                    f'bald_converter_path: "{self.repo_dir / "models/stage1/pytorch_model.bin"}"',
                    "",
                    'fusion_blocks: "full"',
                    "",
                    "inference_kwargs:",
                    f'  source_image: "{inputs.source_face}"',
                    f'  reference_image: "{inputs.hair_shape_reference}"',
                    f"  random_seed: {seed}",
                    f"  step: {step}",
                    f"  guidance_scale: {guidance_scale}",
                    f"  controlnet_conditioning_scale: {controlnet_conditioning_scale}",
                    f"  scale: {hair_encoder_scale}",
                    f"  size: {size}",
                    "",
                    f'output_path: "{output_dir}"',
                    f'save_name: "{save_name}"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def build_prepared_command(self, config_path: Path) -> list[str]:
        return [
            self.python_executable(),
            str(self.runner_path),
            "--config",
            str(config_path),
        ]

    def command_text(self, command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)

    def build_command(self, inputs: TransferInputs) -> list[str]:
        return [
            self.python_executable(),
            str(self.runner_path),
            "--config",
            str(self.repo_dir / "configs/hair_transfer.yaml"),
        ]


MODEL_ADAPTERS = {
    "barbershop": BarbershopAdapter(),
    "hairfastgan": HairFastGANAdapter(),
    "stable-hair": StableHairAdapter(),
}
