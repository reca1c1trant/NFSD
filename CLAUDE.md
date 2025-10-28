# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is Stability AI's generative models repository, implementing various diffusion models including SDXL (Stable Diffusion XL), SVD (Stable Video Diffusion), SV3D, SV4D, and SV4D 2.0. The codebase uses a config-driven approach where modules are built using `instantiate_from_config()` on YAML configurations.

## Environment Setup

**Python Version:** 3.10 (other versions may encounter conflicts)

**PyTorch 2.0 Installation:**
```bash
python3 -m venv .pt2
source .pt2/bin/activate  # On Windows: .pt2\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements/pt2.txt
pip3 install .
pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
```

**Note:** For autoencoder training, only PyTorch 1.13 is supported.

## Running Models

### SV4D 2.0 (Video-to-4D)
```bash
# Download model
huggingface-cli download stabilityai/sv4d2.0 sv4d2.safetensors --local-dir checkpoints

# Run inference
python scripts/sampling/simple_video_sample_4d2.py --input_path <path/to/video>
```

### SV4D (Video-to-4D with SV3D reference)
```bash
# Download models
huggingface-cli download stabilityai/sv3d sv3d_u.safetensors --local-dir checkpoints
huggingface-cli download stabilityai/sv4d sv4d.safetensors --local-dir checkpoints

# Run inference
python scripts/sampling/simple_video_sample_4d.py --input_path <path/to/video>
```

### SV3D (Image-to-3D)
```bash
# SV3D_u (orbital without camera conditioning)
python scripts/sampling/simple_video_sample.py --input_path <path/to/image.png> --version sv3d_u

# SV3D_p (with camera path control)
python scripts/sampling/simple_video_sample.py --input_path <path/to/image.png> --version sv3d_p --elevations_deg 10.0
```

### SDXL and SVD
```bash
# Streamlit demo for text-to-image and image-to-image
streamlit run scripts/demo/sampling.py --server.port <port>

# Video sampling demo
streamlit run scripts/demo/video_sampling.py

# Gradio demos (community-built)
python -m scripts.demo.gradio_app        # General demo
python -m scripts.demo.gradio_app_sv4d   # SV4D-specific demo
```

## Training

**Basic training command:**
```bash
python main.py --base configs/<config1.yaml> configs/<config2.yaml>
```

Configs are merged left-to-right, with later configs overwriting earlier values.

**Example - MNIST training:**
```bash
python main.py --base configs/example_training/toy/mnist_cond.yaml
```

**Important:** When using non-toy configs (`imagenet-f8_cond.yaml`, `txt2img-clipl.yaml`), search for `USER:` comments in the config files - these indicate sections requiring customization for your dataset. Training latent models also requires downloading VAE checkpoints from Hugging Face and updating the `CKPT_PATH` placeholder.

**Training output locations:**
- Logs: `logs/<timestamp>-<name>/`
- Checkpoints: `logs/<timestamp>-<name>/checkpoints/`
- Config snapshots: `logs/<timestamp>-<name>/configs/`

**Logging and monitoring:**
- Training uses PyTorch Lightning with WandB integration
- Configure WandB logging in the config's `lightning.logger` section
- Images are logged via the `ImageLogger` callback (configurable frequency)

**Resuming training:**
```bash
# Resume from checkpoint directory
python main.py --base configs/<config.yaml> --resume logs/<timestamp>-<name>/

# Resume from specific checkpoint file
python main.py --base configs/<config.yaml> --resume logs/<timestamp>-<name>/checkpoints/last.ckpt
```

## Testing

```bash
# Run all inference tests
pytest -v tests/inference/test_inference.py

# Run tests with specific marker
pytest -m inference -v

# Skip inference tests (they require model downloads)
pytest -m "not inference" -v
```

## Code Architecture

### Core Components

**DiffusionEngine (`sgm/models/diffusion.py`):**
- Main diffusion model class (formerly `LatentDiffusion`)
- Handles all conditioning types in a single unified class
- Uses PyTorch Lightning for training orchestration

**Modular Design:**
- **Conditioner** (`sgm/modules/encoders/modules.py`): `GeneralConditioner` handles all conditioning inputs (text, class, spatial) via a list of embedders. Each embedder defines `is_trainable`, `ucg_rate`, and `input_key`.
- **Network** (`network_config`): The neural network backbone (UNet or transformer-based). Configured via YAML.
- **Denoiser** (`sgm/modules/diffusionmodules/denoiser.py`): Implements the denoiser framework for continuous/discrete time models. Independent components:
  - Weighting: `sgm/modules/diffusionmodules/denoiser_weighting.py`
  - Preconditioning: `sgm/modules/diffusionmodules/denoiser_scaling.py`
  - Noise sampling: `sgm/modules/diffusionmodules/sigma_sampling.py`
- **Guiders** (`sgm/modules/diffusionmodules/guiders.py`): Classifier-free guidance and other guidance strategies
- **Samplers** (`sgm/modules/diffusionmodules/sampling.py`): Numerical solvers, independent of the model
- **Autoencoder** (`sgm/models/autoencoder.py`): Encoding/decoding models

### Key Directories

- `sgm/`: Core library code
  - `models/`: `DiffusionEngine`, `AutoencodingEngine`
  - `modules/`: Neural network modules (attention, diffusion, encoders, autoencoders)
  - `data/`: Small map-style datasets (MNIST, CIFAR-10)
- `configs/`: YAML configuration files
  - `inference/`: Model configs for inference (SDXL, SV3D, SV4D, etc.)
  - `example_training/`: Training configuration examples
- `scripts/`: Executable scripts
  - `sampling/`: Inference scripts (`simple_video_sample*.py`)
  - `demo/`: Interactive demos (Streamlit, Gradio)
- `main.py`: Training entry point

### Config-Driven Architecture

Modules are instantiated via `instantiate_from_config()` (defined in `sgm/util.py`) from YAML configs. Each config block requires:
- `target`: Full Python path to the class (e.g., `sgm.models.diffusion.DiffusionEngine`)
- `params`: Dictionary of constructor arguments

Key config sections:
- **conditioner_config**: List of embedders with `input_key`, `is_trainable`, `ucg_rate`
- **network_config**: Neural network specification
- **loss_config**: Loss configuration including `sigma_sampler_config`
- **sampler_config**: Solver type, discretization, guidance settings

**Important:**
- Order of embedders in `conditioner_config` matters for concatenation
- Configs merge left-to-right: `--base config1.yaml config2.yaml` means config2 overwrites config1 values

### Dataset Format

Small datasets should return a dict:
```python
{
    "jpg": tensor,  # -1 to 1, channel-first (CHW)
    "txt": "description"
}
```

For large-scale training, use the [datapipelines](https://github.com/Stability-AI/datapipelines) project with webdataset format.

## Model Variants

### SV4D 2.0 Configuration
- **Default:** 12 frames × 4 views at 576×576
- **8-view variant:** 5 frames × 8 views
- Autoregressive generation for longer videos (21 frames total)

### SV4D Configuration
- 5 frames × 8 views at 576×576
- Requires SV3D reference views from first frame
- Anchor frame sampling for 21-frame generation

### SV3D Configuration
- 21 frames at 576×576
- **SV3D_u:** Orbital videos without camera conditioning
- **SV3D_p:** Custom camera paths with elevation/azimuth control

## Common Parameters

**Low VRAM optimization:**
- `--encoding_t=1`: Encode 1 frame at a time
- `--decoding_t=1`: Decode 1 frame at a time
- `--img_size=512`: Lower resolution

**Background removal:**
- `--remove_bg=True`: Use rembg for plain backgrounds
- For noisy backgrounds: pre-process with Clipdrop or SAM2

## Building Packages

```bash
pip install hatch
hatch build -t wheel
pip install dist/*.whl
```

**Note:** Dependencies are not included in the wheel - install them manually based on your PyTorch version.

## Watermark Detection

Generated images contain invisible watermarks (different from SD 1.x/2.x):
```bash
python scripts/demo/detect.py <filename>          # Single file
python scripts/demo/detect.py <file1> <file2>    # Multiple files
python scripts/demo/detect.py <folder>/*          # Entire folder
```

## Troubleshooting

**CUDA Out of Memory:**
- Use `--encoding_t=1` and `--decoding_t=1` for inference scripts
- Reduce batch size in training configs
- Lower resolution with `--img_size=512`

**Module Import Errors:**
- Ensure you've run `pip install .` from the repo root
- Verify virtual environment is activated
- For training, ensure `sdata` is installed: `pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata`

**Config Errors:**
- Check for `USER:` comments in configs that need customization
- Verify all checkpoint paths exist (look for `CKPT_PATH` placeholders)
- Ensure config merge order is correct (later configs override earlier ones)

**PyTorch Version Conflicts:**
- Use Python 3.10 specifically
- PyTorch 2.0+ for inference and diffusion model training
- PyTorch 1.13 only for autoencoder training

## Output Directories

- `checkpoints/`: Downloaded model weights (not in git, create manually)
- `outputs/`: Generated samples from inference scripts
- `logs/`: Training logs, checkpoints, and configs
- `dist/`: Built wheel packages (from `hatch build`)
