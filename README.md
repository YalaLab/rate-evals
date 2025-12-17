# RATE-Evals

A comprehensive evaluation pipeline for Vision-Language Models on medical imaging tasks, with built-in support for multi-GPU processing, real-time progress tracking, and disease finding classification.

## Installation

### From Source (Recommended for Development)

Clone and install with uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv add flash-attn --no-build-isolation
source .venv/bin/activate

# Install rad-vision-engine
git clone https://github.com/yalalab/rad-vision-engine ../rad-vision-engine
uv pip install -e ../rad-vision-engine
```

### Setting up Console Scripts

After installation, the console scripts (`rate-extract`, `rate-evaluate`) are installed in `~/.local/bin/`. If you get "command not found" errors, you have two options:

1. **Add ~/.local/bin to PATH** (recommended):
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   # or for zsh users:
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Use the module format** (alternative):
   ```bash
   python -m rate_eval.cli.extract [OPTIONS]
   python -m rate_eval.cli.evaluate [OPTIONS]
   ```

## Evaluate Pillar0 on Merlin Abdominal CT Dataset

```bash
# Extract embeddings from Abdominal CT
uv run rate-extract \
    --model pillar0 \
    --dataset abd_ct_merlin \
    --all-splits \
    --batch-size 4 \
    --output-dir cache/pillar0_abd_ct_merlin \
    --model-repo-id YalaLab/Pillar0-AbdomenCT \
    --ct-window-type all \
    --modality abdomen_ct

# Evaluate the model
uv run rate-evaluate \
    --checkpoint-dir cache/pillar0_abd_ct_merlin \
    --dataset-name abd_ct_merlin \
    --labels-json data/merlin/final_results.json \
    --output-dir results/pillar0_abd_ct_merlin
```

## Running with a Custom Example Dataset

You can test the pipeline with a small example dataset using Hydra configuration overrides. This is useful as an example for setting up your own dataset.

### Example: Using Custom Data Paths

We demonstrate how to extract embeddings using the Abdomen CT model using a public example from the Merlin dataset:
```bash
uv run rate-extract \
    --model pillar0 \
    --dataset abd_ct_merlin \
    --split train \
    --batch-size 4 \
    --model-repo-id YalaLab/Pillar0-AbdomenCT \
    --ct-window-type all \
    --output-dir cache/pillar0_abd_ct_merlin \
    data.train_json=data/merlin_example/train.json \
    data.cache_manifest=data/merlin_example/manifest.csv
```

To extract vision embeddings using the CT models, please refer to the example metadata in [data/rve_example](data/rve_example/). 
```bash
uv run rate-extract \
    --model pillar0 \
    --dataset rve_chest_ct \ # rve_abd_ct, rve_brain_ct, rve_chest_ct 
    --split train \
    --batch-size 4 \
    --model-repo-id YalaLab/Pillar0-ChestCT \ # YalaLab/Pillar0-AbdomenCT, YalaLab/Pillar0-BrainCT, YalaLab/Pillar0-ChestCT 
    --ct-window-type all \
    --output-dir /path/to/cache \
    data.train_json=/path/to/json \
    data.cache_manifest=/path/to/csv
```

### Key Points

1. **Hydra Configuration Overrides**: The `data.train_json` and `data.cache_manifest` arguments use Hydra-style overrides (without `--` prefix, using `key=value` format)
2. **Required File Structure**:
   - `train.json`: JSON file with sample metadata (e.g., `{"sample_name": "EXAMPLE_ACCESSION", "nii_path": null, "report_metadata": "FINDINGS: ..."}`)
   - `manifest.csv`: CSV mapping samples to cached volumes (columns: `sample_name`, `image_cache_path`)
   - Volume directories: Each volume should be in a directory with `volume.mp4` and `metadata.json`

## Troubleshooting
### Common Issues

1. **"Command not found" errors**: Add `~/.local/bin` to your PATH or use module format
2. **HuggingFace authentication**: Run `huggingface-cli login` for gated models like MedGemma and MedImageInsight
3. **Memory issues**: Reduce batch size or use more GPUs for memory-intensive models
4. **Missing dependencies**: Some models may require additional packages (e.g., `flash-attn` for optimized attention)

# Citation
If you use this code in your research, please cite the following paper:

```
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  year    = {2025}
}
```
