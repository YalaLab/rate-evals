# RATE Evals

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
cd ../rad-vision-engine && git checkout release
cd ../rate_evals
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
    --model-repo-id YalaLab/Pillar0-Merlin \
    --model-revision epoch_24 \
    --ct-window-type all \
    --modality abdomen_ct

# Evaluate the model
uv run rate-evaluate \
    --checkpoint-dir cache/pillar0_abd_ct_merlin \
    --dataset-name abd_ct_merlin \
    --labels-json data/merlin/final_results.json \
    --output-dir results/pillar0_abd_ct_merlin
```

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
