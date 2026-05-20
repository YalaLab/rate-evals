# RATE-Evals Tutorial

A step-by-step walkthrough of feature extraction and evaluation using the
RATE-Evals pipeline. By the end you will know how to:

1. Extract vision embeddings from a model using the CLI
2. Use the Python API to extract features directly
3. Run disease-finding evaluation on those embeddings
4. Add a custom dataset of your own

## Prerequisites

- Python 3.8+
- GPU recommended (CPU works but is slower)
- Install the project:

```bash
uv sync
```

> **Note**: You do **not** need the `rad-vision-engine` (`rve`) package for
> this tutorial. The `DummyDataset` used below is self-contained.
> If you hit an `rve` import error elsewhere, see
> [Troubleshooting](#troubleshooting).

## Setup

Generate the synthetic image and labels used throughout the tutorial:

```bash
python tutorial/setup_tutorial.py
```

This creates two files:

| File | Purpose |
|------|---------|
| `assets/CXR145_IM-0290-1001.png` | 1024x1024 grayscale synthetic chest X-ray (required by `DummyDataset`) |
| `tutorial/dummy_labels.json` | Labels for 100 dummy studies with 3 binary findings |

---

## Part 1: Extract Features (CLI)

Use `rate-extract` to compute embeddings for the dummy dataset:

```bash
# Extract training split
uv run rate-extract \
    --model pillar0 \
    --dataset dummy \
    --split train \
    --batch-size 4 \
    --output-dir cache/pillar0_dummy \
    --max-samples 100
```

**Flags explained:**

| Flag | Meaning |
|------|---------|
| `--model pillar0` | Use the Pillar-0 vision foundation model |
| `--dataset dummy` | Use the built-in `DummyDataset` (reads the synthetic image) |
| `--split train` | Process the training split |
| `--batch-size 4` | Images per GPU batch (lower this if you hit OOM) |
| `--output-dir cache/pillar0_dummy` | Where to save embedding `.npz` files |
| `--max-samples 100` | Limit to 100 samples (matches our labels) |

Now extract the test split too:

```bash
uv run rate-extract \
    --model pillar0 \
    --dataset dummy \
    --split test \
    --batch-size 4 \
    --output-dir cache/pillar0_dummy \
    --max-samples 100
```

**Verify the output:**

```bash
ls cache/pillar0_dummy/train/
ls cache/pillar0_dummy/test/
# You should see .npz files containing the cached embeddings.
```

---

## Part 2: Extract Features (Python API)

For programmatic access you can call the model directly:

```python
from rate_eval import create_model, create_dataset, setup_pipeline

# Load configuration
config = setup_pipeline()

# Create model and dataset
model = create_model("pillar0", config)
dataset = create_dataset("dummy", config, split="train")

print(f"Dataset size : {len(dataset)}")
print(f"Accession [0]: {dataset.get_accession(0)}")

# Get a single sample (two-view chest X-ray tensor)
sample = dataset[0]
print(f"Sample shape : {sample.shape}")
# -> torch.Size([1, 2, 1024, 1024])  (C, views, H, W)

# Extract features (must pass modality matching the dataset)
features = model.extract_features(sample.unsqueeze(0), modality="chest_xray_two_view")
print(f"Embedding dim: {features.shape}")
```

This is useful when you want to integrate RATE-Evals into a larger pipeline or
inspect intermediate representations. It also shows how the model converts an
image into a fixed-size embedding vector.

---

## Part 3: Run Evaluation

Once you have cached embeddings for both `train` and `test` splits, run
evaluation:

```bash
uv run rate-evaluate \
    --checkpoint-dir cache/pillar0_dummy \
    --dataset-name dummy \
    --labels-json tutorial/dummy_labels.json \
    --output-dir results/pillar0_dummy \
    evaluation.use_wandb=false
```

> `evaluation.use_wandb=false` is a Hydra-style override (no `--` prefix).
> It disables Weights & Biases logging so you can run offline.

**What gets produced:**

| File | Contents |
|------|----------|
| `results/pillar0_dummy/detailed_results.csv` | Per-question metrics (AUC, F1, accuracy, etc.) |
| `results/pillar0_dummy/summary_stats.json` | Aggregated metrics across all findings |
| `results/pillar0_dummy/training_stats.json` | Class distribution statistics from training |
| `results/pillar0_dummy/exam_probabilities.csv` | Per-exam predicted probabilities |

**Inspect the results:**

```python
import json, pandas as pd

summary = json.load(open("results/pillar0_dummy/summary_stats.json"))
print(f"Average AUC: {summary['avg_auc']:.3f}")

detailed = pd.read_csv("results/pillar0_dummy/detailed_results.csv")
print(detailed[["question", "auc", "f1", "num_positive", "num_negative"]])
```

> **Note**: Because the dummy dataset repeats the same synthetic image, the
> embeddings carry no discriminative signal and metrics will be near-random.
> With real data, expect meaningful AUC scores.

---

## Part 4: Add a Custom Dataset

To evaluate on your own data, follow these four steps:

### Step A: Create a Dataset Class

Your class must implement the following interface:

```python
class MyDataset:
    def __init__(self, config, split="train", transforms=None, model_preprocess=None):
        ...

    def __getitem__(self, idx):
        """Return a tensor of shape (C, views, H, W) or (D, H, W) for 3-D volumes."""
        ...

    def __len__(self):
        """Total number of samples."""
        ...

    def get_accession(self, idx):
        """Return a unique string identifier (accession) for sample idx."""
        ...

    def get_all_accessions(self):
        """Return a list of all accession strings (fast, no data loading)."""
        ...
```

Place the file in `rate_eval/datasets/` and import it in
`rate_eval/datasets/__init__.py`.

### Step B: Create a Config YAML

Create `configs/dataset/my_dataset.yaml`:

```yaml
# @package dataset
name: my_dataset
modality: chest_xray_two_view   # or abdomen_ct, chest_ct, brain_ct, etc.

data:
  root_dir: /path/to/images

processing:
  target_size: [1024, 1024]
  image_mode: L                 # L=grayscale, RGB=colour

labels:
  labels_json: /path/to/labels.json
```

### Step C: Register in Config

Add an entry to `configs/config.yaml` under the `datasets:` section:

```yaml
datasets:
  # ... existing entries ...
  my_dataset:
    class: MyDataset
    config: configs/dataset/my_dataset.yaml
```

### Step D: Prepare Labels JSON

Create a JSON file where each key is an accession string and the value contains
`qa_results` with lists of question-answer dicts:

```json
{
  "ACCESSION_001": {
    "qa_results": {
      "findings": [
        {"Is there evidence of cardiomegaly?": "yes"},
        {"Is there a pleural effusion?": "no"}
      ]
    }
  }
}
```

Answers should be `"yes"` or `"no"` (case-insensitive). The evaluator
discovers all unique questions automatically.

### Run It

```bash
uv run rate-extract --model pillar0 --dataset my_dataset --all-splits \
    --batch-size 4 --output-dir cache/pillar0_my_dataset

uv run rate-evaluate --checkpoint-dir cache/pillar0_my_dataset \
    --dataset-name my_dataset \
    --labels-json /path/to/labels.json \
    --output-dir results/pillar0_my_dataset \
    evaluation.use_wandb=false
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **`ModuleNotFoundError: rve`** | The `rve` (rad-vision-engine) package is only needed for RVE-format datasets. Use `DummyDataset` or NIfTI-based datasets instead, or install it: `git clone https://github.com/yalalab/rad-vision-engine ../rad-vision-engine && uv pip install -e ../rad-vision-engine` |
| **Out of memory (OOM)** | Reduce `--batch-size` (try 1 or 2) or use `--device cpu` |
| **NaN embeddings** | Add `--check-nan` to `rate-extract` for detailed diagnostics |
| **"Command not found"** | Add `~/.local/bin` to your `PATH` or use `uv run rate-extract` / `uv run rate-evaluate` |
| **HuggingFace auth errors** | Run `huggingface-cli login` for gated models (MedGemma, MedImageInsight) |
