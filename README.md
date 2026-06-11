# Image Separator

Zero-shot image classifier that automatically sorts a mixed folder of images and videos into labelled subfolders using OpenAI's CLIP model — no training data required.

## How it works

### Model
Uses **CLIP ViT-L/14 @ 336px** (`openai/clip-vit-large-patch14-336`), a vision-language model that maps images and text into a shared embedding space. Classification is done entirely through natural language — no fine-tuning needed.

### Prompt centroid ensembling
Each category is described by multiple text prompts (e.g. `"a mobile phone screenshot"`, `"a screenshot of a website"`). Rather than picking the single best-matching prompt, all prompts for a category are encoded and **averaged into a centroid vector** on the unit sphere:

```
centroid_k = normalize( mean{ normalize(embed(p)) : p ∈ category_k } )
```

This is the least-squares optimal representative direction in CLIP's embedding space, more stable than any single prompt.

### Scoring
For each image, cosine similarity is computed against every category centroid, then scaled by CLIP's learned temperature and passed through softmax to produce a probability distribution over categories.

### Entropy-based uncertainty
Instead of a raw confidence threshold, classification confidence is measured using **normalized Shannon entropy**:

```
H = -Σ p_i · log(p_i) / log(N)     (N = number of categories)
```

- `H = 0.0` → model is completely certain
- `H = 1.0` → model is uniformly uncertain across all categories

Images above the threshold go to `Unclassified/` rather than being forced into a wrong category.

### Categories

| Folder | What it captures |
|---|---|
| `Screenshots` | Mobile/web UI, social media comments, chat threads |
| `Documents` | Printed documents, receipts, scanned pages |
| `Whiteboards` | Whiteboards, chalkboards, flipcharts |
| `Art_Design` | Paintings, illustrations, anime, digital art |
| `Memes_Infographics` | Memes, infographics, motivational quotes |
| `Food` | Meals, restaurant shots, drinks |
| `Camera_Photos` | People, landscapes, selfies, streets |
| `Videos` | All video files (moved, not classified by CLIP) |
| `Unclassified` | Images the model was uncertain about |

### Nested folders

The source directory is scanned **recursively** — all subdirectories are included automatically. Files from subfolders are renamed at the destination to avoid collisions:

| Source path | Destination filename |
|---|---|
| `source/photo.jpg` | `photo.jpg` |
| `source/2024/photo.jpg` | `2024_photo.jpg` |
| `source/2024/Paris/photo.jpg` | `2024_Paris_photo.jpg` |

If two paths still produce the same destination name after prefixing, a numeric suffix is appended (`_2`, `_3`, …). The CSV log always records the final destination filename so the original location can be traced.

### Supported formats

**Images:** `.jpg` `.jpeg` `.png` `.webp` `.heic` `.heif` `.gif` `.bmp` `.tiff`

**Videos:** `.mp4` `.mov` `.avi` `.mkv` `.wmv` `.flv` `.webm` `.m4v` `.3gp`

> HEIC/HEIF support (iPhone default format) requires `pillow-heif`. See installation below.

---

## Installation

```bash
# 1. Install PyTorch with CUDA (GPU support — recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install remaining dependencies
pip install -r requirements.txt
```

CPU-only fallback works automatically if no CUDA GPU is available, but will be significantly slower.

---

## Usage

```bash
# Minimal
python main.py --source "/path/to/mixed/folder"

# With all options
python main.py --source "/path/to/mixed/folder" \
               --target "./sorted_output" \
               --threshold 0.80 \
               --batch-size 64
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--source` | Yes | — | Folder containing unsorted images and videos |
| `--target` | No | `./sorted_output` | Output folder |
| `--threshold` | No | `0.80` | Entropy threshold (0–1). Lower = stricter classification |
| `--batch-size` | No | `64` | Images per GPU batch |

### Tuning `--threshold`

| Value | Behaviour |
|---|---|
| `0.70` | Strict — more images go to `Unclassified`, fewer misclassifications |
| `0.80` | Balanced (default) |
| `0.90` | Lenient — fewer images go to `Unclassified`, higher risk of wrong folder |

After a run, inspect `Unclassified/` — filenames are prefixed with their entropy score (e.g. `0.84_photo.jpg`) to help you decide whether to raise or lower the threshold.

---

## Output

```
sorted_output/
├── Screenshots/
├── Documents/
├── Whiteboards/
├── Art_Design/
├── Memes_Infographics/
├── Food/
├── Camera_Photos/
├── Videos/
├── Unclassified/
└── classification_log_20260611_143022.csv
```

Each run produces a timestamped CSV log with columns: `filename`, `assigned_folder`, `top_confidence`, `normalized_entropy`. Previous logs are never overwritten.

---

## Adding or modifying categories

Edit the `CATEGORIES` dict in `main.py`. Each key is a folder name, each value is a list of descriptive prompts:

```python
"Receipts": [
    "a photo of a paper receipt",
    "a thermal printed receipt from a store",
],
```

More prompts per category improve robustness. There is no limit, but avoid redundant prompts — diversity of description matters more than quantity.
