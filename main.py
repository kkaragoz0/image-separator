import os
import shutil
import csv
import math
import argparse
import torch
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed. HEIC/HEIF files will be skipped. Run: pip install pillow-heif")

MODEL_ID  = "openai/clip-vit-large-patch14-336"
BATCH_SIZE = 64

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".3gp"}

CATEGORIES = {
    "Screenshots": [
        "a mobile phone screenshot",
        "a user interface of an app",
        "a screenshot of a website",
        "a digital screen with text and icons",
        "a screenshot of social media comments or replies",
        "a screenshot of a chat or comment thread"
    ],
    "Whiteboards": [
        "a photo of a whiteboard with handwriting",
        "technical diagrams on a whiteboard",
        "a flipchart with markers",
        "a chalkboard with equations"
    ],
    "Documents": [
        "a close-up photo of a printed document",
        "a paper receipt",
        "a scan of a book page",
        "a formal letter on a table"
    ],
    "Art_Design": [
        "a classical oil painting",
        "a digital art illustration",
        "a pencil sketch or drawing",
        "a colorful graphic design piece",
        "museum art",
        "an anime or manga style character illustration",
        "a cartoon or chibi character drawing"
    ],
    "Memes_Infographics": [
        "a meme with bold text on an image",
        "an infographic with statistics and icons",
        "a motivational quote over a background photo",
        "a viral social media image with caption",
        "a funny animal photo with humorous text caption",
        "an image with subtitle-style text at the bottom"
    ],
    "Food": [
        "a close-up photo of a meal or dish",
        "a restaurant food photography shot",
        "a flat lay of ingredients on a table",
        "a drink or coffee in a cafe"
    ],
    "Camera_Photos": [
        "a portrait of a person",
        "a group of people posing for a photo",
        "a selfie",
        "a natural landscape photo",
        "a photo of a pet at home with people",
        "a photo of a street or building"
    ],
}


def load_model(model_id: str, device: str):
    print(f"Loading {model_id} on {device}...")
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def build_category_centroids(model, processor, device) -> torch.Tensor:
    """
    Pre-compute one normalized centroid embedding per category.

    Centroid = mean of all prompt embeddings, re-normalized onto the unit sphere.
    This is the least-squares optimal representative direction in CLIP embedding space —
    better than picking the single max-logit prompt at inference time.

    Returns shape: [num_categories, embedding_dim]. Computed once, reused every batch.
    """
    print("Pre-computing category centroids...")
    centroids = []
    with torch.no_grad():
        for prompts in CATEGORIES.values():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            text_features = model.text_projection(model.text_model(**text_inputs).pooler_output)  # [num_prompts, dim]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            centroid = text_features.mean(dim=0)
            centroid = centroid / centroid.norm()                         # re-normalize after averaging
            centroids.append(centroid)
    return torch.stack(centroids)                                         # [num_categories, dim]


def classify_batch(
    images: list,
    model,
    processor,
    category_centroids: torch.Tensor,
    device,
) -> list[tuple[str, float, float]]:
    """
    Classify a batch of images.

    - Runs only the image encoder (text encoder ran once at startup).
    - Scores via cosine similarity against category centroids scaled by CLIP's learned temperature.
    - Uncertainty measured as normalized Shannon entropy H / log(N):
        0.0 = completely certain, 1.0 = uniform over all categories.

    Returns list of (predicted_folder, top_category_prob, normalized_entropy).
    """
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs["pixel_values"] = image_inputs["pixel_values"].to(device, dtype=torch.float16)
    with torch.no_grad():
        image_features = model.visual_projection(model.vision_model(**image_inputs).pooler_output)  # [batch, dim]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (image_features @ category_centroids.T)        # [batch, num_categories]
    probs = torch.softmax(logits, dim=-1)

    num_categories = probs.shape[1]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    normalized_entropy = entropy / math.log(num_categories)               # [batch]

    category_names = list(CATEGORIES.keys())
    results = []
    for i in range(len(images)):
        best_cat = probs[i].argmax().item()
        results.append((
            category_names[best_cat],
            round(probs[i][best_cat].item(), 3),
            round(normalized_entropy[i].item(), 3),
        ))
    return results


def _collect_files(source_dir: str, extensions: set) -> list[tuple[str, str]]:
    """
    Recursively collect all files matching extensions under source_dir.
    Returns list of (full_source_path, dest_filename).

    Files at root level keep their original name.
    Files in subfolders are prefixed with their relative path (sep → _) to avoid
    collisions between files with the same name in different subdirectories.
    Any remaining collision gets a numeric suffix.
    """
    used: set[str] = set()
    result = []

    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in extensions:
                continue

            full_path = os.path.join(dirpath, filename)
            rel_dir   = os.path.relpath(dirpath, source_dir)
            dest_name = filename if rel_dir == "." else f"{rel_dir.replace(os.sep, '_')}_{filename}"

            if dest_name in used:
                base, ext = os.path.splitext(dest_name)
                i = 2
                while f"{base}_{i}{ext}" in used:
                    i += 1
                dest_name = f"{base}_{i}{ext}"

            used.add(dest_name)
            result.append((full_path, dest_name))

    return result


def move_videos(source_dir: str, target_dir: str):
    video_dir = os.path.join(target_dir, "Videos")
    os.makedirs(video_dir, exist_ok=True)

    moved = 0
    for full_path, dest_name in _collect_files(source_dir, VIDEO_EXTENSIONS):
        shutil.move(full_path, os.path.join(video_dir, dest_name))
        moved += 1

    print(f"Moved {moved} video(s) to {video_dir}")


def run_classification(
    source_dir: str,
    target_dir: str,
    entropy_threshold: float,
    model,
    processor,
    category_centroids: torch.Tensor,
    device,
):
    for folder in list(CATEGORIES.keys()) + ["Unclassified"]:
        os.makedirs(os.path.join(target_dir, folder), exist_ok=True)

    all_files = _collect_files(source_dir, IMAGE_EXTENSIONS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(target_dir, f"classification_log_{timestamp}.csv")
    counts = {folder: 0 for folder in list(CATEGORIES.keys()) + ["Unclassified"]}

    with open(log_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "assigned_folder", "top_confidence", "normalized_entropy"])

        for i in tqdm(range(0, len(all_files), BATCH_SIZE)):
            batch = all_files[i: i + BATCH_SIZE]
            images, valid_files = [], []

            for full_path, dest_name in batch:
                try:
                    img = Image.open(full_path).convert("RGB")
                    images.append(img)
                    valid_files.append((full_path, dest_name))
                except Exception as e:
                    print(f"Error opening {full_path}: {e}")

            if not images:
                continue

            try:
                results = classify_batch(images, model, processor, category_centroids, device)
            except Exception as e:
                print(f"Error classifying batch: {e}")
                continue
            finally:
                for img in images:
                    img.close()

            for (full_path, dest_name), (folder, conf, entropy) in zip(valid_files, results):
                writer.writerow([dest_name, folder, conf, entropy])

                if entropy <= entropy_threshold:
                    shutil.move(full_path, os.path.join(target_dir, folder, dest_name))
                    counts[folder] += 1
                else:
                    shutil.move(full_path, os.path.join(target_dir, "Unclassified", f"{entropy}_{dest_name}"))
                    counts["Unclassified"] += 1

    print("\n--- Classification Summary ---")
    total = sum(counts.values())
    for folder, count in counts.items():
        pct = (count / total * 100) if total else 0
        print(f"  {folder:<22} {count:>4}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<22} {total:>4}")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort images into categories using CLIP.")
    parser.add_argument("--source",    required=True,        help="Folder containing unsorted images/videos.")
    parser.add_argument("--target",    default="./sorted_output", help="Output folder (default: ./sorted_output).")
    parser.add_argument("--threshold", type=float, default=0.80,  help="Entropy threshold 0–1 (default: 0.80). Lower = stricter.")
    parser.add_argument("--batch-size", type=int,  default=BATCH_SIZE, help=f"Images per batch (default: {BATCH_SIZE}).")
    args = parser.parse_args()

    move_videos(args.source, args.target)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model, processor = load_model(MODEL_ID, device)

    if device == "cuda":
        model = torch.compile(model)

    category_centroids = build_category_centroids(model, processor, device)
    run_classification(args.source, args.target, args.threshold, model, processor, category_centroids, device)
