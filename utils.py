import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.nn import functional as F
from torchvision import transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_signature_paths(data_dir="dataset/process"):
    writers = {}
    for folder, key in [("full_org", "org"), ("full_forg", "forg")]:
        if not (Path(data_dir) / folder).is_dir():
            raise FileNotFoundError("Run datapreprocessing.ipynb to create dataset/process first.")
        for path in sorted((Path(data_dir) / folder).iterdir()):
            parts = path.stem.split("_")
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            writer_id = int(parts[1])
            writers.setdefault(writer_id, {"org": [], "forg": []})[key].append(path)
    for writer_id, images in writers.items():
        if len(images["org"]) < 2 or not images["forg"]:
            raise ValueError(f"Writer {writer_id} needs two genuine images and one forged image.")
    return writers


def split_writers(writers, seed=42):
    ids = sorted(writers)
    random.Random(seed).shuffle(ids)
    train, validation, test = ids[:40], ids[40:48], ids[48:]
    if not train or not validation or not test:
        raise ValueError("The 40/8/rest split needs at least 49 writers.")
    return train, validation, test


def sample_triplets(writers, writer_ids, samples_per_writer, seed=42):
    rng = random.Random(seed)
    triplets = []
    for writer_id in writer_ids:
        for _ in range(samples_per_writer):
            a, p = rng.sample(writers[writer_id]["org"], 2)
            n = rng.choice(writers[writer_id]["forg"])
            triplets.append((a, p, n))
    rng.shuffle(triplets)
    return triplets


def make_transform(image_size, rgb=False):
    if rgb:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=False),
        ])
    return transform


def load_image(path, transform, rgb=False):
    with Image.open(path) as image:
        return transform(image.convert("RGB" if rgb else "L"))


def get_batches(triplets, image_size, batch_size=16, rgb=False):
    transform = make_transform(image_size, rgb)
    for start in range(0, len(triplets), batch_size):
        anchors, positives, negatives = [], [], []
        for a, p, n in triplets[start:start + batch_size]:
            for path, images in [(a, anchors), (p, positives), (n, negatives)]:
                images.append(load_image(path, transform, rgb))
        yield torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def get_embeddings(model, images, name):
    if name == "simple":
        features = model["features"](images).permute(0, 2, 3, 1).flatten(1)
        output = model["head"](features)
    else:
        output = model(images)
    return F.normalize(output, dim=1)


def triplet_loss(a, p, n, margin=0.2):
    positive = (a - p).square().sum(1)
    negative = (a - n).square().sum(1)
    return F.relu(positive - negative + margin).mean()


def run_epoch(model, name, triplets, config, device, optimizer=None, frozen=False):
    model.train(optimizer is not None)
    if frozen:
        model.backbone.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for a, p, n in get_batches(triplets, config["image_size"], config["batch_size"], name not in {"simple", "complex"}):
            images = torch.cat([a, p, n]).to(device)
            ea, ep, en = get_embeddings(model, images, name).chunk(3)
            loss = triplet_loss(ea, ep, en, config["margin"])
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(a)
    return total_loss / len(triplets)


@torch.no_grad()
def pair_distances(model, name, triplets, config, device):
    model.eval()
    distances, labels = [], []
    for a, p, n in get_batches(triplets, config["image_size"], config["batch_size"], name not in {"simple", "complex"}):
        images = torch.cat([a, p, n]).to(device)
        ea, ep, en = get_embeddings(model, images, name).chunk(3)
        distances.extend([(ea - ep).square().sum(1).cpu().numpy(),
                          (ea - en).square().sum(1).cpu().numpy()])
        labels.extend([np.ones(len(a)), np.zeros(len(a))])
    return np.concatenate(distances), np.concatenate(labels).astype(int)


def select_threshold(distances, labels):
    fpr, tpr, thresholds = roc_curve(labels, -distances, drop_intermediate=False)
    finite = np.flatnonzero(np.isfinite(thresholds))
    best = finite[np.argmax((tpr - fpr)[finite])]
    return float(-thresholds[best])


def calculate_metrics(distances, labels, threshold):
    prediction = (distances <= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, prediction),
        "balanced_accuracy": balanced_accuracy_score(labels, prediction),
        "precision": precision_score(labels, prediction, zero_division=0),
        "recall": recall_score(labels, prediction, zero_division=0),
        "f1": f1_score(labels, prediction, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auc": roc_auc_score(labels, -distances),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_test_pairs(writers, writer_ids):
    pairs = []
    for writer_id in writer_ids:
        genuine = writers[writer_id]["org"]
        forged = writers[writer_id]["forg"]
        for i, reference in enumerate(genuine):
            pairs.extend((reference, query, 1) for query in genuine[i + 1:])
            pairs.extend((reference, query, 0) for query in forged)
    return pairs


@torch.no_grad()
def test_pair_distances(model, name, pairs, config, device):
    model.eval()
    rgb = name not in {"simple", "complex"}
    transform = make_transform(config["image_size"], rgb)
    paths = sorted({path for reference, query, label in pairs for path in (reference, query)})
    embeddings = {}
    batch_size = config["batch_size"]
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        images = torch.stack([load_image(path, transform, rgb) for path in batch]).to(device)
        vectors = get_embeddings(model, images, name).cpu().numpy()
        embeddings.update(zip(batch, vectors))
    distances = [np.sum((embeddings[reference] - embeddings[query]) ** 2) for reference, query, label in pairs]
    labels = [label for reference, query, label in pairs]
    return np.array(distances), np.array(labels)


def calculate_triplet_metrics(distances, labels, margin=0.2):
    positive = distances[labels == 1]
    negative = distances[labels == 0]
    return {
        "triplet_loss": float(np.maximum(positive - negative + margin, 0).mean()),
        "triplet_accuracy": float(np.mean(positive < negative)),
        "margin_accuracy": float(np.mean(positive + margin <= negative)),
    }


def plot_history(history, name):
    if not history["train"]:
        return
    plt.plot(history["train"], label="Train")
    plt.plot(history["validation"], label="Validation")
    plt.title(name)
    plt.xlabel("Epoch")
    plt.ylabel("Triplet loss")
    plt.legend()
    plt.show()


def show_pairs(pairs, distances, labels, threshold, name, count=4):
    predictions = (distances <= threshold).astype(int)
    groups = [
        ("Correct genuine", (labels == 1) & (predictions == 1)),
        ("Correct fake", (labels == 0) & (predictions == 0)),
        ("Fake predicted genuine", (labels == 0) & (predictions == 1)),
        ("Genuine predicted fake", (labels == 1) & (predictions == 0)),
    ]
    for title, mask in groups:
        indices = np.flatnonzero(mask)[:count]
        print(f"{name}: {title} = {int(mask.sum())}")
        if not len(indices):
            continue
        fig, axes = plt.subplots(len(indices), 2, figsize=(10, 3 * len(indices)), squeeze=False)
        fig.suptitle(f"{name}: {title} predictions | Threshold: {threshold:.4f}")
        for row, index in enumerate(indices):
            for column, path in enumerate(pairs[index][:2]):
                with Image.open(path) as image:
                    axes[row, column].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
                axes[row, column].axis("off")
            actual = "Genuine" if labels[index] else "Fake"
            predicted = "Genuine" if predictions[index] else "Fake"
            axes[row, 0].set_title(f"Reference: {pairs[index][0].name}")
            axes[row, 1].set_title(f"{pairs[index][1].name}\nActual: {actual} | Predicted: {predicted}\nDistance: {distances[index]:.4f}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_distances(distances, labels, threshold, name):
    plt.hist(distances[labels == 1], bins=30, alpha=0.6, density=True, label="Genuine pairs")
    plt.hist(distances[labels == 0], bins=30, alpha=0.6, density=True, label="Fake pairs")
    plt.axvline(threshold, color="black", linestyle="--", label="Validation threshold")
    plt.title(name)
    plt.xlabel("Squared embedding distance")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
