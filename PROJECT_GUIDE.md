# Signature Verification Project

This project compares a known genuine signature with another signature. The model produces one embedding vector for each image. Small squared Euclidean distance means `Real`; distance above a validation threshold means `Fake`.

## Files

| File | Purpose |
|---|---|
| `datapreprocessing.ipynb` | Apply Otsu to original images and save them under `dataset/process`. |
| `simple.ipynb` | Train or load the simple CNN. |
| `complex.ipynb` | Train or load the deeper CNN. |
| `pretrained.ipynb` | Train or load ResNet18, MobileNet V3 Small, EfficientNet B0 and DenseNet121. |
| `test.ipynb` | Load every complete saved model, calibrate validation thresholds and evaluate random test pairs. |
| `verify_signature.py` | Select a saved model and compare two user images. |
| `utils.py` | Shared data loading, embeddings, triplet loss, training, metrics and image display functions. |
| `requirements.txt` | Python dependencies. |
| `model/*_model.pt` | Complete models with their configuration and training history. |
| `model/*_threshold.json` | Validation thresholds and image sizes. |
| `model/test_results.csv` | Latest test metrics and sampling seeds. |

## Data and Otsu

Original genuine signatures are in `dataset/full_org`; forgeries are in `dataset/full_forg`. `datapreprocessing.ipynb` converts images to grayscale, applies `cv2.THRESH_BINARY + cv2.THRESH_OTSU`, and saves PNG images with the same stems under `dataset/process/full_org` and `dataset/process/full_forg`.

Otsu chooses a grayscale cutoff automatically. Pixels become black or white. This reduces differences in ink darkness and background brightness. Shadows and ruled paper can remain, so use clear signature crops when supplying new images.

`normalize_signature(image)` applies this operation in the preprocessing notebook. The other notebooks read the processed images directly. `prepare_image(path)` applies the same Otsu operation to user images in verification, with EXIF orientation handling and a white background for transparency.

## Data functions in utils.py

- `set_seed(seed)`: seeds Python, NumPy and PyTorch for reproducibility.
- `get_signature_paths(data_dir)`: groups image paths by writer, with `org` and `forg` lists. Writer IDs come from the filenames.
- `split_writers(writers, seed)`: shuffles writer IDs and returns 40 training writers, 8 validation writers and the remaining 7 test writers for this dataset. A writer belongs to one split only.
- `sample_triplets(writers, writer_ids, samples_per_writer, seed)`: samples an anchor and a different genuine signature from the same writer, plus a forgery of that writer.
- `make_transform(image_size, rgb)`: prepares image tensors. Custom CNNs use grayscale tensors resized without antialiasing. Pretrained CNNs use RGB resizing and ImageNet normalization.
- `load_image(path, transform, rgb)`: opens a processed image and applies the transform.
- `get_batches(triplets, image_size, batch_size, rgb)`: yields anchor, positive and negative batches without a custom Dataset class.

## Siamese embeddings and triplet loss

There is one network with shared parameters. Anchor, positive and negative images are concatenated for a forward pass, then their embeddings are separated. The architecture does not contain three independently trained models.

`get_embeddings(model, images, name)` returns L2-normalized vectors. The simple model stores `features` and `head` in a `ModuleDict`, so this helper performs its forward pass explicitly. Its feature flattening order is height, width, channels. The other models support `model(images)` directly.

```python
vectors = get_embeddings(model, images, name)
distance = (vectors[0] - vectors[1]).square().sum()
```

`triplet_loss(a, p, n, margin)` computes:

```python
positive = (a - p).square().sum(1)
negative = (a - n).square().sum(1)
loss = torch.relu(positive - negative + margin).mean()
```

The loss encourages genuine signatures to be closer to their reference than forged signatures by at least the margin. The margin is a training parameter, not the prediction threshold.

`run_epoch(model, name, triplets, config, device, optimizer, frozen)` calculates this loss over batches. With an optimizer, it runs backpropagation and updates parameters. Without an optimizer, it evaluates validation loss with gradients disabled. While the pretrained backbone is frozen, its BatchNorm layers stay in evaluation mode.

## Architectures

Architectures live in the training notebooks. They are constructed only when training a new model. Existing complete models are loaded directly.

| Model | Input | Main layers | Embedding size |
|---|---|---|---:|
| Simple | Grayscale, 155 x 220 | Conv 16, 32, 64; ReLU and MaxPool; Linear head | 64 |
| Complex | Grayscale, 155 x 220 | Five two-convolution blocks, channels 16 through 256; BatchNorm, pooling, dropout; global pooling and head | 128 |
| Pretrained models | RGB, 224 x 224 | ImageNet backbone with its original classifier removed; new 256-unit head | 128 |

`build_simple` and `build_complex` create the custom networks. `conv_block` creates one repeated complex block. `build_pretrained` replaces the original backbone classifier with `Identity` and adds the signature embedding head.

Transfer learning initializes a new backbone from ImageNet. The head starts randomly initialized. The first two epochs freeze backbone parameters; later epochs fine-tune them. Adam uses learning rate `1e-5` for the backbone and `1e-4` for the head. Loading an already trained signature model does not initialize or download an ImageNet backbone again.

## Training and saving complete models

Each training notebook has one short title and four code cells: setup, configuration, architecture definitions, and load-or-train.

`FORCE_RETRAIN = False` loads an existing `*_model.pt` and skips training. Set it to `True` to train a new model. The default maximum is 10 epochs, with triplet margin 0.2 and batch size 16 triplets.

Training samples fresh triplets each epoch and uses fixed validation triplets. It keeps a full copy of the model with the lowest validation loss. After three consecutive epochs without improvement, training stops. Pretrained training resets the best validation loss when fine-tuning begins, so the saved model is selected from that phase.

```python
best_model = copy.deepcopy(model).cpu()
torch.save({"name": name, "model": best_model.eval(), "config": CONFIG, "history": history}, path)
```

The saved dictionary contains:

- `name`: model name.
- `model`: the complete network object and its learned parameters.
- `config`: preprocessing, image size, embedding size, writer splits and training settings.
- `history`: training and validation loss lists.

```python
saved = torch.load(path, map_location=DEVICE, weights_only=False)
model = saved["model"].eval()
```

`weights_only=False` allows loading the complete model object. The project uses its own saved files. There are no separate weight checkpoint files, no calls to `load_state_dict`, and no architecture rebuilding when reusing a trained model. Existing models retain their original learned parameters and history.

`plot_history(history, name)` plots the saved training and validation losses. Reusing a model skips optimization; this is not optimizer-state resume training.

## Testing

`test.ipynb` has one title and three code cells. It loads all `model/*_model.pt` files directly. Validation triplets use the saved `val_w` split; test images use only `test_w`.

`pair_distances` produces genuine and forged distances from validation triplets. `select_threshold` uses the ROC curve to maximize true-positive rate minus false-positive rate on validation. Threshold selection does not use test labels.

`make_test_pairs` creates genuine-genuine and genuine-forged pairs for each test writer. For each model, the notebook samples 300 unique pairs from each class and shuffles them using `SEED + i`. Models have different sample sets, which may overlap. The CSV records each seed and sample size. Change `PAIRS_PER_CLASS` to change the number of pairs.

`test_pair_distances` caches one embedding per distinct image, then calculates each pair distance from these vectors.

`calculate_metrics` treats genuine as label 1 and fake as label 0:

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of correct pair predictions. |
| Balanced accuracy | Mean of genuine recall and fake detection rate. |
| Precision | Fraction of predicted genuine signatures that are actually genuine. |
| Recall | Fraction of genuine signatures accepted. |
| F1 | Harmonic mean of genuine precision and recall. |
| Specificity | Fraction of fake signatures rejected. |
| ROC AUC | Ranking quality using negative distance as the genuine score. |
| TN | Fake correctly rejected. |
| FP | Fake incorrectly accepted. |
| FN | Genuine incorrectly rejected. |
| TP | Genuine correctly accepted. |

`show_pairs` displays correctly accepted genuine pairs, correctly rejected fake pairs, fake signatures incorrectly accepted, and genuine signatures incorrectly rejected. The test notebook displays up to two examples per group for each model.

Testing writes thresholds and `test_results.csv`. Different random samples introduce sampling variation, so model scores are less directly comparable than scores on the same test set. The current samples are balanced; predicting fake for every pair gives 50% accuracy.

## Comparing user images

Run from the project folder:

```bash
python verify_signature.py
```

`choose(items)` shows a menu for saved models or reference images. Enter a number or name. The query image path is requested when `--query` is omitted.

```bash
python verify_signature.py --model complex --reference "Signatures/Dang/dang_riel.jpg" --query "Signatures/Dang/dang_fake.jpg"
python verify_signature.py --model simple --reference-dir "dang_riel" --query "user_signature.jpg"
python verify_signature.py --model densenet121 --query "user_signature.jpg" --output "comparison.png" --no-show
```

Available models are `simple`, `complex`, `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`, and `densenet121`. Without an explicit reference, the script uses the `dang_riel` folder if present, otherwise `Signatures/Dang/dang_riel.jpg`.

`compare_signatures` loads the selected complete model and threshold, preprocesses both images, computes normalized embeddings and compares their squared distance. It returns the prediction and processed images. `main` handles inputs, prints the result and displays the pair. `--output` saves the figure; `--no-show` skips the figure window.

The reference must be a known genuine signature of the person being checked. Prediction is `Real` when distance is at or below the saved threshold and `Fake` otherwise. A distance is not a probability or confidence percentage.

## Running locally or on Colab

```bash
pip install -r requirements.txt
```

Run preprocessing first, then the desired training notebooks, then `test.ipynb`. After training again, run test again to calibrate thresholds for the new models before comparing user signatures.

For Colab, upload `pretrained.ipynb`, `utils.py`, and `dataset/process`, then select a GPU runtime. Run from the folder containing these files. Set `MODEL_DIR` to a mounted Google Drive directory to keep saved models after the runtime ends:

```python
from google.colab import drive
drive.mount("/content/drive")
MODEL_DIR = Path("/content/drive/MyDrive/sign-verification/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
```

Download the complete `*_model.pt` files to the local `model` folder, then run the local test notebook. If GPU memory is insufficient, reduce batch size.
