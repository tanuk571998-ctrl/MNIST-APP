# Handwritten Digit Data Collection App (Module 1)

A Streamlit tool for drawing handwritten digits and converting them into
an MNIST-compatible labeled dataset (`handwritten_digits_dataset.csv`),
ready to feed into a neural network training pipeline (Module 2).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — canvas, controls, results display, analytics |
| `image_processing.py` | Pure image-processing functions (grayscale → crop → center → resize → smooth → flatten) |
| `dataset_manager.py` | Validation, CSV creation/append, dataset statistics |
| `requirements.txt` | Python dependencies |

## 1. Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Running the app

```bash
streamlit run app.py
```

This opens the app in your browser (default `http://localhost:8501`).

## 3. Using the app

### Manual mode (default)
1. Pick the digit label (0–9) in the sidebar.
2. Draw the digit on the black 400×400 canvas using white strokes. A live
   preview (original + processed 28×28 image, pixel summary, flattened
   vector preview) appears automatically as you draw — no "Process" step
   needed.
3. Click **Save Sample** to append the validated `(label, 784 pixels)` row
   to `handwritten_digits_dataset.csv`. The canvas clears automatically.
4. Use the small toolbar above the canvas (Undo / Redo / Clear / Download)
   to fix mistakes at any time.

### 🤖 Automatic mode (hands-free)
Turn on **"Auto-save when I pause drawing"** in the sidebar and the app
runs the whole loop for you:

1. You draw a digit.
2. About 2–3 seconds after you stop moving the pen, the app detects the
   pause, processes the drawing, validates it, and saves it — no click
   required.
3. The canvas clears itself automatically, ready for the next digit.
4. If **"Auto-advance digit label after each save"** is also on (default),
   the label cycles 0 → 1 → 2 → … → 9 → 0 automatically, so you can just
   keep drawing digits in order without touching the sidebar.

This is the fastest way to collect a large, balanced dataset: draw digit
after digit and let the app save, clear, and re-label itself between each
one. The manual **Save Sample** button still works even with Automatic
mode on, if you want to save early.

The sidebar's **Dataset Statistics** panel (total count + per-digit
counts, as a bar chart and table) refreshes automatically after every
successful save, whether manual or automatic.

## 4. Preprocessing workflow explanation

The pipeline in `image_processing.py` mirrors how the original MNIST
dataset was constructed, so downstream models trained on it transfer well
to real MNIST-style data:

1. **Grayscale conversion** — canvas RGBA → single channel. Because the
   canvas uses a black background with white strokes, intensity already
   matches MNIST's convention (0 = background, 255 = digit).
2. **Bounding-box detection** — locate the tight rectangle around all
   non-background pixels; raises a clear error if the canvas is empty.
3. **Crop** — remove all empty margin outside that bounding box.
4. **Aspect-preserving resize into a 20×20 box** — the digit is scaled so
   its longer side fits in 20 pixels, avoiding distortion.
5. **Center in a 28×28 frame** — the resized digit is pasted into the
   middle of a black 28×28 canvas, leaving a small border like real MNIST.
6. **Anti-aliasing** — a light Gaussian blur smooths jagged edges left by
   freehand drawing and scaling.
7. **Pixel clipping** — values are clamped to the valid `0–255` `uint8` range.
8. **Flattening** — the 28×28 array is flattened in row-major (`C`) order
   into a 784-element vector, matching the standard MNIST CSV pixel order.

## 5. Dataset generation explanation

`dataset_manager.py` owns all file I/O:

- **On first save**, if `handwritten_digits_dataset.csv` doesn't exist,
  it's created with the header `label,pixel_1,...,pixel_784`.
- **On every subsequent save**, exactly one new row is appended — existing
  rows are never rewritten or overwritten.
- **Validation before every write**:
  - Label is present, numeric, and in `0–9`.
  - Image is exactly 28×28 (784 pixels).
  - Row totals exactly 785 columns (1 label + 784 pixels).
  - Any failure raises `DatasetValidationError` with a descriptive
    message, shown to the user instead of writing bad data.
- **Statistics** are computed live by reading just the `label` column and
  counting occurrences per digit — cheap even as the dataset grows.

## 6. MNIST compatibility explanation

Each saved row satisfies every MNIST-standard requirement from the spec:

- 28×28, single-channel grayscale
- Pixel values in `0–255`
- Digit centered via bounding-box + aspect-preserving resize + padding
- Anti-aliased edges (Gaussian smoothing)
- Flattened to exactly 784 values in row-major order
- CSV layout (`label` + `pixel_1..pixel_784`, 785 columns total) matches
  the common Kaggle/MNIST-CSV convention, so it can be loaded directly
  with `pandas.read_csv()` and split into `X = df.iloc[:, 1:].values`,
  `y = df.iloc[:, 0].values` for training.

## 7. Recommendations for future neural network integration (Module 2)

- **Normalization**: divide pixel columns by 255.0 (`X = X / 255.0`) before
  feeding into a model; consider standardization if using architectures
  that expect zero-centered input.
- **Reshaping**: reshape each row's 784 values back to `(28, 28, 1)` for
  CNN input, or keep flattened for a simple MLP baseline.
- **Train/validation split**: use `sklearn.model_selection.train_test_split`
  with `stratify=y` since class balance may be uneven early on (check the
  sidebar stats panel).
- **Data augmentation**: since hand-drawn samples from a single user may be
  visually similar, consider light augmentation (small rotations, shifts,
  elastic distortions) to improve generalization — a well-known technique
  from MNIST-style training.
- **Incremental training**: because the CSV grows via appends, Module 2
  can support periodic retraining as new samples accumulate, or an
  online/incremental learning loop.
- **Model architecture**: a small CNN (2–3 conv layers + pooling + dense
  head) is a good default for 28×28 grayscale digit classification and
  will vastly outperform a plain MLP with limited data.
- **Data quality checks**: before training, consider deduplicating rows,
  checking per-class balance from the same stats panel, and visually
  spot-checking a random sample grid to catch mislabeled entries early.
