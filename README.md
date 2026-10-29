# Brain Tumor MRI Classifier (PyTorch + Streamlit)

Classifies brain MRI slices into **glioma / meningioma / pituitary tumor / no tumor** using transfer learning
(EfficientNet-B0 by default), explains predictions with **Grad-CAM** heat-maps, and ships a **Streamlit dashboard**
with results, image upload, camera snapshot and a **live feed** (simulated scanner stream or real webcam video).

> Educational project only. It is **not a medical device** and must not be used for diagnosis.
> Models trained on Kaggle data often generalise poorly to scans from other hospitals.

## Project layout

```
brain-tumor-mri/
├── requirements.txt
├── app/dashboard.py          Streamlit dashboard
├── src/
│   ├── files.py              find images, remove duplicates / train-test leaks
│   ├── data.py               transforms, datasets, loaders
│   ├── model.py              EfficientNet-B0 / ResNet18 / MobileNetV3 + freezing helpers
│   ├── train.py              2-stage transfer learning (head, then fine-tune)
│   ├── evaluate.py           test metrics, confusion matrix, Grad-CAM gallery
│   ├── gradcam.py            Grad-CAM
│   ├── inference.py          Predictor (PyTorch) and OnnxPredictor
│   ├── export_onnx.py        export + verify ONNX
│   └── predict.py            command-line prediction
├── scripts/
│   ├── download_data.py      download dataset from Kaggle
│   └── make_dummy_data.py    tiny synthetic dataset for a 1-minute pipeline test
├── data/  models/  outputs/  (empty; filled as you go)
```

## 1. Setup (once)

Python 3.10+ is required.

**Windows (PowerShell)**
```powershell
cd brain-tumor-mri
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in cmd.)

**macOS / Linux**
```bash
cd brain-tumor-mri
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Quick pipeline test with fake data (about 2 minutes, no downloads)

```bash
python scripts/make_dummy_data.py --out data/dummy --per-class 60
python -m src.train --data-dir data/dummy --no-pretrained --epochs-head 1 --epochs-ft 3 --img-size 128 --num-workers 0
python -m src.evaluate --data-dir data/dummy --num-workers 0
streamlit run app/dashboard.py
```
This only checks that everything is wired up. The images are synthetic, so the numbers mean nothing medically.
Delete `outputs/` and `models/best_model.pt` afterwards so they don't mix with your real run.

## 3. Get the real dataset from Kaggle

Dataset: [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Option A: script (needs a free Kaggle account + API token)**
1. kaggle.com → profile picture → *Settings* → *Create New Token* (downloads `kaggle.json`).
2. Put `kaggle.json` in `C:\Users\<you>\.kaggle\` (Windows) or `~/.kaggle/` (macOS/Linux; then `chmod 600 ~/.kaggle/kaggle.json`).
3. Run:
```bash
python scripts/download_data.py
```

**Option B: manual.** Download the zip from the page above and extract it so you have:
```
data/brain-tumor-mri-dataset/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
data/brain-tumor-mri-dataset/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
```
(One extra nesting level is fine; the code finds the `Training`/`Testing` folders itself.)

## 4. Train

```bash
python -m src.train --data-dir data/brain-tumor-mri-dataset
```
What it does: removes duplicate images (and any training image that also appears in the test set), holds out 15 %
of training data for validation, trains the new head for 3 epochs, then fine-tunes the whole network for up to 12
epochs with early stopping, and saves the best model to `models/best_model.pt`.

Options worth knowing:

| Goal | Flag |
|---|---|
| Different model | `--model resnet18` or `--model mobilenet_v3_small` |
| Faster on a CPU laptop | `--img-size 160 --max-per-class 600 --epochs-ft 6` |
| Out of memory | `--batch-size 16` |
| Force device | `--device cpu` / `cuda` / `mps` |
| Windows data-loader problems | `--num-workers 0` |

Timing guide for the full dataset at 224 px: roughly 3 to 8 minutes per epoch on a free Kaggle/Colab GPU, but
roughly 20 to 40 minutes per epoch on a typical laptop CPU. If you have no NVIDIA GPU, see the next section.

### Train on Kaggle/Colab (free GPU) and run on your laptop
1. Upload this project folder to a Kaggle Notebook (or Colab), add the Brain Tumor MRI dataset as input, enable the GPU and Internet (Internet is needed once to fetch the ImageNet weights).
2. In a notebook cell:
```python
%cd /kaggle/working/brain-tumor-mri
!pip install -q -r requirements.txt
!python -m src.train --data-dir /kaggle/input/brain-tumor-mri-dataset
!python -m src.evaluate --data-dir /kaggle/input/brain-tumor-mri-dataset
```
3. Download `models/best_model.pt` and the whole `outputs/` folder, and put them in the same folders on your laptop. The dashboard then works without training locally.

## 5. Evaluate

```bash
python -m src.evaluate --data-dir data/brain-tumor-mri-dataset
```
Prints accuracy and per-class precision/recall/F1, and writes to `outputs/`: `metrics.json`,
`confusion_matrix.png`, `test_predictions.csv` and a `gradcam_samples/` gallery (correct and wrong predictions).
Look at the Grad-CAM images: if the heat is on image borders or text instead of the tumour, the model is cheating.

## 6. Dashboard

```bash
streamlit run app/dashboard.py
```
It opens at http://localhost:8501.

| Tab | What it does |
|---|---|
| **Results** | Accuracy, macro F1, per-class table, confusion matrix, training curves, Grad-CAM gallery |
| **Analyze image** | Upload an MRI slice, get class probabilities and a Grad-CAM overlay |
| **Camera snapshot** | Take a photo with the webcam (e.g. of an MRI on another screen) and classify it |
| **Live feed** | *Simulated MRI stream*: images from a folder are fed one by one, classified on arrival, with running accuracy.<br>*Webcam*: continuous live video with the prediction drawn on every frame |

Tip for the live feed demo: choose *Simulated MRI stream*, leave the folder as the dataset's `Testing` folder,
press **Start live stream**. For the webcam mode, show an MRI image full-screen on your phone and point the
laptop camera at it (allow camera access in the browser).

## 7. Fast CPU inference with ONNX (optional)

```bash
python -m src.export_onnx                       # writes models/model.onnx (+ .json) and verifies it
python -m src.predict path/to/mri.jpg --backend onnx
python -m src.predict path/to/mri.jpg --gradcam # PyTorch backend, saves a heat-map to outputs/predictions/
```
In the dashboard sidebar you can switch the backend to *ONNX Runtime*.

## Troubleshooting

* **"Could not find Training and Testing folders"**: check the folder layout in step 3, and the `--data-dir` path.
* **Webcam tab shows a warning**: `pip install streamlit-webrtc av`, or use the *Camera snapshot* tab. Use Chrome/Edge/Firefox and allow camera permission.
* **`torch` install is huge or fails**: install a CPU build from https://pytorch.org/get-started/locally/ first, then `pip install -r requirements.txt`.
* **Dashboard says the model is not available**: run training first, or copy `best_model.pt` to `models/`.
* **ONNX export error on a very new PyTorch**: `pip install onnxscript`, or skip ONNX; the PyTorch backend works on its own.
