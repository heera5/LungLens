# LungLens
**Explainable multimodal decision support for lung cancer screening**
CT-scan deep learning (Normal / Benign / Malignant, with Grad-CAM) + symptom-questionnaire machine learning, in one Flask web app.

> Research prototype for screening support. Not a medical device and not a diagnosis.

## What is inside
```
LungLens/
├── BACKEND/
│   ├── LungLens_CT_Training.ipynb   # Colab notebook: trains DenseNet121 / ResNet50 / EfficientNet-B0 on CT slices
│   ├── lungdisease.ipynb            # original symptom-model notebook (Naive Bayes, SVM, Perceptron)
│   ├── data.csv                     # symptom dataset
│   └── perceptron.pkl               # symptom model
├── FRONTEND/
│   ├── app.py                       # Flask routes (symptoms, CT analysis, combined report, login)
│   ├── ct_model.py                  # loads ct_model.pt, predicts, draws the Grad-CAM heatmap
│   ├── ct_core.py                   # model code shared with the notebook
│   ├── perceptron.pkl, users.db
│   ├── static/ , templates/
│   └── ct_model.pt                  # <- you create this in Colab and copy it here
├── requirements.txt
└── run.py
```

## 1. Train the CT model (Google Colab, free GPU)
1. Download the IQ-OTH/NCCD lung cancer CT dataset from its original source (see Data availability) and upload `archive_1.zip` to Google Drive: `MyDrive/LungLens/archive_1.zip`.
2. Open `BACKEND/LungLens_CT_Training.ipynb` in Colab (File > Upload notebook).
3. Runtime > Change runtime type > **T4 GPU**.
4. Optional smoke test: set `QUICK_TEST = True`, run all (about 2 minutes), then set it back to `False`.
5. Runtime > Run all (about 15-40 minutes).
6. The notebook downloads **`ct_model.pt`** and `LungLens_results.zip` (tables, figures, results.json).

## 2. Run the web app on Windows
```
cd LungLens
pip install flask joblib pandas numpy scikit-learn pillow opencv-python-headless
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Copy `ct_model.pt` into the `FRONTEND` folder, then:
```
python run.py
```
Open **http://127.0.0.1:5000**. A demo doctor account is created on first run (see `FRONTEND/app.py`); change it before any real use.
Without `ct_model.pt` the app still works; the CT page shows a notice instead.

## 3. Using the app
**Full Assessment (recommended)** - a guided 3-step flow:
1. **Symptom questionnaire** (or use the Low/High-Risk preset buttons).
2. **CT scan** - upload a CT slice (PNG/JPG). You get the class, probabilities and a Grad-CAM heatmap. You can skip this step for a provisional result.
3. **Final report** - one risk level (LOW / MODERATE / HIGH) with the reason, the next step and the rule table.

**How the final level is decided (CT leads; the questionnaire can only raise the level, never lower it):**

| Questionnaire | CT scan | Final level |
|---|---|---|
| Lower risk | Normal | LOW - routine checkup |
| Higher risk | Normal | MODERATE - doctor review recommended |
| Lower risk | Benign | MODERATE - follow-up scan advised |
| Higher risk | Benign | HIGH - doctor review soon |
| Any | Malignant | HIGH - urgent specialist referral |

**History:** sign in *before* starting an assessment and the finished result is saved to the local SQLite database
(`users.db`). Open **History** to see or delete saved rows. Only results are saved - CT images are processed in memory and never stored.
You can also use **Symptom Check** and **CT Analysis** on their own from the menu.

## Honest limitations (please keep these in the paper)
* The CT dataset has no patient IDs and contains near-identical neighbouring slices. The notebook uses a **block-wise split with purging** as a proxy for a patient-wise split, removes exact duplicates, and reports a **leakage score** and a comparison with a naive random split.
* The CT model classifies single slices, was trained on a small public dataset (Benign has few images) and has no external validation.
* `data.csv` (symptoms) contains many repeated rows, so symptom-model accuracy on a random split is optimistic.
* The combined suggestion is a simple transparent rule that has not been clinically validated. No dataset contains patients with both a questionnaire and a CT scan, so the combination cannot be trained or validated statistically; it is rule-based decision support.
* Saved assessments contain patient data (age, sex, results). Keep `users.db` private.

## Troubleshooting
| Problem | Fix |
|---|---|
| CT page says the model is not ready | Put `ct_model.pt` in `FRONTEND/` and make sure `torch` and `torchvision` are installed |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| `'python' is not recognized` | Reinstall Python and tick "Add Python to PATH" |
| Port 5000 already in use | Close the other program or restart the terminal |

## Data availability
The CT images come from the public IQ-OTH/NCCD dataset ([cite the original source]); the symptom data in `BACKEND/data.csv` is from [source/licence]. The CT images are not redistributed here.
`ct_model.pt` (deployed DenseNet121 checkpoint) is attached to the GitHub Release and the Zenodo archive.

## Citation and licence
Released under the MIT licence (see `LICENSE`). Please cite using `CITATION.cff`
(Heera G, LungLens, v1.0.0, https://github.com/heera5/LungLens, DOI: 10.5281/zenodo.23623015).
