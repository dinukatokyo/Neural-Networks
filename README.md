# Inside a Neural Network

A complete PyTorch + Streamlit teaching demo for customer churn. The model uses Support Calls, Payment Delay, Total Spend, and Last Interaction in a **4 → 8 → 4 → 1** network with ReLU, Sigmoid, and 81 trainable parameters.

## Run

From the project root, using the existing environment:

```powershell
.venv\Scripts\python.exe -m neural_networks.train
.venv\Scripts\python.exe -m streamlit run src/neural_networks/app.py
```

If creating an environment from the existing lockfile, use `uv sync`. `uv run python -m neural_networks.train` and `uv run streamlit run src/neural_networks/app.py` are equivalent launch options.

Training runs once and writes `artifacts/`. The frontend only loads saved assets. The app displays a helpful setup message if they are missing. Run the notebook from its first cell after restarting its kernel for a reproducible demonstration.

## What is included

- **Concept lesson:** interactive single-neuron sandbox and a presenter guide in `LESSON.md`.
- **Live network:** editable inputs, three real held-out examples, actual intermediate activations, a stage reveal, a per-neuron product/weight/bias calculation, and a one-feature sensitivity plot.
- **Training and evaluation:** training/validation loss, confusion matrix, accuracy, precision, recall, F1, ROC AUC, average precision, and binary cross-entropy. Compare with logistic regression and a majority-class baseline.
- **Guided notebook:** `notebooks/model_training.ipynb` calls the shared training pipeline and independently checks saved predictions, explains code, and renders evaluation charts.

## Reproducibility and evaluation

Invalid rows are removed before constructing features and labels. Identical selected-feature/label rows are deduplicated. Seed 42 creates stratified train/validation/test splits of 64%/16%/20%. Only training rows fit the StandardScaler and model parameters. Twenty epochs use Adam at 0.001, batches of 512, and BCELoss paired with explicit Sigmoid. The checkpoint with lowest validation loss is selected. Test evaluation uses a fixed 0.5 threshold. Run `--epochs N --batch-size N` to change training; saved metadata records the settings.

The external `data/test.csv` is evaluated separately when present; feature vectors overlapping the original cleaned data are excluded from that check. Its performance is reported separately rather than mixed with the internal holdout. File provenance, temporal independence, and probability calibration remain unverified. Earlier exploratory feature selection used the source dataset, so the internal holdout is a demonstration estimate rather than a pristine final research benchmark. A customer-group/time split and independently collected validation data would be needed for stronger deployment claims.

Artifacts include model weights, scaler statistics in metadata JSON, metrics, loss history, held-out predictions, and example customers. Raw data remains local. The browser never sends customer inputs to an external prediction service.

## Files

```text
src/neural_networks/model.py   Shared network and intermediate activations
src/neural_networks/train.py   Cleaning, splitting, training, evaluation, export
src/neural_networks/app.py     Streamlit teaching studio
notebooks/model_training.ipynb
notebooks/data_analysis.ipynb Original exploratory analysis
LESSON.md                     Presentation script and audience questions
artifacts/                    Generated model and measured results
```

See `VERIFICATION.md` for checks and measured results from this build.

## Verify

Run the repeatable model and frontend checks with: .venv\Scripts\python.exe tests\verify_demo.py
