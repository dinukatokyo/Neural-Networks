# Verification — 28 September 2026

The implementation was trained and checked using the existing project environment (PyTorch 2.14.0 CPU). No dependency installation was required.

## Measured results

| Model / evaluation set | Rows | Accuracy | Churn precision | Churn recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Neural network / internal holdout | 87,725 | 92.02% | 99.36% | 86.44% | 0.9245 | 0.9534 |
| Logistic regression / internal holdout | 87,725 | 84.00% | 87.24% | 83.98% | 0.8558 | 0.9157 |
| Majority baseline / internal holdout | 87,725 | 56.52% | 56.52% | 100.00% | 0.7222 | 0.5000 |
| Neural network / separate test file | 62,784 | 58.69% | 53.42% | 98.92% | 0.6937 | 0.6526 |
| Logistic regression / separate test file | 62,784 | 58.84% | 53.57% | 97.25% | 0.6909 | 0.7626 |

Threshold: 0.5. The separate test file performs substantially worse; this is visible in the notebook and app. Its provenance and distribution are not documented, so this is evidence of limited transfer, not a reason to tune on that test set.

## Data and training

- Source train.csv: 440,833 rows. Removed one invalid row and 2,208 identical selected-feature/label rows.
- Training: 280,719 rows. Validation: 70,180. Internal holdout: 87,725.
- External test.csv: 64,374 rows; removed 199 duplicate rows and 1,391 feature vectors overlapping cleaned source data before evaluation.
- Twenty epochs, batches of 512, seed 42, Adam learning rate 0.001.
- Best validation checkpoint: epoch 20; validation BCE approximately 0.2144.
- Model, scaler statistics, metrics, history, and examples saved together in artifacts/.

## Checks completed

- Executed all nine notebook code cells in a fresh IPython session. Saved actual text, tables, and plot outputs into the notebook.
- Reloaded saved weights and preprocessing statistics; reproduced held-out probabilities within 1e-6.
- Independently recomputed accuracy, precision, recall, F1, ROC AUC, and log loss from those predictions.
- Recomputed every layer's weighted sums, biases, and activations with direct array arithmetic and compared with the neuron inspector.
- Verified exactly one history record per epoch.
- Used Streamlit AppTest to render all four pages and exercise example buttons, edited input, threshold invariance, layer selection, reveal stages, and external evaluation selection.
- Visually inspected the concept page and live network in the browser; corrected a clipped formula and SVG rendering found during visual QA.

## Remaining scope limits

This is an educational predictor, not a validated deployment. Calibration, customer-group/time independence, feature collection timing, and dataset provenance remain unverified. The existing exploratory feature selection used the source data. The separate-file gap is shown openly in the demo. Changing an input is a sensitivity experiment, not evidence of a causal intervention.
