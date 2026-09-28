import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from neural_networks.model import ChurnNN, FEATURES
from neural_networks.train import evaluate

meta = json.loads((root/'artifacts/metadata.json').read_text())
metrics = json.loads((root/'artifacts/metrics.json').read_text())
history = json.loads((root/'artifacts/history.json').read_text())
assert len(history) == meta['epochs'] == 20
holdout = pd.read_csv(root/'artifacts/holdout_predictions.csv')
model = ChurnNN()
model.load_state_dict(torch.load(root/'artifacts/model.pt', weights_only=True))
model.eval()
scaled = (holdout[FEATURES].to_numpy() - meta['scaler_mean'])/meta['scaler_scale']
with torch.no_grad(): p = model(torch.tensor(scaled, dtype=torch.float32)).numpy().ravel()
np.testing.assert_allclose(p, holdout.probability, atol=1e-6)
measured = evaluate(holdout.Churn, p)
for key in ['accuracy','precision','recall','f1','roc_auc','log_loss']:
    assert abs(measured[key] - metrics['neural_network'][key]) < 1e-6, key
trace = model.inspect(torch.tensor(scaled[:1], dtype=torch.float32))
previous = trace['inputs']
for layer, pre, post in [(model.layer1,'z1','a1'),(model.layer2,'z2','a2'),(model.output,'z3','probability')]:
    manual = layer.weight.detach().numpy() @ previous + layer.bias.detach().numpy()
    np.testing.assert_allclose(manual, trace[pre], atol=1e-5)
    previous = np.maximum(manual,0) if post != 'probability' else 1/(1+np.exp(-manual))
    np.testing.assert_allclose(previous, trace[post], atol=1e-5)
print('PASS: saved prediction reload, evaluation metrics, history, and all neuron calculations', flush=True)

app = AppTest.from_file(str(root/'src/neural_networks/app.py'), default_timeout=40).run()
assert not app.exception, app.exception
for page in ['02 · Live network','03 · Training & evaluation','04 · The code','01 · The concept']:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
    print('PASS page:',page, flush=True)
app.sidebar.radio[0].set_value('02 · Live network').run()
for button in app.button:
    button.click().run()
    assert not app.exception, app.exception
app.number_input[0].set_value(6.0).run()
assert not app.exception, app.exception
before = app.metric[0].value
app.slider[0].set_value(0.85).run()
assert app.metric[0].value == before
for layer in ['Hidden layer 2','Output']:
    app.selectbox[0].set_value(layer).run()
    assert not app.exception, app.exception
app.select_slider[0].set_value('Standardized inputs').run()
assert not app.exception, app.exception
app.sidebar.radio[0].set_value('03 · Training & evaluation').run()
app.radio[0].set_value('Separate test.csv').run()
assert not app.exception, app.exception
print('PASS: examples, input edits, threshold invariance, layer inspection, reveal stages, external evaluation', flush=True)
