"""Streamlit teaching studio: concepts, real activations, code, and evaluation."""
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_networks.model import ChurnNN, FEATURES
from neural_networks.fundamentals import render_fundamentals

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Inside a Neural Network", page_icon="🧠", layout="wide")
st.markdown('''<style>
.stApp {background: #0b1020; color: #e7edf7;}
[data-testid="stHeader"] {background: #0b1020;}
[data-testid="stSidebar"] {background: #11192b; border-right: 1px solid #26334b;}
.block-container {max-width: 1440px; padding-top: 2.4rem;}
h1,h2,h3 {letter-spacing: -.035em;}
.eyebrow {color:#5de0c0; text-transform:uppercase; letter-spacing:.18em; font-size:12px; font-weight:700;}
.hero {font-size: clamp(30px,4vw,54px); line-height:1.08; font-weight:750; margin:12px 0; letter-spacing:-.05em;}
.sub {color:#a9b7cd; font-size:17px; max-width:780px; line-height:1.7; margin-bottom:25px;}
.card {background:#141e32; border:1px solid #293650; border-radius:16px; padding:22px; min-height:155px;}
.card h3 {font-size:20px; margin:8px 0; color:#edf4ff;}
.card p {color:#b9c7db; line-height:1.6; font-size:15px;}
.step {font-size:12px; color:#5de0c0; letter-spacing:.12em;}
[data-testid="stMetric"] {background:#141e32; border:1px solid #293650; padding:16px; border-radius:12px;}
.stButton>button {border-radius:9px;}
[data-testid="stCaptionContainer"] {color:#a9b7cd;}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def load_assets(version):
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    model = ChurnNN()
    model.load_state_dict(torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return (model, metadata, json.loads((ARTIFACTS / "metrics.json").read_text()),
            pd.DataFrame(json.loads((ARTIFACTS / "history.json").read_text())),
            pd.read_csv(ARTIFACTS / "examples.csv"))


def card(step, title, body):
    st.markdown(f'<div class="card"><div class="step">{step}</div><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True)


def graph_svg(trace, model, stage):
    values = [trace["inputs"], trace["a1"], trace["a2"], trace["probability"]]
    xs = [105, 370, 635, 900]
    ys = [np.linspace(95, 435, len(v)) if len(v)>1 else np.array([265]) for v in values]
    weights = [model.layer1.weight.detach().numpy(), model.layer2.weight.detach().numpy(),
               model.output.weight.detach().numpy()]
    parts = ['<svg viewBox="0 0 1010 515" xmlns="http://www.w3.org/2000/svg" role="img" '
             'aria-label="Actual neural network activations and weighted connections">',
             '<rect width="1010" height="515" rx="18" fill="#11192b"/>']
    titles = ['INPUT · 4', 'HIDDEN · 8 / ReLU', 'HIDDEN · 4 / ReLU', 'OUTPUT · Sigmoid']
    for layer, title in enumerate(titles):
        parts.append(f'<text x="{xs[layer]}" y="35" text-anchor="middle" fill="#acbad0" '
                     f'font-family="sans-serif" font-size="13">{title}</text>')
    for layer, w in enumerate(weights):
        scale = max(float(np.max(np.abs(w))), 1e-8)
        for j in range(w.shape[0]):
            for i in range(w.shape[1]):
                active = layer < stage
                color = '#5de0c0' if w[j,i] >= 0 else '#bd9bff'
                opacity = (0.16 + 0.48 * abs(float(w[j,i])) / scale) if active else .04
                parts.append(f'<line x1="{xs[layer]+23}" y1="{ys[layer][i]}" '
                             f'x2="{xs[layer+1]-23}" y2="{ys[layer+1][j]}" stroke="{color}" '
                             f'stroke-width="{.6+2.2*abs(float(w[j,i]))/scale}" opacity="{opacity}">'
                             f'<title>Weight {float(w[j,i]):.5f}</title></line>')
    for layer, vals in enumerate(values):
        for i, value in enumerate(vals):
            active = layer <= stage
            fill = '#24564f' if active and value > 0 else '#1c2940'
            edge = '#5de0c0' if active and value > 0 else '#64738b'
            label = f'{value:.2f}' if active else '…'
            parts.append(f'<circle cx="{xs[layer]}" cy="{ys[layer][i]}" r="23" fill="{fill}" '
                         f'stroke="{edge}" stroke-width="1.5"/>'
                         f'<text x="{xs[layer]}" y="{ys[layer][i]+4}" text-anchor="middle" '
                         f'fill="#f1f6ff" font-family="monospace" font-size="11">{label}</text>')
            if layer == 0:
                parts.append(f'<text x="{xs[layer]}" y="{ys[layer][i]+39}" text-anchor="middle" '
                             f'fill="#acbad0" font-family="sans-serif" font-size="11">{html.escape(FEATURES[i])}</text>')
    parts.append('<text x="505" y="498" text-anchor="middle" fill="#acbad0" font-family="sans-serif" '
                 'font-size="12">Mint = positive weight · Lilac = negative weight · Line width = weight magnitude within each layer</text></svg>')
    return ''.join(parts)


st.markdown('<div class="eyebrow">A hands-on introduction / Customer churn</div>'
            '<div class="hero">Inside a neural network.</div>'
            '<div class="sub">Follow four customer signals through 13 neurons. See a prediction, '
            'open up the mathematics, and discover what learning actually changes.</div>', unsafe_allow_html=True)

required = ["model.pt", "metadata.json", "metrics.json", "history.json", "examples.csv"]
if not all((ARTIFACTS / name).exists() for name in required):
    st.info("Train the model once to prepare the lesson and live demo.")
    st.code("python -m neural_networks.train", language="bash")
    st.stop()
model, meta, metrics, history, examples = load_assets(tuple((ARTIFACTS / n).stat().st_mtime_ns for n in required))
with st.sidebar:
    st.markdown("### The learning studio")
    page = st.radio("Explore", ["01 · The concept", "02 · Live network", "03 · Training & evaluation", "04 · The code"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**One question**\n\nWill this customer churn?")
    st.caption("Churn = the customer leaves. The model learns associations from historical examples.")
    st.markdown("**4 → 8 → 4 → 1**")
    st.caption(f"{meta['parameter_count']} learned weights and biases · CPU model · no live retraining")
    st.divider()
    st.caption("Suggested order: concept → live network → evaluation → code. Predictions are educational estimates, not calibrated guarantees.")

if page.startswith("01"):
    render_fundamentals()

elif page.startswith("02"):
    st.markdown("### Give the network a customer")
    st.caption("All activations and calculations below come from the saved trained model, using your current inputs.")
    for feature in FEATURES:
        st.session_state.setdefault(feature, float(meta["feature_median"][feature]))
    def use_example(index):
        for feature in FEATURES:
            st.session_state[feature] = float(examples.iloc[index][feature])
    buttons = st.columns(3)
    for index, col in enumerate(buttons):
        col.button(f"Load {examples.iloc[index]['example'].lower()} example", on_click=use_example, args=(index,), width="stretch")
    cols = st.columns(4)
    values = []
    for col, feature in zip(cols, FEATURES):
        with col:
            values.append(st.number_input(feature, min_value=0.0, step=1.0 if feature != "Total Spend" else 10.0,
                                           key=feature, help="Keep the dataset's original units; no currency or time unit is assumed."))
    raw = np.array(values, dtype=np.float32)
    scaled = (raw - np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])
    trace = model.inspect(torch.tensor(scaled, dtype=torch.float32).reshape(1, 4))
    probability = float(trace["probability"][0])
    outside = [f for f, value in zip(FEATURES, values) if value < meta['feature_min'][f] or value > meta['feature_max'][f]]
    if outside: st.warning("Outside the training range: " + ", ".join(outside) + ". Predictions here are extrapolations.")
    threshold = st.slider("Decision threshold (changes the label, not the network)", 0.05, 0.95, 0.5, 0.05)
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn estimate", f"{probability:.1%}")
    c2.metric("Decision", "Churn" if probability >= threshold else "Stay")
    c3.metric("Decision threshold", f"{threshold:.0%}")
    st.caption("A sigmoid score is interpreted as an estimated probability. Calibration has not been established. Changing an input shows model behavior, not a causal effect.")
    st.markdown("### Watch the forward pass")
    stage_name = st.select_slider("Reveal the stages", options=["Standardized inputs", "Hidden layer 1", "Hidden layer 2", "Prediction"], value="Prediction")
    stage = ["Standardized inputs", "Hidden layer 1", "Hidden layer 2", "Prediction"].index(stage_name)
    st.image(graph_svg(trace, model, stage), width="stretch")
    st.caption("Node labels are actual activation values. ReLU nodes with a zero output are inactive for this customer. Connection widths are normalized per layer, not an attribution of feature importance.")
    with st.expander("1 · What numbers does the network receive?", expanded=False):
        st.latex(r"x_{scaled} = (x - \mu_{train}) / \sigma_{train}")
        st.dataframe(pd.DataFrame({"Feature": FEATURES, "Your input": raw, "Training mean": meta['scaler_mean'],
                                   "Training scale": meta['scaler_scale'], "Network input": scaled}), hide_index=True, width="stretch")
        st.write("The scaler was fitted on training data only. These saved statistics are reused unchanged for every prediction.")
    with st.expander("2 · Inspect the calculation inside any neuron", expanded=True):
        layer_name = st.selectbox("Layer", ["Hidden layer 1", "Hidden layer 2", "Output"])
        layer, incoming, pre, post = {"Hidden layer 1": (model.layer1, trace['inputs'], trace['z1'], trace['a1']),
                                     "Hidden layer 2": (model.layer2, trace['a1'], trace['z2'], trace['a2']),
                                     "Output": (model.output, trace['a2'], trace['z3'], trace['probability'])}[layer_name]
        neuron = st.selectbox("Neuron", list(range(1, layer.out_features+1))) - 1
        weights = layer.weight.detach().numpy()[neuron]
        bias = float(layer.bias.detach()[neuron])
        labels = FEATURES if layer_name == 'Hidden layer 1' else [f"Previous neuron {i+1}" for i in range(len(incoming))]
        st.dataframe(pd.DataFrame({"Input": labels, "Activation": incoming, "Weight": weights,
                                   "Input × weight": incoming*weights}), hide_index=True, width="stretch")
        st.write(f"Sum of products **{float(np.dot(incoming,weights)):.5f}** + bias **{bias:.5f}** = **{float(pre[neuron]):.5f}**")
        activation = "Sigmoid" if layer_name == "Output" else "ReLU"
        st.write(f"{activation}({float(pre[neuron]):.5f}) → **{float(post[neuron]):.5f}**")
        if activation == 'Sigmoid': st.latex(r"\sigma(z) = \frac{1}{1+e^{-z}}")
    with st.expander("3 · How does one input affect this prediction?"):
        feature = st.selectbox("Vary one feature; hold the other three fixed", FEATURES)
        i = FEATURES.index(feature)
        grid = np.linspace(meta['feature_min'][feature], meta['feature_max'][feature], 80)
        rows = np.tile(raw, (len(grid),1)); rows[:,i] = grid
        network_inputs = (rows - np.array(meta['scaler_mean'])) / np.array(meta['scaler_scale'])
        with torch.no_grad(): curve = model(torch.tensor(network_inputs, dtype=torch.float32)).numpy().ravel()
        st.line_chart(pd.DataFrame({feature:grid, "Estimated churn probability":curve}).set_index(feature), color="#5de0c0")
        st.caption("This is a model sensitivity plot. Some combinations may be uncommon in the data; it does not establish causation.")

elif page.startswith("03"):
    st.markdown("### Did it learn? Does it generalize?")
    st.write("Training adjusts weights. Validation selects the best epoch. The holdout is evaluated after selection, with a fixed 0.5 decision threshold.")
    cols = st.columns(3)
    for col, (name, count) in zip(cols, meta['splits'].items()): col.metric(name.title() + " rows", f"{count:,}")
    st.line_chart(history.set_index('epoch')[['training_loss', 'validation_loss']], color=["#5de0c0", "#bd9bff"])
    st.caption(f"Saved model: epoch {meta['best_epoch']} (lowest validation loss). Training loss is averaged over batches while weights change; validation loss is measured at the end of each epoch.")
    options = ['Internal holdout'] + (['Separate test.csv'] if 'external_test_file' in metrics else [])
    dataset = st.radio("Evaluation dataset", options, horizontal=True)
    result = metrics['neural_network' if dataset == 'Internal holdout' else 'external_test_file']
    cols = st.columns(4)
    for col, key in zip(cols, ['accuracy', 'precision', 'recall', 'roc_auc']): col.metric(key.replace('_',' ').title(), f"{result[key]:.3f}")
    st.caption(f"{result['rows']:,} evaluated customers · {result['churn_rate']:.1%} churn prevalence · F1 {result['f1']:.3f} · Average precision {result['average_precision']:.3f} · BCE {result['log_loss']:.3f}")
    left,right = st.columns(2)
    with left:
        st.markdown("#### Where were the mistakes?")
        cm = result['confusion_matrix']
        st.dataframe(pd.DataFrame(cm, index=['Actual stay', 'Actual churn'], columns=['Predicted stay', 'Predicted churn']), width="stretch")
        st.write(f"**{cm[1][0]:,} missed churners** (false negatives) and **{cm[0][1]:,} false alarms** (false positives).")
        st.caption("Precision: of predicted churners, how many churned? Recall: of actual churners, how many were found?")
    with right:
        st.markdown("#### Across decision thresholds")
        st.line_chart(pd.DataFrame({'False positive rate':result['roc']['fpr'], 'True positive rate':result['roc']['tpr']}).set_index('False positive rate'), color='#5de0c0')
        st.caption("ROC AUC summarizes ranking across thresholds; 0.5 corresponds to random ranking. It does not measure probability calibration.")
    st.markdown("#### Compare with simpler models")
    names = ['neural_network','logistic_regression','majority_baseline'] if dataset == 'Internal holdout' else ['external_test_file','external_logistic_regression']
    comparison = pd.DataFrame({name.replace('_',' ').title():{k:metrics[name][k] for k in ['accuracy','precision','recall','f1','roc_auc']} for name in names}).T
    st.dataframe(comparison.style.format('{:.3f}'), width="stretch")
    st.caption("The majority baseline predicts the most common training label; its probability is the training churn rate. Logistic regression uses exactly the same training rows and scaled features.")
    if 'external_test_file' in metrics:
        internal, external = metrics['neural_network'], metrics['external_test_file']
        st.info(f"Separate-file check: accuracy {internal['accuracy']:.1%} → {external['accuracy']:.1%}; ROC AUC {internal['roc_auc']:.3f} → {external['roc_auc']:.3f}. The provenance and sampling of test.csv are not documented here, so treat it as a separate distribution check, not a guaranteed production estimate.")
    with st.expander("Data quality and evaluation limits"):
        st.json(meta['cleaning'])
        if meta['external_cleaning']: st.json(meta['external_cleaning'])
        st.write("Invalid rows are removed before features and labels are built. Exact duplicate selected-feature/label rows are removed before splitting. This is a random row split, not a customer-group or time split. The four features were chosen in earlier exploration; for a rigorous final study, freeze feature selection before examining a final holdout. Dataset provenance, collection timing, and calibration remain unverified.")

else:
    st.markdown("### From the diagram to Python")
    st.write("The diagram, saved model, and frontend share this exact model class.")
    st.code((ROOT/'src/neural_networks/model.py').read_text(), language='python')
    st.markdown("### Five lines that explain training")
    st.code('''optimizer.zero_grad()                 # Clear previous gradients
probabilities = model(inputs)         # Forward pass
loss = criterion(probabilities, labels)  # Binary cross-entropy
loss.backward()                       # Backpropagation: compute gradients
optimizer.step()                      # Adam: update weights and biases''', language='python')
    st.info("This model includes Sigmoid, so training uses BCELoss. A common alternative is to remove Sigmoid and use BCEWithLogitsLoss; do not apply both sigmoid transformations.")
    st.markdown("### From raw customer to prediction")
    st.code('''scaled = (customer - training_mean) / training_scale
model.eval()
with torch.no_grad():
    probability = model(torch.tensor(scaled, dtype=torch.float32))
prediction = probability >= 0.5''', language='python')
    st.caption("Evaluation mode controls training-specific layer behavior. no_grad disables gradient recording. They have different jobs.")
    st.markdown("### Reproduce the demo")
    st.code('''python -m neural_networks.train
python -m streamlit run src/neural_networks/app.py''', language='bash')
    st.write("Open notebooks/model_training.ipynb for the complete guided walkthrough, and LESSON.md for the presentation script.")
