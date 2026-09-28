from pathlib import Path
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(root/'src/neural_networks/app.py'), default_timeout=40).run()
assert not app.exception, app.exception
steps = list(app.selectbox[0].options)
for step in steps:
    app.selectbox(key='foundation_step').set_value(step).run()
    assert not app.exception, app.exception
    print('PASS:', step, flush=True)
app.selectbox(key='foundation_step').set_value(steps[1]).run()
app.slider(key='weight_w').set_value(0.0).run()
assert float(app.metric[0].value) == 0
app.slider(key='weight_w').set_value(-1.0).run()
assert float(app.metric[0].value) == -2
app.selectbox(key='foundation_step').set_value(steps[2]).run()
app.slider(key='bias_x').set_value(0.0).run()
assert float(app.metric[0].value) == 0 and float(app.metric[1].value) == -3
app.selectbox(key='foundation_step').set_value(steps[6]).run()
before = float(app.metric[2].value)
next(b for b in app.button if b.label == 'Learn from this example once').click().run()
assert float(app.metric[2].value) > before
state = app.session_state['toy_learning']
assert state['history'][0]['After loss'] < state['history'][0]['Before loss']
assert abs(state['weight'] - .304995837495788) < 1e-8
next(b for b in app.button if b.label == 'Reset learning experiment').click().run()
assert app.session_state['toy_learning']['weight'] == .2
assert not app.exception, app.exception
print('PASS: zero and negative weights, bias at zero input, exact gradient update, lower loss, reset', flush=True)
