"""A beginner-first, interactive lesson, independent of the trained model."""
import numpy as np
import pandas as pd
import streamlit as st

STEPS = ["1 · Inputs and answers", "2 · What is a weight?", "3 · Why add a bias?",
         "4 · What is a neuron?", "5 · Why an activation?", "6 · Why several layers?",
         "7 · How does it learn?"]


def render_fundamentals():
    st.markdown("### Start here. No machine learning knowledge needed.")
    st.write("A neural network turns numbers into a prediction. Let's build that idea one small piece at a time, before looking at the customer model.")
    st.session_state.setdefault("foundation_step", STEPS[0])
    def move(direction):
        index = STEPS.index(st.session_state.foundation_step)
        st.session_state.foundation_step = STEPS[max(0, min(6, index + direction))]
    step = st.selectbox("Choose a lesson", STEPS, key="foundation_step")
    index = STEPS.index(step)
    st.progress((index + 1) / len(STEPS), text=f"Lesson {index + 1} of 7 · Move the controls and explain what changes")
    left, right = st.columns(2)
    left.button("← Previous idea", on_click=move, args=(-1,), disabled=index == 0, width="stretch")
    right.button("Next idea →", on_click=move, args=(1,), disabled=index == 6, width="stretch")
    st.divider()
    if index == 0:
        st.markdown("### An input is a fact. A label is the known answer.")
        st.write("For a customer, an input might be the number of support calls. A feature is one kind of input; a row is one example. The label tells us what happened: did the customer leave?")
        st.dataframe(pd.DataFrame({"Customer (made up)": ["A", "B", "C"],
            "Support calls · input": [1, 6, 3], "Payment delay · input": [0, 18, 4],
            "Left? · label": [0, 1, 0]}), hide_index=True, width="stretch")
        st.info("During training we show the inputs AND the known answer. For a new customer we supply only inputs, and ask the model to estimate an answer.")
        st.markdown("#### What do we actually want it to learn?")
        st.write("A useful rule for combining the inputs. We choose the kind of mathematical function; training adjusts its numbers using examples. Three made-up rows are only an illustration, not enough to validate a model.")
        st.markdown("**Input → mathematical function → prediction**")
        st.markdown("#### The vocabulary, in plain language")
        st.table(pd.DataFrame({"Word": ["Input / feature", "Label / target", "Prediction", "Parameter", "Training"],
            "Meaning": ["A number describing this example", "The known answer for a historical example",
                        "The model's estimated answer", "An adjustable number inside the model, such as a weight or bias",
                        "Using examples to adjust the parameters"]}))
    elif index == 1:
        st.markdown("### A weight is a multiplier on an input.")
        st.write("Suppose a toy customer has a complaint score of 2. A weight of 3 makes that input contribute 6 to the neuron's score. A weight of 0 makes it contribute nothing; a negative weight reverses its direction.")
        st.caption("Toy arithmetic only. These sliders are not the trained churn model.")
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("Input: complaint score", 0.0, 5.0, 2.0, .5, key="weight_x")
            w = st.slider("Weight: how this input is multiplied", -3.0, 3.0, 1.0, .5, key="weight_w")
        with col2:
            st.metric("Contribution to the score", f"{x*w:.2f}")
            st.write(f"**{x:g} × {w:g} = {x*w:g}**")
            st.write("Positive weight: increasing this input raises the sum. Negative weight: increasing it lowers the sum. Zero weight: this connection contributes zero.")
        grid = np.linspace(0, 5, 21)
        st.line_chart(pd.DataFrame({"Input": grid, "Input × selected weight": grid*w}).set_index("Input"), color="#5de0c0")
        st.markdown("#### Why use weights?")
        st.write("Simply adding inputs would treat every input with the same multiplier. Separate weights let the model learn different combinations and directions. In a neuron with four inputs, there are four weights—one per incoming connection.")
        st.success("Try it: keep the input at 2. Set the weight to 0, then −1, then 3. Explain the result before reading it.")
        st.caption("A weight is not automatically a feature-importance score: input units, other weights, and activations also affect the final prediction.")
    elif index == 2:
        st.markdown("### Bias is an adjustable starting offset.")
        st.write("A weight multiplies an input. A bias is added afterward. It lets a neuron shift its score up or down without having to change the input multiplier.")
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("Input", 0.0, 5.0, 2.0, .5, key="bias_x")
            b = st.slider("Bias: added to the score", -5.0, 5.0, -3.0, .5, key="bias_b")
            st.caption("Weight stays fixed at 1, so you can isolate what bias does.")
        with col2:
            st.metric("Without bias: 1 × input", f"{x:.2f}")
            st.metric("With bias: 1 × input + bias", f"{x+b:.2f}")
        grid = np.linspace(0, 5, 21)
        st.line_chart(pd.DataFrame({"Input": grid, "Without bias": grid,
                                   "With selected bias": grid+b}).set_index("Input"), color=["#8795ab", "#5de0c0"])
        st.markdown("#### Why isn't a weight enough?")
        st.write("Without bias, the weighted sum must be zero whenever all its inputs are zero. Bias removes that restriction. On this graph, weight changes the slope; bias moves the line up or down.")
        st.write("For example, with weight 1 and bias −3, the sum is x − 3. A ReLU neuron starts producing positive output only when x exceeds 3. Without bias it would start as soon as x exceeds 0.")
        st.info("Bias here means a mathematical offset. It is not the same as social bias or unfairness. Every neuron has its own bias, learned during training.")
        st.success("Try it: set input to 0. Change the bias. The neuron can still have a nonzero score.")
    elif index == 3:
        st.markdown("### A neuron is a small calculation.")
        st.write("Multiply each input by its own weight, add the contributions, add one bias, then apply an activation. The activation decides how this score becomes the neuron's output.")
        st.caption("A two-input teaching neuron with made-up scores. The trained network has four inputs.")
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.slider("Complaint score x₁", 0.0, 5.0, 2.0, .5)
            w1 = st.slider("Complaint weight w₁", -3.0, 3.0, 1.0, .5)
        with c2:
            x2 = st.slider("Engagement score x₂", 0.0, 5.0, 1.0, .5)
            w2 = st.slider("Engagement weight w₂", -3.0, 3.0, -1.0, .5)
        bias = st.slider("Neuron bias b", -5.0, 5.0, -.5, .5)
        z = x1*w1 + x2*w2 + bias
        st.latex(r"z = w_1x_1 + w_2x_2 + b")
        st.table(pd.DataFrame({"Operation": ["Complaint contribution", "Engagement contribution", "Add bias", "Total score z"],
                               "Calculation": [f"{x1:g} × {w1:g} = {x1*w1:g}", f"{x2:g} × {w2:g} = {x2*w2:g}", f"{bias:g}", f"{z:g}"]}))
        st.metric("ReLU output: keep positive scores, otherwise return zero", f"{max(0,z):.2f}")
        st.write("This is what the circles in a network diagram represent: calculations with adjustable parameters. A software neuron is a simplified mathematical unit, not a biological brain cell.")
        st.info("Inputs change from customer to customer. Learned weights and biases stay fixed when making predictions. During training, the optimizer changes those parameters.")
    elif index == 4:
        st.markdown("### An activation transforms the neuron's score.")
        st.write("The weighted sum can be any number. An activation turns that sum into the output sent onward. Our hidden layers use ReLU; our final output uses Sigmoid.")
        z = st.slider("Try a score z", -6.0, 6.0, -2.0, .25)
        c1,c2,c3 = st.columns(3)
        c1.metric("No activation", f"{z:.3f}")
        c2.metric("ReLU: max(0, z)", f"{max(0,z):.3f}")
        c3.metric("Sigmoid: 1 / (1 + exp(−z))", f"{1/(1+np.exp(-z)):.3f}")
        grid = np.linspace(-6,6,121)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**ReLU: a bend in the function**")
            st.line_chart(pd.DataFrame({"Score":grid,"ReLU":np.maximum(0,grid)}).set_index("Score"), color="#5de0c0")
        with c2:
            st.markdown("**Sigmoid: a score between 0 and 1**")
            st.line_chart(pd.DataFrame({"Score":grid,"Sigmoid":1/(1+np.exp(-grid))}).set_index("Score"), color="#bd9bff")
        st.markdown("#### Why do hidden layers need a nonlinear activation?")
        st.write("Without one, stacking layers only makes a longer version of a linear calculation. For example, 2 × (3x + 1) + 4 simplifies to 6x + 6. ReLU introduces a bend, allowing combinations that a single linear layer cannot represent.")
        st.markdown("#### A probability and a decision are different.")
        st.write("Sigmoid converts a score of 0 into 0.5. A score of 2 becomes about 0.88. We interpret that as an estimated churn probability, then apply a separate threshold such as 0.5 to choose a label. The transformation alone does not guarantee accurate probabilities.")
    elif index == 5:
        st.markdown("### A layer is several neurons working on the same inputs.")
        st.write("Each neuron has different weights and its own bias, so it can produce a different combination. The next layer receives those outputs as its inputs.")
        st.markdown("**Customer's 4 inputs → 8 calculated outputs → 4 calculated outputs → 1 estimated probability**")
        st.table(pd.DataFrame({"Part of our network": ["Input", "First hidden layer", "Second hidden layer", "Output layer"],
            "What happens": ["Supply four scaled customer features", "8 neurons each combine the 4 inputs, then apply ReLU",
                             "4 neurons each combine the previous 8 outputs, then apply ReLU", "1 neuron combines 4 outputs, then applies Sigmoid"],
            "Learned parameters": ["None", "4 × 8 weights + 8 biases = 40", "8 × 4 weights + 4 biases = 36", "4 weights + 1 bias = 5"]}))
        st.info("Hidden means these values are intermediate calculations, not supplied answers. A hidden neuron does not automatically represent a named concept like 'loyalty'.")
        st.markdown("#### Who chose these numbers?")
        st.write("We chose 8 and 4 neurons to keep the demo small. Training learns the 81 weights and biases inside that structure. The number of layers, learning rate, and batch size are settings we choose, called hyperparameters. More neurons do not automatically mean better predictions.")
        st.markdown("#### Why scale the inputs first?")
        st.write("Spending might be measured in hundreds while calls are single digits. Subtracting the training mean and dividing by the training scale makes magnitudes more comparable, which helps optimization. The same saved transformation is used for every new customer.")
        st.success("Next, open Live network to inspect exactly these calculations for your own customer inputs.")
    else:
        st.markdown("### Learning means changing weights and biases using examples.")
        st.write("They start at initial values, usually small random weights. The model predicts; a loss function measures the mismatch with known answers. Gradients say how small parameter changes would change that loss. An optimizer uses those gradients to update parameters.")
        st.caption("A separate one-neuron experiment: fixed input x = 2, known label y = 1. It uses simple gradient descent for clarity; the full churn model uses Adam.")
        st.session_state.setdefault("toy_learning", {"weight": .2, "bias": -.5, "history": []})
        rate = st.slider("Learning rate: size of each update", .01, 1.0, .1, .01)
        def reset():
            st.session_state.toy_learning = {"weight": .2, "bias": -.5, "history": []}
        def learn():
            state = st.session_state.toy_learning
            score = 2*state['weight'] + state['bias']
            p = 1/(1+np.exp(-score))
            # Sigmoid + binary cross-entropy: dL/dz = p - y.
            error = p - 1
            state['weight'] -= rate * error * 2
            state['bias'] -= rate * error
            new_score = 2*state['weight'] + state['bias']
            state['history'].append({"Update":len(state['history'])+1,
                "Before loss":float(np.logaddexp(0,-score)), "After loss":float(np.logaddexp(0,-new_score)),
                "Weight":state['weight'], "Bias":state['bias']})
        c1,c2 = st.columns(2)
        c1.button("Learn from this example once", on_click=learn, width="stretch")
        c2.button("Reset learning experiment", on_click=reset, width="stretch")
        state = st.session_state.toy_learning
        score = 2*state['weight']+state['bias']
        p = 1/(1+np.exp(-score))
        c1,c2,c3 = st.columns(3)
        c1.metric("Current weight", f"{state['weight']:.4f}")
        c2.metric("Current bias", f"{state['bias']:.4f}")
        c3.metric("Prediction (known answer is 1)", f"{p:.4f}")
        st.write(f"Current score: 2 × {state['weight']:.4f} + ({state['bias']:.4f}) = **{score:.4f}**. Binary cross-entropy loss: **{np.logaddexp(0,-score):.4f}**.")
        if state['history']:
            st.dataframe(pd.DataFrame(state['history']), hide_index=True, width="stretch")
        st.markdown("#### What does the button actually do?")
        st.markdown("1. Calculate the prediction with the current weight and bias.\n2. Compare it with label 1 using a loss function.\n3. Calculate gradients: which direction changes the loss?\n4. Subtract learning rate × gradient from each parameter.")
        with st.expander("Show the arithmetic for the next update"):
            dw, db = (p-1)*2, p-1
            st.write(f"Weight gradient = (prediction − label) × input = ({p:.4f} − 1) × 2 = {dw:.4f}.")
            st.write(f"Bias gradient = prediction − label = {db:.4f}.")
            st.write(f"Next weight = {state['weight']:.4f} − {rate:.2f} × ({dw:.4f}) = {state['weight']-rate*dw:.4f}.")
            st.write(f"Next bias = {state['bias']:.4f} − {rate:.2f} × ({db:.4f}) = {state['bias']-rate*db:.4f}.")
        st.info("This experiment learns one example only. Doing well here says nothing about new customers. A real model learns from many examples and is evaluated on unseen data.")
        st.table(pd.DataFrame({"Term": ["Loss", "Gradient", "Backpropagation", "Optimizer", "Learning rate", "Batch", "Epoch", "Inference"],
            "Plain-language meaning": ["A number measuring prediction error", "How loss changes with a parameter",
                "Calculate gradients through the network", "Use gradients to update parameters", "Control update size",
                "A group of training examples", "One pass through the training set", "Predict with fixed learned parameters"]}))
        answer = st.radio("What changes when the model learns?", ["Choose an answer", "The historical labels", "The weights and biases", "The customer's inputs"])
        if answer == "The weights and biases": st.success("Exactly. Training adjusts the model's parameters to better predict the known labels.")
        elif answer != "Choose an answer": st.info("The examples and their labels provide the feedback. The optimizer changes the weights and biases.")
