# Neural networks from the very beginning

Allow 45–55 minutes for a group new to machine learning. Open **01 · The concept** in the Streamlit app and follow the seven lessons using **Next idea**. Ask people to predict each result before moving a control.

## 1. Inputs and answers · 4 minutes

“An input is a fact about an example. A feature is a kind of input, such as support-call count. A label is the known answer for a historical example: did this customer leave? A prediction is the model's estimated answer.”

Show the made-up customer rows. During training, inputs and labels are available. For a new customer, only the inputs are supplied. We want a mathematical function that combines these inputs usefully. The small table illustrates terminology; it is not a training dataset.

## 2. What is a weight? · 5 minutes

“A weight is a multiplier. Input 2 with weight 3 contributes 6 to a score.”

Keep the input at 2. Change weight from 1 to 0, then −1, then 3. Let colleagues explain the change. Show the graph: weight changes the slope. Positive and negative describe how increasing this input changes this neuron's sum.

Why use it? Simply summing inputs gives every connection the same multiplier. Individual weights let the model learn different combinations and directions. Each incoming connection has a weight. A large weight alone is not proof of global feature importance; input units and the rest of the network matter.

## 3. Why add a bias? · 5 minutes

“A bias is a number added after multiplying and summing. It gives the neuron an adjustable offset.”

The weight is fixed at 1 in this lesson. Set input to zero and move the bias. Compare the two lines. Without bias the score must be zero at zero input; with bias it can start elsewhere. Weight changes slope, bias shifts the line.

Use x − 3 as an example: a ReLU output becomes positive only above x = 3. This explains how bias shifts when the neuron activates. Every neuron has its own bias. Mathematical bias is not the same as unfairness or social bias.

## 4. What is a neuron? · 5 minutes

“Multiply each input by its own weight, add those contributions, add one bias, and transform the result with an activation.”

Use the two-input calculator. Read each table row aloud. Ask which numbers belong to the customer and which belong to the model. Customer inputs change for each prediction; trained parameters stay fixed. Parameters change during training.

The toy scores and weights are illustrative; they do not change the saved churn model. A software neuron is a mathematical unit, not a biological cell.

## 5. Why an activation? · 5 minutes

“An activation transforms the neuron's score into the output sent onward.”

Move the score from negative to zero to positive. ReLU returns zero below zero and preserves positive values. Sigmoid maps scores into (0, 1): zero becomes 0.5; two becomes about 0.88.

Explain why nonlinearity matters: 2 × (3x + 1) + 4 still simplifies to 6x + 6. Stacking linear layers alone doesn't escape a linear function. ReLU introduces a bend.

Sigmoid output is interpreted as an estimated probability. A threshold separately converts it into a label. Sigmoid alone does not guarantee calibration or correct predictions.

## 6. Why several layers? · 4 minutes

“A layer contains several neurons, each calculating a different combination. The next layer uses those outputs as its inputs.”

Trace the 4 → 8 → 4 → 1 architecture. There are 81 learned parameters: 40, 36, and 5 across the three computing layers. Hidden means intermediate, not a mysterious named concept such as loyalty.

We chose the layer sizes; the training process learns weights and biases. Settings such as layer size, learning rate, and batch size are called hyperparameters. Larger networks are not automatically better.

Explain scaling: a spending value in the hundreds and a small call count have different magnitudes. Training-only scaling makes magnitudes more comparable. New inputs use the same saved transformation.

## 7. How does it learn? · 6 minutes

Keep the experiment's learning rate at 0.1. Its input is 2 and known label is 1. Before clicking **Learn from this example once**, read the current weight, bias, probability, and loss.

Click once and inspect the before/after losses. Both parameters change. Open the arithmetic to show gradients and the update rule: new parameter = old parameter − learning rate × gradient.

Explain the terms in order:

- **Loss:** a number measuring prediction error.
- **Gradient:** how the loss changes with a parameter.
- **Backpropagation:** calculates gradients through layers.
- **Optimizer:** uses those gradients to change parameters.
- **Learning rate:** controls update size.
- **Batch:** a group of training examples.
- **Epoch:** one pass over all training examples.
- **Inference:** predicting with learned parameters fixed.

The toy uses simple gradient descent; the full model uses Adam. Improving on one example does not establish performance on new examples. Use the final quiz to check that the audience knows what changes during learning.

## Connect it to the real model · 8–12 minutes

Open **02 · Live network**. Load a held-out example and inspect its scaling, actual activations, weights, bias, and per-neuron calculations. Reveal the layers in order. Change one customer input, then change the decision threshold: the former can change the probability, while the latter only changes the decision.

Open **03 · Training & evaluation**. Show training versus validation loss, unseen holdout metrics, false alarms and missed churners, and the simpler baselines. Be explicit that accuracy falls from about 92.0% on the internal holdout to 58.7% on the separate test file. Dataset provenance and calibration are unverified; these are educational results, not a deployment claim.

Open **04 · The code**. Map the diagram to Linear, ReLU, and Sigmoid. Walk through forward, loss, backward, and optimizer step. The guided notebook contains the full implementation and saved evaluation outputs.

## Setup

The saved model is ready; no retraining is needed for this revised lesson. From the project root, run `.venv\Scripts\python.exe -m streamlit run src/neural_networks/app.py` if the app is not already open. The fundamentals experiments run locally and do not modify the model weights.
