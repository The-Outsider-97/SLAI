# Adaptive Model Modules

The `modules` package contains reusable trainable model implementations used by SLAI's Adaptive agent subsystem.

These modules implement model behavior only. Agent orchestration, memory lifecycle, policy coordination, imitation-learning workflows, meta-learning and parameter tuning remain in `src/agents/adaptive/`.

## Architecture

```text
adaptive workers / memory
          |
          v
      modules/
          |
          v
       utils/
```

## Public API

```python
from src.agents.adaptive.modules import (
    NeuralNetwork,
    BayesianDQN,
    ActorCriticNetwork,
    SGDRegressor,
)
```

## Neural Network
NeuralNetwork is a configurable PyTorch feed-forward neural network.

### Processing flow

```text
input
  |
  v
input validation
  |
  v
hidden linear layers
  |
  +-- activation
  +-- optional batch normalization
  +-- optional dropout
  |
  v
output layer
  |
  v
logits / regression output
```

Architecture is configured through the neural_network section of *adaptive_config.yaml*.

The network supports:
- regression;
- binary classification;
- multiclass classification;
- Adam, AdamW, SGD and RMSprop optimizers;
- MSE, MAE, Huber, cross-entropy and BCE-with-logits losses;
- gradient clipping;
- StepLR, cosine annealing and ReduceLROnPlateau;
- validation;
- early stopping;
- checkpoint persistence;
- model diagnostics.

### Output contract

Regression uses continuous outputs.

Binary classification uses either:

one logit with BCEWithLogitsLoss; or two logits with CrossEntropyLoss.

Multiclass classification uses class logits with CrossEntropyLoss.

Training losses receive logits whenever the chosen loss expects logits.
Probability activations are applied only for prediction APIs.

## BayesianDQN

*BayesianDQN* extends *NeuralNetwork* with Monte Carlo dropout.

During uncertainty inference:

1. the model remains in evaluation mode;
2. dropout modules are temporarily enabled;
3. multiple forward passes are collected;
4. prediction mean, variance and standard deviation are calculated;
5. the configured uncertainty threshold is used for classification/flagging, not to truncate the measured uncertainty.

MC statistics are computed in the same output domain used by the corresponding prediction API.

## ActorCriticNetwork

*ActorCriticNetwork* implements the trainable policy/value model used by *SkillWorker*.

```text
                     +--> actor --> action distribution
state --> optional --|
         shared base +--> critic --> state value
```

For discrete actions the actor produces logits for a categorical distribution.

For continuous actions the actor produces action means and maintains a trainable standard deviation constrained by configured minimum and maximum bounds.

The network owns:

actor and critic forward passes;
action-distribution construction;
action sampling;
log-probability calculation;
entropy calculation;
action evaluation;
model parameter grouping.

*SkillWorker* owns RL objectives, trajectory handling, GAE, optimization and memory interaction.

When *shared_base=True*, shared parameters must be optimized through one coordinated optimizer step. They must not be independently owned by both an actor optimizer and a critic optimizer.

## SGDRegressor

`SGDRegressor` is a lightweight online linear regression model used by
Adaptive Memory for incremental parameter-impact and intervention learning.

### Prediction model

For an input vector $\mathbf{x}$, coefficient vector $\mathbf{w}$, and
intercept $b$, the prediction is:

$$
\hat{y} = \mathbf{w}^{T}\mathbf{x} + b
$$

For squared-error regression, the prediction error is:

$$
e = \hat{y} - y
$$

where:

- $\hat{y}$ is the predicted value;
- $y$ is the target value;
- $e$ is the prediction error.

With L2 regularization, the coefficient gradient is:

$$
\nabla_{\mathbf{w}} = e\mathbf{x} + \alpha\mathbf{w}
$$

where $\alpha$ controls the strength of the L2 regularization penalty.

If an intercept is enabled, its gradient is:

$$
\nabla_b = e
$$

Sample weighting scales the contribution of individual observations to the
data-dependent update, while L2 regularization remains a global model
penalty.

---

### Learning-rate schedules

`SGDRegressor` supports three learning-rate schedules.

#### `constant`

The learning rate remains fixed at the configured initial learning rate:

$$
\eta_t = \eta_0
$$

where:

- $\eta_t$ is the learning rate at training step $t$;
- $\eta_0$ is the configured initial learning rate.

#### `invscaling`

The learning rate decreases as training progresses:

$$
\eta_t =
\frac{\eta_0}
{(t + 1)^{\text{power}_t}}
$$

The rate of decay is controlled by `power_t`.

#### `adaptive`

The learning rate begins at `eta0` and is reduced when the loss fails to
improve sufficiently for the configured number of epochs.

The adaptive schedule is controlled by:

- `adaptive_patience`
- `adaptive_tolerance`
- `adaptive_decay`
- `lr_decay_frequency`
- `lr_decay_factor`

All learning-rate schedules respect the configured lower bound:

```text
min_learning_rate
```
so the effective learning rate cannot decay below that value.