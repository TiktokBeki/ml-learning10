# ml-learning10
Page 1 — Course Overview & Goal

Goal: Learn Python → NumPy → PyTorch → sklearn → Neural Networks.

Mini-project hints:

Daily learning log (CSV).

CLI progress tracker.

Plot progress with matplotlib.

Page 2 — Functions & Arguments def greet(name, greeting="Hello"): print(f"{greeting}, {name}!")

greet("Beki") greet("Beki", greeting="Hi")

Projects:

Sum, mean, median function.

CLI calculator.

Optional parameters: skip negatives, rounding.

Page 3 — Loops numbers = [1,2,3,4,5] for n in numbers: print(n**2)

Projects:

Multiplication table generator.

Filter even numbers into a new list.

Batch normalization loop.

Page 4 — Lists, Tuples, Dictionaries fruits = ["apple","banana","cherry"] fruits.append("orange") coords = (10,20) student = {"name":"Beki","age":11}

Projects:

Store daily logs in dictionary.

List of datasets for preprocessing.

Convert CSV to dictionary format.

Page 5 — Conditionals x = 10 if x > 5: print("x is big") elif x == 5: print("x is five") else: print("x is small")

Projects:

Categorize numbers small/medium/large.

Filter dataset rows.

CLI guessing game.

Page 6 — Nested Loops & List Comprehensions matrix = [[i*j for j in range(3)] for i in range(3)]

Projects:

Generate multiplication matrix.

Create feature matrices for ML.

Flatten matrix with comprehension.

**Page 7 — Functions: *args, kwargs def sum_all(*args, **kwargs): total = sum(args) print("Total:", total) if kwargs.get("verbose", False): print("Numbers:", args)

Projects:

Flexible dataset normalization function.

Logging wrapper using **kwargs.

Extend CLI calculator to multiple inputs.

Page 8 — File Handling with open("data.csv", "r") as f: lines = f.readlines()

Projects:

Read CSV and print first 5 rows.

Write normalized dataset to CSV.

Implement daily progress tracker file system.

Page 9 — Python OOP: Basics class Person: def init(self, name): self.name = name def greet(self): print("Hello,", self.name)

Projects:

Dataset class for ML.

Logger class for training.

CLI app objects.

Page 10 — Python OOP: Methods & Attributes p = Person("Beki") p.greet() p.age = 11

Projects:

Extend Dataset with split method.

Add normalization as a method.

Track stats with object attributes.

Page 11 — Python: Decorators def log_call(func): def wrapper(*args, **kwargs): print("Calling", func.name) return func(*args, **kwargs) return wrapper

@log_call def add(a,b): return a+b

Projects:

Log Dataset calls.

Log training loop functions.

Track preprocessing steps automatically.

Page 12 — Python: Context Managers class Timer: def enter(self): import time self.start = time.time() return self def exit(self, exc_type, exc_val, exc_tb): import time print("Elapsed:", time.time() - self.start)

with Timer(): x = sum(range(100000))

Projects:

Time preprocessing steps.

Time model training.

Measure forward pass speed.

Page 13 — NumPy Basics: Arrays import numpy as np X = np.array([[1,2],[3,4]]) y = np.array([1,0])

Projects:

Convert Dataset lists to NumPy arrays.

Prepare input matrix for ML.

Batch operations with arrays.

Page 14 — NumPy: Vectorization weights = np.array([0.2,0.5]) bias = 0.1 y_pred = X @ weights + bias

Projects:

Manual linear regression.

Batch predictions for mini-batches.

Combine multiple datasets using vectorized operations.

Page 15 — NumPy: Indexing & Boolean Masks mask = X[:,0] > 2 filtered = X[mask]

Projects:

Filter samples by label.

Mini-batch selection.

Conditional feature selection.

Page 16 — NumPy: Shapes & Reshape X = np.arange(12).reshape(3,4) X_T = X.T

Projects:

Prepare input for linear layers.

Flatten images for fully connected network.

Batch reshaping for neural network input.

Page 17 — NumPy: Random Numbers np.random.seed(42) weights = np.random.randn(3,1)

Projects:

Initialize network weights.

Compare different seeds on network output.

Random dataset splits.

Page 18 — Python Logging import logging logging.basicConfig(level=logging.INFO) logging.info("Training started")

Projects:

Log batch loss per epoch.

Save preprocessing logs.

Log experiment hyperparameters.

Page 19 — PyTorch: Tensors import torch x = torch.tensor([[1,2],[3,4]], dtype=torch.float32) y = x * 2

Projects:

Convert NumPy dataset to tensor.

Apply simple linear layer.

Experiment with requires_grad=True.

Page 20 — PyTorch: Autograd x = torch.tensor([2.0,3.0], requires_grad=True) y = x**2 z = y.sum() z.backward() print(x.grad)

Projects:

Manual gradient descent.

Compute gradients for batch loss.

Experiment with small functions before neural network.

Page 21 — PyTorch: Linear Layers import torch.nn as nn layer = nn.Linear(3,1) output = layer(torch.randn(5,3))

Projects:

Build small feedforward network.

Chain two linear layers with activation.

Test forward pass on random inputs.

Page 22 — PyTorch: Loss Functions loss_fn = nn.MSELoss() pred = torch.tensor([0.5,0.8]) target = torch.tensor([0.6,1.0]) loss = loss_fn(pred, target)

Projects:

Compare MSE vs L1 loss.

Compute batch loss.

Track loss for mini-batches.

Page 23 — PyTorch: Optimizers optimizer = torch.optim.SGD(layer.parameters(), lr=0.01) optimizer.zero_grad() loss.backward() optimizer.step()

Projects:

Wrap optimizer in training loop.

Compare different learning rates.

Try Adam optimizer.

Page 24 — PyTorch: Training Loop Skeleton for epoch in range(100): optimizer.zero_grad() y_pred = model(X) loss = loss_fn(y_pred, y) loss.backward() optimizer.step()

Projects:

Add validation loss per epoch.

Store weights after each epoch.

Experiment with early stopping.

Page 25 — PyTorch: Saving & Loading Models torch.save(model.state_dict(), "model.pth") model.load_state_dict(torch.load("model.pth"))

Projects:

Train small network, save and reload.

Continue training from saved weights.

Compare models with different hyperparameters.

Page 26 — PyTorch: Activation Functions import torch.nn.functional as F x = torch.randn(5) relu = F.relu(x) sigmoid = torch.sigmoid(x)

Projects:

Use ReLU in hidden layers, Sigmoid in output.

Compare activations on same input.

Create small MLP with ReLU.

Page 27 — Sklearn: Linear Regression from sklearn.linear_model import LinearRegression model = LinearRegression() model.fit(X_numpy, y_numpy) pred = model.predict(X_numpy)

Projects:

Compare manual vs sklearn regression.

Evaluate metrics: MSE, R2.

Add preprocessing steps: scaling.

Page 28 — Sklearn: Train/Test Split & Pipelines from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.2)

Projects:

Pipeline: scaling + linear regression.

Experiment with different splits.

Extend to polynomial regression.

Page 29 — Sklearn: Feature Scaling & Encoding from sklearn.preprocessing import StandardScaler, OneHotEncoder

Projects:

Normalize numeric features.

Encode categorical features.

Combine in pipeline for final dataset.

Page 30 — PyTorch: Dataset & DataLoader from torch.utils.data import Dataset, DataLoader

Projects:

Custom Dataset class for CSV.

Batch loading with DataLoader.

Shuffle and mini-batch data.

Page 31 — PyTorch: Forward Pass

Projects:

Single-layer linear model.

Multi-layer perceptron.

Print outputs at each layer for debugging.

Page 32 — PyTorch: Backward Pass & Gradients

Projects:

Manual weight updates.

Compute gradient norms.

Compare autograd vs manual gradient.

Page 33 — PyTorch: Batch Training

Projects:

Mini-batch SGD.

Track batch loss.

Experiment with batch sizes.

Page 34 — PyTorch: Regularization nn.Dropout(0.5)

Projects:

Add dropout to MLP.

Compare with/without dropout.

Implement L2 regularization manually.

Page 35 — PyTorch: Optimizer Experiments

Projects:

Compare SGD, Adam, RMSProp.

Tune learning rate.

Observe convergence speed.

Page 36 — PyTorch: Evaluation Metrics

Projects:

Compute accuracy, precision, recall.

Track metrics per epoch.

Visualize metrics with matplotlib.

Page 37 — PyTorch: Saving & Loading Checkpoints

Projects:

Save intermediate weights.

Resume training.

Compare saved checkpoints.

Page 38 — PyTorch: Advanced Layer Types

Projects:

Add BatchNorm layer.

Experiment with different activations.

Combine layers to form deeper network.

Page 39 — PyTorch: Learning Rate Scheduling

Projects:

ReduceLROnPlateau.

StepLR.

Observe effects on loss.

Page 40 — PyTorch: Multi-class Classification

Projects:

CrossEntropyLoss.

Softmax output.

Track per-class accuracy.

Page 41 — PyTorch: Saving Entire Model torch.save(model, "full_model.pth") model = torch.load("full_model.pth")

Projects:

Save full model + inference.

Compare with state_dict approach.

Deploy model in small script.

Page 42 — PyTorch: Tensor Operations

Projects:

Matrix multiplication, transpose.

Elementwise ops.

Batch matrix operations for forward pass.

Page 43 — PyTorch: Debugging Tips

Projects:

Print gradients.

Check layer outputs.

Compare manual vs PyTorch outputs.

Page 44 — PyTorch: Combining Datasets

Projects:

Merge multiple CSV datasets.

Shuffle combined dataset.

Mini-batch from combined dataset.

Page 45 — PyTorch: Full Training Loop Skeleton

Projects:

Train network with train/test split.

Save metrics.

Plot losses per epoch.

Page 46 — Mini Final Project Prep

Hint:

Use all previous lessons: Dataset class + preprocessing + linear + MLP layers + optimizer + loss + evaluation metrics.

Divide your 500-line project into modules: data, model, train, evaluate.

Mini-project hints:

Small MLP on CSV data.

Add validation split and early stopping.

Add metric logging.

Page 47 — Final Project Modules

Modules to implement:

Data loading + preprocessing

Model definition (layers, activations)

Training loop (optimizer, loss)

Evaluation & metrics

Saving and checkpointing

Projects:

Each module can be tested independently.

Combine progressively to reach 500-line project.

Page 48 — Advanced Tips for Flexibility

Hints:

Keep preprocessing modular.

Parameterize model size, batch size, learning rate.

Use functions and classes to reuse code for different datasets.

Mini-project hints:

Add command-line arguments to choose dataset or model depth.

Write a config file to store hyperparameters.

Switch between SGD and Adam dynamically.

Page 49 — Preparing for Experiments

Hints:

Try different architectures (hidden layers, units).

Record metrics for comparison.

Save best performing model.

Mini-project hints:

Compare 2-layer vs 3-layer MLP.

Try dropout on hidden layers.

Track learning curves.

Page 50 — 500-line Final Project Roadmap

Step-by-step:

Dataset Module: Load CSV, normalize features, split train/test, create PyTorch Dataset + DataLoader.

Model Module: Define MLP with configurable layers, activations, and dropout.

Training Module: Forward pass, loss, backward pass, optimizer step, batch loop, epoch loop.

Evaluation Module: Compute accuracy, metrics, validation loss.

Logging Module: Save metrics, print progress, save model checkpoints.

Flexibility Tips: Parametrize everything: batch size, learning rate, layer sizes, activation types, dataset path.

Mini-project Hints:

Train MLP on multiple datasets.

Experiment with different optimizers and learning rates.

Add early stopping.

Log metrics and plot learning curves.

By combining all previous lessons, you can build your 500-line neural network project. Each module is reusable, testable, and adaptable to any dataset.
