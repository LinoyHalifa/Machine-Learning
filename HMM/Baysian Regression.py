# Small Neural Network with Overfitting + Bayesian Regression Comparison

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd
from sklearn.model_selection import train_test_split

# ----- EMOTION TO FEATURE MAPPING -----
emotions = {
    "calm": [2, 8, 2],
    "happy": [9, 7, 1],
    "angry": [10, 3, 4],
    "tired": [3, 5, 9],
    "frustrated": [8, 5, 6]
}

# Get emotion from user
emotion = input("Enter emotion (calm, happy, angry, tired, frustrated): ").strip().lower()
if emotion not in emotions:
    raise ValueError("Emotion not recognized")

# Use full feature vector as input
x_star_vec = np.array(emotions[emotion])

# ----- DATA SETUP -----
# Expanded training data with real frequencies
X_vecs = np.array([
    [2, 8, 2],   # calm
    [4, 7, 3],   # slightly happy
    [6, 6, 2],   # energetic
    [8, 5, 1],   # happy
    [10, 3, 4],  # angry
    [3, 5, 9],   # tired
    [5, 6, 5],
    [7, 4, 2],
    [9, 2, 3],
])

Y = np.array([261.63, 329.63, 392.0, 523.25, 880.0, 196.00, 350.0, 470.0, 700.0])  # Hz

# Feature function: [1, x1, x2, x3, sin(x1), cos(x2), sin(x3)]
def phi(x_vec):
    return np.array([1.0, x_vec[0], x_vec[1], x_vec[2], np.sin(x_vec[0]), np.cos(x_vec[1]), np.sin(x_vec[2])])

# Build design matrix Phi
Phi = np.vstack([phi(x) for x in X_vecs]).T  # shape: (num_features, N)

# Prior parameters
num_features = Phi.shape[0]
mu_prior = np.zeros(num_features)
Sigma_prior = np.eye(num_features)
sigma2 = 25.0  # observation noise variance

# Posterior computation
precision_post = np.linalg.inv(Sigma_prior) + (1 / sigma2) * (Phi @ Phi.T)
Sigma_post = np.linalg.inv(precision_post)
mu_post = Sigma_post @ ((1 / sigma2) * Phi @ Y)

# Predictive mean for test inputs
x_test_raw = np.linspace(1, 10, 100)
x_test_vecs = np.array([[x, 5, 3] for x in x_test_raw])
phi_test = np.vstack([phi(x) for x in x_test_vecs])
mu_fx = phi_test @ mu_post

# ----- SPLIT FOR OVERFITTING DEMO (NEURAL NETWORK) -----
X_train, X_test, Y_train, Y_test = train_test_split(X_vecs, Y, test_size=0.4, random_state=42)

# Convert to PyTorch tensors (using full features)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32)

# Neural network with too many parameters (to encourage overfitting)
model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train neural network
num_epochs = 2000
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, Y_train_tensor)
    train_losses.append(loss.item())

    model.eval()
    y_test_pred = model(X_test_tensor)
    test_loss = loss_fn(y_test_pred, Y_test_tensor)
    test_losses.append(test_loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict for plotting
x1_range = np.linspace(1, 10, 100)
x_plot_tensor = torch.tensor([[x, 5, 3] for x in x1_range], dtype=torch.float32)
y_test_pred_nn = model(x_plot_tensor).detach().numpy()

# ----- PLOT -----
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x1_range, y_test_pred_nn, label="Neural Network Prediction", color='r')
plt.plot(x1_range, mu_fx, label="Bayesian Prediction", color='g')
plt.scatter(X_vecs[:, 0], Y, color='blue', label="Training Data", zorder=5)
plt.title("Neural Network (Overfitting) vs Bayesian Regression")
plt.xlabel("Feature x1 (e.g. Intensity)")
plt.ylabel("Output Y (Frequency in Hz)")
plt.legend()
plt.grid(True)

# Plot loss curves
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Neural Network Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- PLAY SOUND OF BAYESIAN PREDICTION -----
phi_star = phi(x_star_vec)
predicted_freq = phi_star @ mu_post

fs = 44100
duration = 1.0

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
wave = 0.5 * np.sin(2 * np.pi * predicted_freq * t)

print(f"Emotion: {emotion}, Predicted Frequency: {predicted_freq:.2f} Hz")
sd.play(wave, fs)
sd.wait()
from sklearn.metrics import mean_squared_error

# ----- Bayesian Prediction on X_test -----
phi_test_points = np.vstack([phi(x) for x in X_test])
bayesian_preds = phi_test_points @ mu_post

# ----- Neural Network Prediction on X_test -----
nn_preds = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()

# ----- MSE Comparison -----
mse_bayes = mean_squared_error(Y_test, bayesian_preds)
mse_nn = mean_squared_error(Y_test, nn_preds)

print(f"Bayesian Regression MSE on Test Set: {mse_bayes:.2f}")
print(f"Neural Network MSE on Test Set: {mse_nn:.2f}")

