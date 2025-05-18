# Simplified Bayesian Regression vs Neural Network
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----- Emotion to Feature Mapping -----
emotions = {
    "calm": [2, 8, 2],
    "happy": [9, 7, 1],
    "angry": [10, 3, 4],
    "tired": [3, 5, 9],
    "frustrated": [8, 5, 6]
}

# ----- Feature Function (no sin/cos for simplicity) -----
def phi(x_vec):
    return np.array([1.0, *x_vec])

# ----- Training Data (Features + Frequencies) -----
X_vecs = np.array([
    [2, 8, 2],   # calm
    [4, 7, 3],
    [6, 6, 2],
    [8, 5, 1],   # happy
    [10, 3, 4],  # angry
    [3, 5, 9],   # tired
    [5, 6, 5],
    [7, 4, 2],
    [9, 2, 3]
])
Y = np.array([261.63, 329.63, 392.0, 523.25, 880.0, 196.00, 350.0, 470.0, 700.0])

# ----- Build Design Matrix for Bayesian Regression -----
Phi = np.vstack([phi(x) for x in X_vecs]).T
num_features = Phi.shape[0]

mu_prior = np.zeros(num_features)
Sigma_prior = np.eye(num_features)
sigma2 = 25.0

precision_post = np.linalg.inv(Sigma_prior) + (1 / sigma2) * (Phi @ Phi.T)
Sigma_post = np.linalg.inv(precision_post)
mu_post = Sigma_post @ ((1 / sigma2) * Phi @ Y)

# ----- Train/Test Split -----
X_train, X_test, Y_train, Y_test = train_test_split(X_vecs, Y, test_size=0.4, random_state=42)

# ----- Neural Network Model -----
model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_losses, test_losses = [], []

for epoch in range(2000):
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

# ----- Predictions for Plot -----
x1_range = np.linspace(1, 10, 100)
x_plot_vecs = np.array([[x, 5, 3] for x in x1_range])
phi_test = np.vstack([phi(x) for x in x_plot_vecs])
mu_fx = phi_test @ mu_post
x_plot_tensor = torch.tensor(x_plot_vecs, dtype=torch.float32)
y_test_pred_nn = model(x_plot_tensor).detach().numpy()

# ----- Plot Results -----
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x1_range, y_test_pred_nn, label="Neural Network", color='red')
plt.plot(x1_range, mu_fx, label="Bayesian Regression", color='green')
plt.scatter(X_vecs[:, 0], Y, color='blue', label="Training Data")
plt.xlabel("x1 (e.g. Intensity)")
plt.ylabel("Frequency [Hz]")
plt.title("Model Predictions")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Neural Network Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- MSE Comparison on Test Set -----
phi_test_points = np.vstack([phi(x) for x in X_test])
bayesian_preds = phi_test_points @ mu_post
nn_preds = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()

mse_bayes = mean_squared_error(Y_test, bayesian_preds)
mse_nn = mean_squared_error(Y_test, nn_preds)

print(f"Bayesian Regression MSE on Test Set: {mse_bayes:.2f}")
print(f"Neural Network MSE on Test Set: {mse_nn:.2f}")

# ----- Emotion Prediction to Sound -----
emotion = input("Enter emotion (calm, happy, angry, tired, frustrated): ").strip().lower()
if emotion not in emotions:
    raise ValueError("Invalid emotion.")
x_star = np.array(emotions[emotion])
freq_pred = phi(x_star) @ mu_post

print(f"Emotion: {emotion}, Predicted Frequency: {freq_pred:.2f} Hz")
fs = 44100
t = np.linspace(0, 1.0, int(fs * 1.0), endpoint=False)
wav = 0.5 * np.sin(2 * np.pi * freq_pred * t)
sd.play(wav, fs)
sd.wait()
