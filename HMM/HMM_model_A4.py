import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the combined feature dataset
df = pd.read_csv("reduced_features_combined.csv")

# Drop 'observation' if it already exists to avoid duplication
if "observation" in df.columns:
    df = df.drop(columns=["observation"])

# Split the dataset BEFORE KMeans
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Run KMeans on training data only
kmeans = KMeans(n_clusters=4, random_state=42)
train_observations = kmeans.fit_predict(X_train)
test_observations = kmeans.predict(X_test)

# Create DataFrames with observations and labels
train_df = X_train.copy()
train_df["observation"] = train_observations
train_df["label"] = y_train.values

test_df = X_test.copy()
test_df["observation"] = test_observations
test_df["label"] = y_test.values
print("\nDistribution of observations by label:")
print(train_df.groupby("label")["observation"].value_counts())

# Define states and observations
states = df["label"].unique().tolist()
observations = train_df["observation"].unique().tolist()
n_states = len(states)
n_observations = len(observations)

# Map state and observation values to indices
state_to_idx = {state: i for i, state in enumerate(states)}
idx_to_state = {i: state for state, i in state_to_idx.items()}
obs_to_idx = {obs: i for i, obs in enumerate(observations)}

# Initialize HMM parameters: π, A, B
pi = np.zeros(n_states)
A = np.zeros((n_states, n_states))
B = np.zeros((n_states, n_observations))

# Fill π, A, B using training set
prev_state_idx = None
for i, row in train_df.iterrows():
    state_idx = state_to_idx[row["label"]]
    obs_idx = obs_to_idx[row["observation"]]

    if i == 0:
        pi[state_idx] += 1
    if prev_state_idx is not None:
        A[prev_state_idx][state_idx] += 1
    prev_state_idx = state_idx

    B[state_idx][obs_idx] += 1

# Handle case where pi is all zeros to avoid division by zero
if pi.sum() == 0:
    print("Warning: pi is all zeros. Initializing with uniform distribution.")
    pi += 1

# Normalize matrices with Laplace smoothing
epsilon = 1e-6
pi = pi / pi.sum()
A = (A + epsilon) / (A + epsilon).sum(axis=1, keepdims=True)
B = (B + epsilon) / (B + epsilon).sum(axis=1, keepdims=True)

# Print the learned HMM parameters
print("\nπ (initial probabilities):", pi)
print("A (transition matrix):\n", A)
print("B (emission matrix):\n", B)

# Viterbi algorithm implementation
def viterbi(pi, A, B, observations):
    N = A.shape[0]
    T = len(observations)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0] = pi * B[:, observations[0]]

    for t in range(1, T):
        for s in range(N):
            transition_probs = delta[t - 1] * A[:, s]
            best_prev_state = np.argmax(transition_probs)
            delta[t, s] = transition_probs[best_prev_state] * B[s, observations[t]]
            psi[t, s] = best_prev_state

    best_path = [np.argmax(delta[T - 1])]
    for t in range(T - 2, -1, -1):
        best_path.insert(0, psi[t + 1, best_path[0]])

    return best_path, np.max(delta[T - 1])

# Run Viterbi on the test observation sequence
obs_seq = test_df["observation"].map(obs_to_idx).tolist()
viterbi_path, viterbi_prob = viterbi(pi, A, B, obs_seq)

# Evaluate accuracy
true_labels = test_df["label"].map(state_to_idx).tolist()
accuracy = accuracy_score(true_labels, viterbi_path)

# Print results
print("\nMost likely state sequence (first 10):", viterbi_path[:10])
print("Probability of this sequence:", viterbi_prob)
print(f"HMM Accuracy on test set: {accuracy:.3f}")

# ----------------- 🎯 ADDITION: Confusion Matrix Plot -----------------
# Create confusion matrix
cm = confusion_matrix(true_labels, viterbi_path)

# Plot it
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=states, yticklabels=states)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - HMM Predictions vs True Labels")
plt.tight_layout()
plt.show()
# ---------------------------------------------------------------------
