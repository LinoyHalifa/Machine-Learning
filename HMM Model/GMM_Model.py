import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("reduced_features_combined.csv")
df = df.dropna()
X = df.drop("label", axis=1)
y = df["label"]

# Fit GMM with 4 clusters
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X)
cluster_labels = gmm.predict(X)

# Step 1: Create comparison table between true labels and clusters
df_compare = pd.DataFrame({
    "TrueLabel": y,
    "Cluster": cluster_labels
})
conf_matrix = pd.crosstab(df_compare["Cluster"], df_compare["TrueLabel"])

# Step 2: Map each cluster to its most common true label
cluster_to_label = conf_matrix.idxmax(axis=1).to_dict()

# Step 3: Convert predicted clusters to predicted labels
predicted_labels = df_compare["Cluster"].map(cluster_to_label)

# Step 4: Compute Precision, Recall, F1-score
precision = precision_score(y, predicted_labels, average=None, labels=conf_matrix.columns)
recall = recall_score(y, predicted_labels, average=None, labels=conf_matrix.columns)
f1 = f1_score(y, predicted_labels, average=None, labels=conf_matrix.columns)

# Create metrics summary table
metrics_df = pd.DataFrame({
    "Label": conf_matrix.columns,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})

# Print evaluation results
print("🔍 GMM cluster-to-label mapping:")
print(conf_matrix)
print("\n📊 Performance metrics per label:")
print(metrics_df)

# ----------------- 🔵 Visualization -----------------
# Reduce data to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get GMM parameters (means and covariances)
means = gmm.means_
covariances = gmm.covariances_

# Project means and covariances into PCA space
projected_means = pca.transform(means)
projected_covs = []
for cov in covariances:
    projected_cov = pca.components_ @ cov @ pca.components_.T
    projected_covs.append(projected_cov)

# Plotting
plt.figure(figsize=(10, 6))
colors = ["red", "green", "blue", "orange"]

for i in range(4):
    plt.scatter(
        X_pca[cluster_labels == i, 0],
        X_pca[cluster_labels == i, 1],
        label=f"Cluster {i}",
        alpha=0.5,
        s=30,
        color=colors[i]
    )

    # Draw GMM ellipse for each cluster
    mean = projected_means[i]
    cov = projected_covs[i]
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = mpl.patches.Ellipse(xy=mean,
                               width=lambda_[0]*4, height=lambda_[1]*4,
                               angle=np.rad2deg(np.arccos(v[0, 0])),
                               color=colors[i], alpha=0.3)
    plt.gca().add_patch(ell)

plt.title("GMM Clustering Visualization (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ----------------------------------------------------
