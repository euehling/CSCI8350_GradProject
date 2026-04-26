import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ── Data Loading ──────────────────────────────────────────────────────────────
data = Path("Data/paddydataset.csv")
df = pd.read_csv(data)
df.columns = df.columns.str.strip()

# ── Create constants, variables ────────────────────────────────────────────────
random_state = 42

# ── Build Modeling Dataset ────────────────────────────────────────────────────
df_for_clustering = df.copy()

# ──  Drop duplicate rows ────────────────────────────────────────────────────
df_for_clustering = df_for_clustering.drop_duplicates()
print("Shape after dropping duplicates:", df_for_clustering.shape)

# ── Create yield per hectare, drop raw yield and correlated columns ───────────
df_for_clustering['Yield per Hectare'] = df_for_clustering['Paddy yield(in Kg)'] / df_for_clustering['Hectares']
df_for_clustering = df_for_clustering.drop(columns=[
    'LP_Mainfield(in Tonnes)',
    'LP_nurseryarea(in Tonnes)',
    'Nursery area (Cents)',
    'Paddy yield(in Kg)'
])

# ── Identify and print lists of numeric and categorical features ───────────
numeric_features = df_for_clustering.select_dtypes(include=["number"]).columns.tolist()
categorical_features = df_for_clustering.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# ── Set up pipeline for encoding, drop features with variance = 0.0 ───────────
preprocessor = Pipeline(steps=[
    ("column_transform", ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ])),
    ("var_threshold", VarianceThreshold(threshold=0.0))
])

print("Preprocessing pipeline:", preprocessor)


# ── Cumulative Explained Variance Plot ────────────────────────────────────────
# First preprocess the data
X_preprocessed = preprocessor.fit_transform(df_for_clustering)

# Scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed)

# Fit PCA with all components
pca_full = PCA(random_state=random_state)
pca_full.fit(X_scaled)

# Plot cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', markersize=3)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
plt.axhline(y=0.80, color='g', linestyle='--', label='80% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.legend()
plt.show()

# Print exact number of components needed
n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components needed for 80% variance: {n_components_80}")
print(f"Components needed for 90% variance: {n_components_90}")


# ── Set up pipeline for scaler and PCA to reduce dimensionality ───────────
scaler_with_PCA = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('pca', PCA(n_components=7, random_state=random_state))]) #changed to 7

print("Scaling and PCA pipeline:", scaler_with_PCA)

# ── Create transformed dataset for later use with silhouette ────────────────
print("Created transformed dataset after scaler and PCA, but prior to K-means.")
X_transformed = scaler_with_PCA.fit_transform(X_preprocessed)
print(X_transformed)

# ── Set up pipeline for k-means ─────────────────────────────────────────────
k_means = Pipeline(steps=[('kmeans', KMeans(n_clusters=3,
                                            init='k-means++',
                                            n_init=10,
                                            max_iter=300,
                                            tol=0.0001,
                                            random_state=random_state,
                                            algorithm='elkan')
                           )])
print("K-Means pipeline:", k_means)

# ── Set up full pipeline for k-means ─────────────────────────────────────────────
k_means_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                   ("scaler_PCA", scaler_with_PCA),
                                   ("kmeans", k_means)
                                   ])

# ── Test fit k-means to check for errors ────────────────────────────────────────────────────────────
k_means_pipeline.fit(df_for_clustering)
# clusters = k_means_pipeline.predict(df_for_clustering)

# ── Deciding K value ───────────────────────────────────────────────────────────────
inertia = []
silhouette_scores = []

# Set range of K to try
# NOTE: expect algorithm warnings (lloyd instead of elkan)
#   and silhouette warnings if range includes 1 cluster
k_range = range(2, 15)

for k in k_range:
    # ── Full pipeline for k-means ──
    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                       ("scaler_PCA", scaler_with_PCA),
                                       ("kmeans", KMeans(n_clusters=k,
                                                         init='k-means++',
                                                         n_init=10,
                                                         max_iter=300,
                                                         tol=0.0001,
                                                         random_state=random_state,
                                                         algorithm='elkan')
                                        )])
    pipeline.fit(df_for_clustering)
    inertia.append(pipeline.named_steps["kmeans"].inertia_)
# NOTE: Use the preprocessed data (after PCA) as input to the silhouette_score function
    score = silhouette_score(X_transformed, pipeline.named_steps["kmeans"].labels_)
    silhouette_scores.append(score)

# ── Plot Inertia ─────────────────────────────────────────────────────────
plt.figure(1, figsize=(15, 6))
plt.plot(k_range, inertia, 'o')
plt.plot(k_range, inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()

# ── Plot Silhouette Scores ───────────────────────────────────────────────
# More complex example at https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
plt.figure(1, figsize=(15, 6))
plt.plot(k_range, silhouette_scores, 'o')
plt.plot(k_range, silhouette_scores, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Silhouette Score')
plt.show()

# ── Final pipeline for k-means ──
final_k = 6
final_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                   ("scaler_PCA", scaler_with_PCA),
                                   ("kmeans", KMeans(n_clusters=final_k,
                                                     init='k-means++',
                                                     n_init=10,
                                                     max_iter=300,
                                                     tol=0.0001,
                                                     random_state=random_state,
                                                     algorithm='lloyd')
                                    )])
final_pipeline.fit(df_for_clustering)

#  ── Review PCA values ──────────────────────────────────────────────────
# Since scaler_PCA is a pipeline in a pipeline, we have to drill down
#   from scaler_PCA --> pca to get values such as number of components and explained variance
#   see "Attributes" section of https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = final_pipeline.named_steps["scaler_PCA"].named_steps["pca"]
pca_comp = pca.n_components_
pca_explained_variance = pca.explained_variance_ratio_.sum()
print("Number of PCA components:", pca_comp)
print("PCA explained variance:", pca_explained_variance)

# Potential source of code for plotting results with PCA
# https://www.geeksforgeeks.org/machine-learning/kmeans-clustering-and-pca-on-wine-dataset/


#  ── Cluster Visualization Using PCA ──────────────────────────────────────────────────
# Get cluster labels from the pipeline
cluster_labels = final_pipeline.predict(df_for_clustering)

# Use first 2 PCA components for visual
pca_2d = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=2, random_state=random_state))])

X_2d = pca_2d.fit_transform(X_preprocessed)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'K-Means Clusters (k={final_k}) Visualized with PCA')
plt.show()

# ── Cluster Interpretation ────────────────────────────────────────────────────
"""
Cluster visualization shows well seperated, pill shaped clusters which suggests the groupinmgs
are being driven by categorical variables. To confirm, we look at the average numeric feature values
and the most common categorical values within each cluster
"""

df_for_clustering['Cluster'] = cluster_labels

print("\n=== Cluster Sizes ===")
print(df_for_clustering['Cluster'].value_counts().sort_index())

print("\n=== Cluster Profiles (Numeric Features) ===")
print(df_for_clustering.groupby('Cluster')[numeric_features].mean().round(2))

print("\n=== Cluster Profiles (Categorical Features) ===")
for col in categorical_features:
    print(f"\n-- {col} --")
    print(df_for_clustering.groupby('Cluster')[col].agg(lambda x: x.value_counts().index[0]))

# ── Cluster Descriptions ──────────────────────────────────────────────────────
"""
Based on the cluster profiles, each cluster maps primarily to a distinct geographic
region (Agriblock). Numeric features including Yield per Hectare show minimal 
variation across clusters, confirming that the clusters represent geographic 
farming segments rather than meaningful differences in yield performance.
This is a direct consequence of the narrow yield range in the dataset.
"""
cluster_descriptions = {
    0: "Kurinjipadi region — ponmani variety, NW wind pattern",
    1: "Sankarapuram region — delux ponni variety, SSE wind pattern",
    2: "Panruti region — delux ponni variety, ENE wind pattern",
    3: "Chinnasalem region — ponmani variety, E wind pattern",
    4: "Kallakurichi region — ponmani variety, W wind pattern",
    5: "Cuddalore region — delux ponni variety, SW wind pattern"
}

print("\n=== Cluster Descriptions ===")
for cluster, description in cluster_descriptions.items():
    print(f"Cluster {cluster}: {description}")