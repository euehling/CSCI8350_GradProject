import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

# ── Set up pipeline for scaler and PCA to reduce dimensionality ───────────
scaler_with_PCA = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('pca', PCA(n_components=23, random_state=random_state))])

print("Scaling and PCA pipeline:", scaler_with_PCA)

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
k_range = range(1, 15)
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
                                                         algorithm='lloyd')
                                        )])
    # changed from elkan to lloyd due to warning when running this code
    pipeline.fit(df_for_clustering)
    inertia.append(pipeline.named_steps["kmeans"].inertia_)

# ── Plot Inertia ─────────────────────────────────────────────────────────
plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 15), inertia, 'o')
plt.plot(np.arange(1, 15), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()

# ── Could implement silhouette method to identify optimal k─────────────────
# Working first with fairly obvious elbow at k = 6
# Example at https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# Could also use grid to check different values for other k-means parameters

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






