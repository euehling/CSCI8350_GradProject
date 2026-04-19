import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ── Data Loading ──────────────────────────────────────────────────────────────
data = Path("Data/paddydataset.csv")
df = pd.read_csv(data)
df.columns = df.columns.str.strip()

# ── Build Modeling Dataset ────────────────────────────────────────────────────
df_for_modeling = df.copy()

# Drop duplicate rows
df_for_modeling = df_for_modeling.drop_duplicates()
print("Shape after dropping duplicates:", df_for_modeling.shape)

# Create yield per hectare, drop raw yield and correlated columns
df_for_modeling['Yield per Hectare'] = df_for_modeling['Paddy yield(in Kg)'] / df_for_modeling['Hectares']
df_for_modeling = df_for_modeling.drop(columns=[
    'LP_Mainfield(in Tonnes)',
    'LP_nurseryarea(in Tonnes)',
    'Nursery area (Cents)',
    'Paddy yield(in Kg)'
])

# ── Two Binning Strategies ────────────────────────────────────────────────────
# Strategy 1: pd.cut — equal value ranges
df_cut = df_for_modeling.copy()
df_cut['Yield Per Hectare Class'] = pd.cut(
    df_cut['Yield per Hectare'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)
df_cut = df_cut.drop(columns=['Yield per Hectare'])

# Strategy 2: Domain-based thresholds
df_domain = df_for_modeling.copy()
df_domain['Yield Per Hectare Class'] = pd.cut(
    df_domain['Yield per Hectare'],
    bins=[0, 5750, 6200, float('inf')],
    labels=['Low', 'Medium', 'High']
)
df_domain = df_domain.drop(columns=['Yield per Hectare'])

# Strategy 3: Quantile-based (equal frequency)
df_quantile = df_for_modeling.copy()
df_quantile['Yield Per Hectare Class'] = pd.qcut(
    df_quantile['Yield per Hectare'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
df_quantile = df_quantile.drop(columns=['Yield per Hectare'])



print("\n=== Strategy 1: pd.cut class distribution ===")
print(df_cut['Yield Per Hectare Class'].value_counts().sort_index())
print("\n=== Strategy 2: Domain-based class distribution ===")
print(df_domain['Yield Per Hectare Class'].value_counts().sort_index())

# ── Shared Settings ───────────────────────────────────────────────────────────
random_state = 42
target_var = 'Yield Per Hectare Class'

rf_param_dist = {
    'rf__n_estimators': [100, 250, 500],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__min_samples_split': [10, 20, 50]
}
dt_param_dist = {
    'dt__max_depth': [None, 5, 10, 20],
    'dt__min_samples_split': [2, 10, 20, 50],
    'dt__criterion': ['gini', 'entropy']
}
lr_param_dist = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__solver': ['lbfgs', 'saga']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

def evaluate(name, model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob, multi_class='ovr', average='macro')
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}  |  ROC AUC (OvR macro): {auc:.4f}")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return {'accuracy': acc, 'roc_auc': auc}

# ── Run Both Strategies ───────────────────────────────────────────────────────
all_results = {}

for strategy_name, dataset in [("pd.cut (equal ranges)", df_cut), ("Domain-based thresholds", df_domain)]:
    print(f"\n\n{'='*60}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*60}")

    train, test = train_test_split(
        dataset, test_size=0.2,
        stratify=dataset[target_var],
        random_state=random_state
    )

    predictor_columns = dataset.drop(columns=[target_var])
    predictors = predictor_columns.columns

    X_train = train[predictors]
    y_train = train[target_var]
    X_test  = test[predictors]
    y_test  = test[target_var]

    numeric_features     = predictor_columns.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = predictor_columns.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    preprocessor = Pipeline(steps=[
        ("column_transform", ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ])),
        ("var_threshold", VarianceThreshold(threshold=0.0))
    ])

    rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("rf", RandomForestClassifier(random_state=random_state))])
    dt_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("dt", DecisionTreeClassifier(random_state=random_state))])
    lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("lr", LogisticRegression(random_state=random_state, max_iter=1000))])

    rf_gs = GridSearchCV(estimator=rf_pipeline, param_grid=rf_param_dist, cv=cv, scoring='roc_auc_ovr', verbose=0)
    rf_gs.fit(X_train, y_train)
    print("RF Best Params:", rf_gs.best_params_)

    dt_gs = GridSearchCV(estimator=dt_pipeline, param_grid=dt_param_dist, cv=cv, scoring='roc_auc_ovr', verbose=0)
    dt_gs.fit(X_train, y_train)
    print("DT Best Params:", dt_gs.best_params_)

    lr_gs = GridSearchCV(estimator=lr_pipeline, param_grid=lr_param_dist, cv=cv, scoring='roc_auc_ovr', verbose=0)
    lr_gs.fit(X_train, y_train)
    print("LR Best Params:", lr_gs.best_params_)

    best_rf = rf_gs.best_estimator_
    best_dt = dt_gs.best_estimator_
    best_lr = lr_gs.best_estimator_

    print("\n--- Test Set Results ---")
    test_rf = evaluate("Random Forest - Test",       best_rf, X_test, y_test)
    test_dt = evaluate("Decision Tree - Test",       best_dt, X_test, y_test)
    test_lr = evaluate("Logistic Regression - Test", best_lr, X_test, y_test)

    all_results[strategy_name] = {
        "Random Forest":       test_rf,
        "Decision Tree":       test_dt,
        "Logistic Regression": test_lr,
    }

# ── Final Comparison Across Both Strategies ───────────────────────────────────
print("\n\n=== FINAL COMPARISON: Both Strategies ===")
print(f"{'Strategy':<26} {'Model':<25} {'Test Acc':<12} {'Test ROC AUC'}")
print("-" * 75)
for strategy_name, models in all_results.items():
    for model_name, metrics in models.items():
        print(f"{strategy_name:<26} {model_name:<25} {metrics['accuracy']:<12.4f} {metrics['roc_auc']:.4f}")