"""
Modelling and tuning script for mushroom classification.

This script uses MLflow and scikit-learn to train and tune a random forest classifier
for mushroom classification.
"""

import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# KONFIGURASI DAGSHUB & MLFLOW
# Membaca dari environment variables (GitHub Secrets) yang akan di-supply oleh GitHub Actions
DAGSHUB_REPO_OWNER = os.getenv('DAGSHUB_OWNER')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')

# Inisialisasi DagsHub -> mengkonfigurasi MLflow
dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

# Set URI MLflow ke repository DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")

# MEMUAT DAN MEMBAGI DATA
print("Memuat data...")
df = pd.read_csv('mushrooms_preprocessing/mushrooms_preprocessed.csv')

# Pisahkan fitur (X) dan target (y)
X = df.drop('class', axis=1)
y = df['class']

# Bagi data
# Bagi data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Data berhasil dimuat. Ukuran data latih: {X_train.shape}, Ukuran data uji: {X_test.shape}")

# HYPERPARAMETER TUNING & TRAINING DENGAN MLFLOW

# Mulai eksperimen MLflow
with mlflow.start_run(run_name="Mushroom Classification Tuning (Random Forest)") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    mlflow.set_tag("Model Type", "Random Forest Classifier")

    # Definisikan parameter grid untuk GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }

    # Inisialisasi model dan GridSearchCV untuk mencari parameter terbaik
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
    )
    print("\nMemulai Hyperparameter Tuning dengan GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print("Tuning selesai.")

    # Dapatkan model dan parameter terbaik dari hasil tuning
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nParameter terbaik ditemukan: {best_params}")

    # MANUAL LOGGING (Skilled/Advanced)
    # Log (catat) parameter terbaik ke MLflow
    mlflow.log_params(best_params)

    # Evaluasi model terbaik pada data uji
    y_pred = best_model.predict(X_test)

    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Akurasi: {accuracy:.4f}")
    print(f"Presisi: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Log (catat) metrik evaluasi ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # LOGGING ARTIFACT TAMBAHAN (Advanced)
    # Log Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Edible', 'Poisonous'],
        yticklabels=['Edible', 'Poisonous']
        )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", artifact_path="evaluation_plots")
    plt.close()
    print("Plot Confusion Matrix berhasil di-log.")

    # Log Feature Importance Plot
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
        })
    feature_importance_df = (
        feature_importance_df.sort_values(by='importance', ascending=False)
        .head(15)
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png", artifact_path="evaluation_plots")
    plt.close()
    print("Plot Feature Importance berhasil di-log.")

    input_example = X_train.head(1)

    # Log (simpan) model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="mushroom-classifier-model",
        registered_model_name="MushroomClassifierRF",
        input_example=input_example
    )
    print("Model berhasil di-log ke MLflow dengan input signature.")

    print("\nEksperimen selesai! Cek hasilnya di DagsHub:")
    dagshub_link = (
        f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
        f"/experiments/{run.info.experiment_id}/{run_id}"
    )
    print(f"Link: {dagshub_link}")
