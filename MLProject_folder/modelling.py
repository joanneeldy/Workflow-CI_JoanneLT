import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
import sys
import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# VALIDASI OTENTIKASI
print("INFO: Memverifikasi DagsHub Token...")
if 'DAGSHUB_USER_TOKEN' not in os.environ:
    print("="*80)
    print("ERROR: DAGSHUB_USER_TOKEN tidak ditemukan!")
    print("Pastikan Anda sudah setup secrets di repository GitHub.")
    print("="*80)
    sys.exit(1)
print("SUCCESS: DagsHub Token ditemukan.")

# KONFIGURASI DAGSHUB & MLFLOW
DAGSHUB_REPO_OWNER = os.getenv('DAGSHUB_OWNER')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")

# MEMUAT DAN MEMBAGI DATA
print("Memuat data...")
df = pd.read_csv('MLProject_folder/mushrooms_preprocessing/mushrooms_preprocessed.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data berhasil dimuat. Ukuran data latih: {X_train.shape}")

# HYPERPARAMETER TUNING & TRAINING
with mlflow.start_run(run_name="Tuning RandomForest from CI") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    mlflow.set_tag("Model Type", "Random Forest Classifier")

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    print("\nMemulai Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    print("Tuning selesai.")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"\nParameter terbaik: {best_params}")
    
    mlflow.log_params(best_params)
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # LOGGING ARTEFAK TAMBAHAN
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", artifact_path="evaluation_plots")
    plt.close()
    print("Plot Confusion Matrix berhasil di-log.")

    # MENYIMPAN MODEL DENGAN SIGNATURE
    input_example = X_train.head(1)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model", # Nama folder artefak
        registered_model_name="MushroomClassifierRF", # Nama model di registry
        input_example=input_example
    )
    print("Model berhasil di-log ke MLflow dengan input signature.")
    print("\nEksperimen selesai!")