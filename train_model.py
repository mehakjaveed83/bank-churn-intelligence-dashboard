from src.churn_project.pipeline import ARTIFACT_PATH, train_and_save


if __name__ == "__main__":
    bundle = train_and_save(prefer_full=True)
    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print("Metrics:")
    for key, value in bundle.metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.3f}")

