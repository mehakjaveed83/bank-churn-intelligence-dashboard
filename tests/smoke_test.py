import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.churn_project.pipeline import build_feature_frame, load_bank_data, train_and_save


def main() -> None:
    df = load_bank_data(prefer_full=False)
    X, y, assumptions = build_feature_frame(df)
    assert assumptions["target_column"] == "y"
    assert "duration" not in X.columns
    assert y.isin([0, 1]).all()

    bundle = train_and_save(prefer_full=False, sample_rows=1200, save_artifact=False)
    assert bundle.metrics["roc_auc"] >= 0.6
    assert not bundle.shap_summary.empty
    assert bundle.test_sample["churn_risk"].between(0, 1).all()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
