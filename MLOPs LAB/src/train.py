import yaml
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    exp_cfg = config["experiment"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    mlflow.set_experiment(exp_cfg["name"])

    with mlflow.start_run(run_name="baseline_run"):
        mlflow.log_param("model_name", model_cfg["name"])
        mlflow.log_param("C", model_cfg["params"]["C"])
        mlflow.log_param("max_iter", model_cfg["params"]["max_iter"])
        mlflow.log_param("random_state", exp_cfg["random_state"])
        mlflow.log_param("test_size", data_cfg["test_size"])

        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target,
            test_size=data_cfg["test_size"],
            random_state=exp_cfg["random_state"]
        )

        model = LogisticRegression(**model_cfg["params"])
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("configs/config.yaml")

        print(f"Run completed. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
