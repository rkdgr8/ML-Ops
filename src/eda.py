"""Exploratory data analysis utilities for the Heart Disease dataset."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_OUTPUT_DIR, PROCESSED_DATA_PATH
from src.data_processing import prepare_dataset


TARGET_LABELS = {0: "No Disease", 1: "Disease"}


def load_or_prepare_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset, creating it first if needed."""
    if path.exists():
        return pd.read_csv(path)

    return prepare_dataset()


def summarize_dataset(data: pd.DataFrame) -> dict:
    """Return basic EDA summary statistics for report writing."""
    return {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "missing_values": int(data.isna().sum().sum()),
        "target_distribution": data["target"].value_counts().sort_index().to_dict(),
    }


def save_class_distribution(data: pd.DataFrame, output_dir=EDA_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = data.copy()
    plot_data["target_label"] = plot_data["target"].map(TARGET_LABELS)

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=plot_data,
        x="target_label",
        hue="target_label",
        palette="Set2",
    )
    ax.set_title("Heart Disease Class Distribution")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Patient Count")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()


def save_histograms(data: pd.DataFrame, output_dir=EDA_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    axes = data[numeric_columns].hist(figsize=(12, 8), bins=20, color="#4c78a8")
    for axis_row in axes:
        for axis in axis_row:
            axis.set_xlabel(axis.get_title())
            axis.set_ylabel("Frequency")
            axis.set_title(axis.get_title().title())

    plt.suptitle("Numeric Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_histograms.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_correlation_heatmap(data: pd.DataFrame, output_dir=EDA_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 9))
    sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_missing_value_plot(data: pd.DataFrame, output_dir=EDA_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_counts = data.isna().sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=missing_counts.index, y=missing_counts.values, color="#f58518")
    ax.set_title("Missing Values After Cleaning")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Value Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=160)
    plt.close()


def save_feature_relationships(data: pd.DataFrame, output_dir=EDA_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = data.copy()
    plot_data["target_label"] = plot_data["target"].map(TARGET_LABELS)

    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(
        data=plot_data,
        x="age",
        y="thalach",
        hue="target_label",
        palette="Set2",
        alpha=0.85,
    )
    ax.set_title("Age vs Maximum Heart Rate by Diagnosis")
    ax.set_xlabel("Age")
    ax.set_ylabel("Maximum Heart Rate Achieved")
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(output_dir / "age_vs_thalach.png", dpi=160)
    plt.close()


def run_eda(data: pd.DataFrame, output_dir=EDA_OUTPUT_DIR) -> dict:
    """Create EDA plots and return summary values."""
    save_class_distribution(data, output_dir)
    save_histograms(data, output_dir)
    save_correlation_heatmap(data, output_dir)
    save_missing_value_plot(data, output_dir)
    save_feature_relationships(data, output_dir)
    return summarize_dataset(data)


def main() -> None:
    data = load_or_prepare_processed_data()
    summary = run_eda(data)

    print(f"Saved EDA plots to: {EDA_OUTPUT_DIR}")
    print("EDA summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
