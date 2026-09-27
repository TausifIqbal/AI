# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# Each library serves a specific role in the classification pipeline.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np           # numerical operations (array math, sorting, etc.)
import pandas as pd          # tabular data manipulation (loading CSV, filtering columns)
import matplotlib.pyplot as plt          # plotting engine for all charts
from sklearn.ensemble import RandomForestClassifier  # the ML model we use to classify
from sklearn.metrics import (
    accuracy_score,       # fraction of all predictions that were correct
    auc,                  # computes area under a curve (used with roc_curve)
    classification_report,  # full per-class precision/recall/F1 summary as text
    confusion_matrix,     # table showing true vs predicted counts per class
    f1_score,             # harmonic mean of precision and recall (balances both)
    precision_score,      # of all predicted positives, how many were truly positive
    recall_score,         # of all actual positives, how many did the model catch
    roc_curve,            # generates the ROC curve points (FPR vs TPR at each threshold)
)
from sklearn.model_selection import train_test_split  # splits data into train and test sets


# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# Stores loaded DataFrames in memory so the same file is not read from disk
# again if the function is called multiple times in a session.
# ─────────────────────────────────────────────────────────────────────────────

DATAFRAME_CACHE = {}  # maps file_name → DataFrame; acts like an in-memory store


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classification_dataset(file_name: str, target_column: str) -> dict[str, float]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the classification target.
    Returns:
        dict[str, float]: A dictionary with the model's metrics.
    """

    # ── Step 1: Load Data ────────────────────────────────────────────────────
    if file_name not in DATAFRAME_CACHE:           # only load if not already in memory
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)  # read CSV into a DataFrame
        except FileNotFoundError:                  # handles the case where the file path is wrong
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except (ValueError, OSError) as e:         # handles corrupt files or permission errors
            return {"error": f"Error loading '{file_name}': {e!s}"}

    df = DATAFRAME_CACHE[file_name]                # retrieve the cached DataFrame

    # ── Step 2: Validate Target Column ──────────────────────────────────────
    if target_column not in df.columns:            # make sure the label column actually exists
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}

    # ── Step 3: Split Features and Label ────────────────────────────────────
    # In classification:
    #   X = the input features the model learns patterns from
    #   y = the label (class) the model tries to predict
    X = df.drop(columns=[target_column])  # all columns except the target become features
    y = df[target_column]                 # the target column is the label (e.g. 0=malignant, 1=benign)

    # ── Step 4: Train/Test Split ─────────────────────────────────────────────
    # We hold out 20% of rows for testing so the model is evaluated on data it never saw.
    # random_state=42 makes the split reproducible (same split every run).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,     # 20% of rows go to the test set
        random_state=42,   # seed for reproducibility
    )

    # ── Step 5: Train the Model ──────────────────────────────────────────────
    # A Random Forest trains many decision trees and combines their votes.
    # More trees → more stable predictions, but slower training.
    model = RandomForestClassifier(
        n_estimators=100,  # build 100 individual decision trees
        random_state=42,   # seed so the forest is reproducible
    )
    model.fit(X_train, y_train)  # learn patterns from the training data

    # ── Step 6: Make Predictions ─────────────────────────────────────────────
    y_pred  = model.predict(X_test)           # hard class predictions (0 or 1)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of being class 1 (used for ROC)

    # ── Step 7: Compute Metrics ──────────────────────────────────────────────
    #
    # CONFUSION MATRIX BUILDING BLOCKS (binary case: Positive = class 1, Negative = class 0)
    #   TP (True Positive)  — model predicted Positive, and it was actually Positive
    #   TN (True Negative)  — model predicted Negative, and it was actually Negative
    #   FP (False Positive) — model predicted Positive, but it was actually Negative  ← "false alarm"
    #   FN (False Negative) — model predicted Negative, but it was actually Positive  ← "missed case"
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ ACCURACY                                                            │
    # │   Definition : Fraction of all predictions that were correct.      │
    # │   Formula    : (TP + TN) / (TP + TN + FP + FN)                    │
    # │   Range      : 0.0 (worst) → 1.0 (perfect)                        │
    # │   Best when  : Classes are roughly balanced.                       │
    # │   Pitfall    : On a dataset that is 95% class 0, a model that      │
    # │                always predicts class 0 gets 95% accuracy — yet     │
    # │                never detects a single class-1 case.                │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ PRECISION                                                           │
    # │   Definition : Of all samples the model labelled as Positive,      │
    # │                what fraction were truly Positive?                  │
    # │   Formula    : TP / (TP + FP)                                      │
    # │   Range      : 0.0 → 1.0                                           │
    # │   Best when  : False alarms (FP) are costly.                       │
    # │   Example    : Spam filter — you don't want real emails in spam.   │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ RECALL  (also called Sensitivity or True Positive Rate)            │
    # │   Definition : Of all samples that were truly Positive, what       │
    # │                fraction did the model correctly identify?          │
    # │   Formula    : TP / (TP + FN)                                      │
    # │   Range      : 0.0 → 1.0                                           │
    # │   Best when  : Missing a Positive (FN) is costly.                 │
    # │   Example    : Cancer detection — missing a malignant tumour is    │
    # │                far more dangerous than a false alarm.              │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ F1-SCORE                                                            │
    # │   Definition : Harmonic mean of Precision and Recall.              │
    # │                Penalises extreme imbalance between the two.        │
    # │   Formula    : 2 × (Precision × Recall) / (Precision + Recall)    │
    # │   Range      : 0.0 → 1.0                                           │
    # │   Best when  : You care equally about FP and FN, or when classes  │
    # │                are imbalanced and accuracy alone is misleading.    │
    # │   Note       : arithmetic mean of 1.0 and 0.0 = 0.5 (looks OK),  │
    # │                but harmonic mean = 0.0 (correctly penalises it).  │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ AUC-ROC  (Area Under the ROC Curve)                                │
    # │   Definition : Probability that the model ranks a random Positive  │
    # │                sample higher than a random Negative sample.        │
    # │   Formula    : Area under the curve of TPR vs FPR across all      │
    # │                decision thresholds.                                │
    # │   Range      : 0.5 (random guessing) → 1.0 (perfect separation)  │
    # │   Best when  : You want a threshold-independent measure of how    │
    # │                well the model separates the two classes overall.  │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    # "weighted" averages each class's metric by how many samples it has —
    # this is fair when classes are imbalanced (e.g. 70% benign, 30% malignant).
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),                         # (TP+TN) / all samples
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),    # TP / (TP+FP)
        "recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),       # TP / (TP+FN)
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted"), 4),           # 2×P×R / (P+R)
    }

    # ── Step 8: Print Results ────────────────────────────────────────────────
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))   # per-class breakdown of precision/recall/F1
    print("=== Metrics ===")
    for k, v in metrics.items():    # loop through each metric and print it
        print(f"  {k}: {v}")

    # ── Step 9: Plot Visualizations ──────────────────────────────────────────
    _plot_metrics(
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=X.columns.tolist(),          # list of feature names for labelling the chart
        importances=model.feature_importances_,    # how much each feature contributed to predictions
        class_counts=y.value_counts().sort_index(),  # how many samples belong to each class
    )

    return metrics  # return the metrics dict for programmatic use


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTION
# Builds a 5-panel dashboard to visually understand model performance.
# ─────────────────────────────────────────────────────────────────────────────

def _plot_metrics(y_test, y_pred, y_proba, feature_names, importances, class_counts):

    # ── Pre-compute values needed by multiple charts ─────────────────────────
    classes = sorted(np.unique(y_test))    # unique class labels in sorted order (e.g. [0, 1])

    # CONFUSION MATRIX — rows = actual class, columns = predicted class
    # For a binary problem the 4 cells are: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)

    # ROC CURVE — sweeps every probability threshold from 0 → 1:
    #   FPR (False Positive Rate) = FP / (FP + TN)  ← x-axis: how often we raise a false alarm
    #   TPR (True Positive Rate)  = TP / (TP + FN)  ← y-axis: how often we catch a real positive
    # A perfect classifier hugs the top-left corner (FPR=0, TPR=1).
    # A random classifier follows the diagonal (FPR = TPR at every threshold).
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # AUC — area under the ROC curve; single summary number.
    # 1.0 = perfect, 0.5 = random guessing, < 0.5 = worse than random.
    roc_auc = auc(fpr, tpr)

    # ── Select top-N most important features ─────────────────────────────────
    top_n = 10                                         # only show the 10 most important features
    sorted_idx = np.argsort(importances)[::-1][:top_n] # indices of top features, sorted descending
    top_features   = [feature_names[i] for i in sorted_idx]  # names of those top features
    top_importances = importances[sorted_idx]                 # their importance scores

    # Each panel is saved as its own PNG so it can be embedded individually in documentation.
    # plt.close() after each save prevents figures from accumulating in memory.

    # ── Plot 1: Class Distribution ───────────────────────────────────────────
    # Shows how many samples belong to each class.
    # An imbalanced dataset (e.g. 90% class 0) can mislead accuracy alone.
    fig1, ax1 = plt.subplots(figsize=(6, 5))  # create a standalone figure for this panel
    bars = ax1.bar(
        [str(c) for c in classes],    # x-axis labels: class names as strings
        class_counts.values,          # bar heights: number of samples per class
        color=["#e05c5c", "#4a90d9"], # red for class 0 (malignant), blue for class 1 (benign)
        edgecolor="white",            # white border between bars for visual clarity
    )
    ax1.set_title("Class Distribution", fontweight="bold")
    ax1.set_xlabel("Class")    # x-axis label
    ax1.set_ylabel("Count")    # y-axis label
    for bar, val in zip(bars, class_counts.values):   # annotate each bar with its exact count
        ax1.text(
            bar.get_x() + bar.get_width() / 2,  # center of the bar horizontally
            bar.get_height() + 2,               # slightly above the bar top
            str(val),                           # the count as text
            ha="center", va="bottom", fontsize=10,
        )
    ax1.set_ylim(0, max(class_counts.values) * 1.15)  # add 15% headroom above tallest bar
    fig1.tight_layout()
    fig1.savefig("plot_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)   # free memory before opening the next figure
    print("Saved plot_class_distribution.png")

    # ── Plot 2: Confusion Matrix ──────────────────────────────────────────────
    # A 2×2 grid showing: True Positives, False Positives, True Negatives, False Negatives.
    # Diagonal = correct predictions; off-diagonal = errors.
    fig2, ax2 = plt.subplots(figsize=(5, 4))  # square-ish canvas suits a 2×2 matrix
    im = ax2.imshow(cm, interpolation="nearest", cmap="Blues")  # render matrix as a heatmap
    plt.colorbar(im, ax=ax2)           # add a color scale bar on the right
    ax2.set_title("Confusion Matrix", fontweight="bold")
    ax2.set_xlabel("Predicted Label")  # columns represent what the model predicted
    ax2.set_ylabel("True Label")       # rows represent the actual ground truth
    ax2.set_xticks(range(len(classes)))     # one tick per class on x-axis
    ax2.set_yticks(range(len(classes)))     # one tick per class on y-axis
    ax2.set_xticklabels(classes)       # label ticks with actual class values
    ax2.set_yticklabels(classes)
    thresh = cm.max() / 2.0            # threshold to switch text color (white on dark, black on light)
    for i in range(cm.shape[0]):       # iterate over rows (actual classes)
        for j in range(cm.shape[1]):   # iterate over columns (predicted classes)
            ax2.text(
                j, i, str(cm[i, j]),   # draw the count at position (j, i)
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",  # contrast-aware text color
                fontsize=14,
            )
    fig2.tight_layout()
    fig2.savefig("plot_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved plot_confusion_matrix.png")

    # ── Plot 3: ROC Curve ─────────────────────────────────────────────────────
    # Plots True Positive Rate vs False Positive Rate at every decision threshold.
    # AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing.
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.plot(fpr, tpr, color="#4a90d9", lw=2, label=f"AUC = {roc_auc:.3f}")  # ROC curve line
    ax3.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")  # baseline: random classifier
    ax3.fill_between(fpr, tpr, alpha=0.15, color="#4a90d9")  # shade the area under the curve
    ax3.set_title("ROC Curve", fontweight="bold")
    ax3.set_xlabel("False Positive Rate")  # FPR = false alarms / all actual negatives
    ax3.set_ylabel("True Positive Rate")   # TPR = correct positives / all actual positives
    ax3.legend(loc="lower right")          # show AUC value in the legend
    ax3.set_xlim([0, 1])                   # fix x-axis to [0, 1]
    ax3.set_ylim([0, 1.02])                # fix y-axis to [0, 1] with tiny top margin
    fig3.tight_layout()
    fig3.savefig("plot_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved plot_roc_curve.png")

    # ── Plot 4: Feature Importances ───────────────────────────────────────────
    # Shows which input columns had the most influence on the model's decisions.
    # Importance = average reduction in impurity caused by splits on that feature.
    fig4, ax4 = plt.subplots(figsize=(8, 5))  # wider canvas for long feature names
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]  # gradient of blue shades
    ax4.barh(
        range(top_n),             # y positions: one per feature
        top_importances[::-1],    # reversed so highest importance is at the top
        color=colors[::-1],       # matching reversed color gradient
        edgecolor="white",
    )
    ax4.set_yticks(range(top_n))                        # tick positions
    ax4.set_yticklabels(top_features[::-1], fontsize=9) # feature names as y-axis labels
    ax4.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    ax4.set_xlabel("Importance Score")  # x-axis shows the numerical importance
    for i, v in enumerate(top_importances[::-1]):   # label each bar with its numeric score
        ax4.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    fig4.tight_layout()
    fig4.savefig("plot_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved plot_feature_importances.png")

    # ── Plot 5: Metrics Bar Chart ─────────────────────────────────────────────
    # A quick visual comparison of the four key evaluation metrics side by side.
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]  # metric labels
    metric_values = [
        accuracy_score(y_test, y_pred),                          # % of all correct predictions
        precision_score(y_test, y_pred, average="weighted"),     # quality of positive predictions
        recall_score(y_test, y_pred, average="weighted"),        # coverage of actual positives
        f1_score(y_test, y_pred, average="weighted"),            # harmonic mean of precision & recall
    ]
    bar_colors = ["#4a90d9", "#5cb85c", "#f0ad4e", "#9b59b6"]  # distinct color per metric
    bars2 = ax5.bar(metric_names, metric_values, color=bar_colors, edgecolor="white")
    ax5.set_title("Model Metrics", fontweight="bold")
    ax5.set_ylim(0, 1.15)       # scores range 0–1; extra headroom for labels above bars
    ax5.set_ylabel("Score")
    for bar, val in zip(bars2, metric_values):   # annotate each bar with its exact score
        ax5.text(
            bar.get_x() + bar.get_width() / 2,  # horizontal center of the bar
            bar.get_height() + 0.02,             # just above the bar top
            f"{val:.3f}",                        # score rounded to 3 decimal places
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax5.tick_params(axis="x", labelsize=9)  # smaller font for x-axis metric names
    fig5.tight_layout()
    fig5.savefig("plot_metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Saved plot_metrics_bar.png")

    print("\nAll plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# This block only runs when the script is executed directly (not when imported).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = evaluate_classification_dataset(
        file_name="classification-dataset.csv",  # path to the dataset CSV
        target_column="target",                  # column name that holds the class labels
    )
    print("\nFinal results:", results)  # print the returned metrics dictionary
