import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def get_reconstruction_errors(
    model,
    data_loader,
    device="cpu"
):

    model.eval()

    errors = []
    labels = []

    with torch.no_grad():

        for X, y in data_loader:

            X = X.to(device)

            reconstruction = model(X)

            error = torch.mean(
                (X - reconstruction) ** 2,
                dim=1
            )

            errors.extend(
                error.cpu().numpy()
            )

            labels.extend(
                y.numpy()
            )

    return (
        np.array(errors),
        np.array(labels)
    )


def calculate_metrics(
    errors,
    labels,
    threshold
):

    predictions = (
        errors > threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    far = fp / (fp + tn) \
        if (fp + tn) > 0 else 0

    auc = roc_auc_score(
        labels,
        errors
    )

    pr_auc = average_precision_score(
        labels,
        errors
    )

    print("\nClassification Report")

    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "FAR": far,

        "AUC_ROC": auc,

        "PR_AUC": pr_auc,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp,

        "Threshold": threshold
    }