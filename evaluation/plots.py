import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)


def plot_roc(
    results,
    save_path
):

    plt.figure()

    for name, data in results.items():

        fpr, tpr, _ = roc_curve(
            data["labels"],
            data["errors"]
        )

        plt.plot(
            fpr,
            tpr,
            label=name
        )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_pr(
    results,
    save_path
):

    plt.figure()

    for name, data in results.items():

        precision, recall, _ = \
            precision_recall_curve(
                data["labels"],
                data["errors"]
            )

        plt.plot(
            recall,
            precision,
            label=name
        )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()