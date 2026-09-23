import os
import random

import numpy as np
import torch
import pandas as pd

from preprocessing.data_loader import load_data

from models.sae import SAE
from models.saer import SAER
from models.saea import SAEA
from models.ra_sae import RA_SAE

from training.trainer import train_model

from evaluation.metrics import (
    get_reconstruction_errors,
    calculate_metrics
)

from config import *


# =====================================================
# Reproducibility
# =====================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# =====================================================
# Output directory
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)


# =====================================================
# Load data
# =====================================================

(
    train_loader,
    test_loader,
    y_train,
    y_test,
    input_dim
) = load_data(

    DATA_PATH,

    TARGET_COLUMN,

    BATCH_SIZE,

    TEST_SIZE,

    SEED
)


# =====================================================
# Models
# =====================================================

models = {

    "SAE": SAE(
        input_dim,
        LATENT_DIM
    ),

    "SAER": SAER(
        input_dim,
        LATENT_DIM
    ),

    "SAEA": SAEA(
        input_dim,
        LATENT_DIM
    ),

    "RA-SAE": RA_SAE(
        input_dim,
        LATENT_DIM
    )
}


# =====================================================
# Store results
# =====================================================

all_metrics = []

plot_data = {}


# =====================================================
# Train each model
# =====================================================

for model_name, model in models.items():

    print("\n")
    print("=" * 70)

    print(
        f"TRAINING MODEL: {model_name}"
    )

    print("=" * 70)

    # ---------------------------------------------
    # Train
    # ---------------------------------------------

    model, history = train_model(

        model,

        train_loader,

        EPOCHS,

        LEARNING_RATE,

        DEVICE
    )

    # ---------------------------------------------
    # Training reconstruction errors
    # ---------------------------------------------

    train_errors, train_labels = \
        get_reconstruction_errors(

            model,

            train_loader,

            DEVICE
        )

    # ---------------------------------------------
    # Test reconstruction errors
    # ---------------------------------------------

    test_errors, test_labels = \
        get_reconstruction_errors(

            model,

            test_loader,

            DEVICE
        )

    # ---------------------------------------------
    # Threshold based on BENIGN training samples
    # ---------------------------------------------

    normal_errors = train_errors[
        train_labels == 0
    ]

    threshold = np.percentile(
        normal_errors,
        THRESHOLD_PERCENTILE
    )

    print(
        f"\nThreshold: {threshold:.8f}"
    )

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    metrics = calculate_metrics(

        test_errors,

        test_labels,

        threshold
    )

    metrics["Model"] = model_name

    metrics["Parameters"] = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    all_metrics.append(metrics)

    # ---------------------------------------------
    # Plot data
    # ---------------------------------------------

    plot_data[model_name] = {

        "errors": test_errors,

        "labels": test_labels
    }

    # ---------------------------------------------
    # Save model
    # ---------------------------------------------

    torch.save(

        model.state_dict(),

        f"results/{model_name}.pth"
    )


# =====================================================
# Save metrics
# =====================================================

results_df = pd.DataFrame(
    all_metrics
)

results_df = results_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FAR",
        "AUC_ROC",
        "PR_AUC",
        "Parameters",
        "Threshold",
        "TN",
        "FP",
        "FN",
        "TP"
    ]
]

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# =====================================================
# Print final table
# =====================================================

print("\n")
print("=" * 90)

print("FINAL MODEL COMPARISON")

print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")

print(
    "results/model_comparison.csv"
)