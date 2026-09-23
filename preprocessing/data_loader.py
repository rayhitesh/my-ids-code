import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(
    path,
    target_column="Label",
    batch_size=256,
    test_size=0.20,
    seed=42
):

    print("\nLoading dataset...")

    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    # --------------------------------------------------
    # Clean column names
    # --------------------------------------------------

    df.columns = df.columns.str.strip()

    # --------------------------------------------------
    # Remove unnecessary index columns
    # --------------------------------------------------

    unwanted = [
        "Unnamed: 0",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ]

    remove_columns = [
        col for col in unwanted
        if col in df.columns
    ]

    df = df.drop(
        columns=remove_columns,
        errors="ignore"
    )

    # --------------------------------------------------
    # Clean infinity values
    # --------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    # --------------------------------------------------
    # Separate target
    # --------------------------------------------------

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            f"not found."
        )

    y = df[target_column].astype(str).str.strip()

    X = df.drop(
        columns=[target_column]
    )

    # --------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_columns:

        X[col] = pd.factorize(
            X[col]
        )[0]

    # --------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = X.notna().all(axis=1)

    X = X.loc[valid]
    y = y.loc[valid]

    # --------------------------------------------------
    # Binary label
    # --------------------------------------------------

    y_binary = np.where(
        y.str.upper() == "BENIGN",
        0,
        1
    )

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y_binary,

        test_size=test_size,

        random_state=seed,

        stratify=y_binary
    )

    # --------------------------------------------------
    # Standardization
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("\nFinal data:")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    print("\nFeatures:", X_train.shape[1])

    print(
        "\nTraining benign:",
        np.sum(y_train == 0)
    )

    print(
        "Training attack:",
        np.sum(y_train == 1)
    )

    print(
        "\nTesting benign:",
        np.sum(y_test == 0)
    )

    print(
        "Testing attack:",
        np.sum(y_test == 1)
    )

    return (
        train_loader,
        test_loader,
        y_train,
        y_test,
        X_train.shape[1]
    )