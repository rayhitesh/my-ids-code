import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    epochs,
    learning_rate,
    device="cpu"
):

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.MSELoss()

    history = []

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for X, _ in train_loader:

            X = X.to(device)

            optimizer.zero_grad()

            reconstruction = model(X)

            loss = criterion(
                reconstruction,
                X
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(train_loader)
        )

        history.append(avg_loss)

        print(
            f"Epoch {epoch + 1:03d}/"
            f"{epochs} | "
            f"Loss: {avg_loss:.6f}"
        )

    return model, history