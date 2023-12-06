import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from dataset import prepare_data
from model.resnet import ResNet

lr = 1e-4
weight_decay = 1e-4
batch_size = 1024
num_epochs = 20


def train():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    model = ResNet(training=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    data = pd.read_csv("data/fer2013/fer2013.csv")
    train_x, train_y, test_x, test_y = prepare_data(data)
    print("Data is loaded.")
    train_x = torch.tensor(train_x / 255.0).view(-1, 1, 48, 48).to(torch.float32).to(device)  # torch.Size([32298, 1, 48, 48])
    train_y = torch.tensor(train_y).to(device)
    test_x = torch.tensor(test_x / 255.0).view(-1, 1, 48, 48).to(torch.float32).to(device)  # torch.Size([32298, 1, 48, 48])
    test_y = torch.tensor(test_y).to(device)
    train_x = torch.cat((train_x, test_x))
    train_y = torch.cat((train_y, test_y))
    train_x_batch = torch.split(train_x, batch_size)
    train_y_batch = torch.split(train_y, batch_size)

    step_history = []
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        for i, (x, y) in enumerate(zip(train_x_batch, train_y_batch)):
            output = model(x)  # torch.Size([1024, 7])
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_history.append(epoch * len(train_x_batch) + i)
            loss_history.append(loss.item())

            if (i + 1) % 4 == 0:
                print(f'Epoch [{epoch}/{num_epochs - 1}], Step [{i + 1}/{len(train_x_batch)}], Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            output = model(test_x)
            pred = torch.argmax(output, dim=1)
            correct = torch.sum(torch.tensor(pred == test_y))
            print(f"Accuracy: {correct / test_y.shape[0]:.2f}")

        torch.save(model.state_dict(), f"resnet_pretrained/epoch{epoch}.pth")
        print(f"resnet_pretrained/epoch{epoch}.pth saved.")

    plt.plot(step_history, loss_history)
    plt.show()


if __name__ == "__main__":
    train()
