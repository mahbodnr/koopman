import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import pytorch_lightning as lightning
import lightning
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

# from kooplearn.models.feature_maps.nn import NNFeatureMap
from nnfeaturemap import NNFeatureMap
from kooplearn.data import traj_to_contexts, TensorContextDataset
from kooplearn.nn import DPLoss
from kooplearn.nn.data import collate_context_dataset
# from kooplearn.models import Nonlinear
from nonlinear import Nonlinear
from tqdm import tqdm


conv = lambda x: np.linalg.norm((x[1:] - x[:-1]).reshape((x.shape[0] - 1, -1)), axis=1)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = "cpu"


class KoopmanNet(nn.Module):
    def __init__(self):
        super(KoopmanNet, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc_deq_x = nn.Linear(64, 64)
        self.fc_deq_z = nn.Linear(64, 64)
        self.deq_iterations = 20
        self.fc3 = nn.Linear(64, 10)
        self.koopman_net = KoopmanNet()
        self.koopman_critarion = DPLoss(
            relaxed=True, metric_deformation=1, center_covariances=False
        )
        self.feature_map = NNFeatureMap(self.koopman_net, self.koopman_critarion)
        self.koopman_optimizer = optim.Adam(self.koopman_net.parameters(), lr=1e-3)

    def deq(self, x):
        z = torch.zeros_like(x)
        x_inj = self.fc_deq_x(x)
        deq_history = []
        for _ in range(self.deq_iterations):
            z = torch.tanh(x_inj + self.fc_deq_z(z))
            deq_history.append(z.clone().detach())

        h = torch.stack(deq_history)  # [deq_iterations, B, 64]
        koopman_loss = self.feature_map.fit(h)
        if self.training:
            self.koopman_optimizer.zero_grad()
            koopman_loss.backward()
            self.koopman_optimizer.step()

        ctx = traj_to_contexts(h, backend="torch")
        koopman = Nonlinear(
            self.feature_map, reduced_rank=True, rank=10, tikhonov_reg=1e-3
        ).fit(ctx)
        pred = koopman.predict(ctx[-1], t=10, predict_observables=True)[0,0]
        if not self.training:
            eigs = koopman.eig()
            plt.scatter(eigs.real, eigs.imag)
            theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(np.cos(theta), np.sin(theta), 'k--')
            plt.axis('equal')
            plt.show()

        return z

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.deq(x)
        x = self.fc3(x)
        return x


def evaluate(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="~/data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.MNIST(
    root="~/data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=512, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training the model
for epoch in range(5):  # 5 epochs
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # print(f"loss: {loss}")
        # print(f"koopman_loss: {model.koopman_loss}")
        # loss += model.koopman_loss * 1e-3
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    acc = evaluate(model, testloader)
    # c = conv(torch.stack(model.deq_history))
    # plt.plot(c)
    # plt.show()
    # print(c[-1])
    print(f"Accuracy after epoch {epoch + 1}: {acc}%")

print("Finished Training")
