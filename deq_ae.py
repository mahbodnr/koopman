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

import torch.autograd as autograd


conv = lambda x: np.linalg.norm((x[1:] - x[:-1]).reshape((x.shape[0] - 1, -1)), axis=1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = "cpu"

def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res


class KoopmanNet(nn.Module):
    def __init__(self):
        super(KoopmanNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc_deq_x = nn.Linear(64, 64)
        self.fc_deq_z = nn.Linear(64, 64)
        self.deq_iterations = 50
        self.fc3 = nn.Linear(64, 10)
        self.koopman_net = KoopmanNet()
        self.koopman_t = 100
        self.koopman_critarion = nn.MSELoss()

        
    def deq_f(self, z, x):
        return torch.tanh(x + self.fc_deq_z(z))

    def deq(self, x):
        x_inj = self.fc_deq_x(x)
        with torch.no_grad():
            z = torch.zeros_like(x)
            deq_history = []
            for _ in range(self.deq_iterations):
                z = self.deq_f(z, x_inj)
                deq_history.append(z.clone().detach())

            h = torch.stack(deq_history)  # [deq_iterations, B, 64]
            r, decoded = self.koopman_net(h)
            self.koopman_loss = self.koopman_critarion(decoded, h)
            r = r.transpose(0, 1) # [B, deq_iterations, 16]
            r = r + 1e-5 * torch.randn_like(r)
            koopman = torch.linalg.lstsq(r[:, :-1], r[:, 1:])
            pred = r[:,-1].unsqueeze(1) @ torch.linalg.matrix_power(koopman.solution, self.koopman_t)
            pred = self.koopman_net.decoder(pred.squeeze(1))


        z = self.deq_f(pred, x_inj)

        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.deq_f(z0, x_inj)
            def backward_hook(grad):
                g, self.backward_res = forward_iteration(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                grad)
                return g
                    
            z.register_hook(backward_hook)

        if not self.training:
            deq_history = deq_history + [pred, z]
            self.conv = conv(torch.stack(deq_history).cpu().detach().numpy())
        return z

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.deq(x)
        x = self.fc3(x)
        return x

    def eig(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        with torch.no_grad():
            z = torch.zeros_like(x)
            x_inj = self.fc_deq_x(x)
            deq_history = []
            for _ in range(self.deq_iterations):
                z = self.deq_f(z, x_inj)
                deq_history.append(z.clone().detach())

            h = torch.stack(deq_history)  # [deq_iterations, B, 64]
            self.koopman_loss = self.feature_map.fit(h)
            ctx = traj_to_contexts(h, backend="torch")
            koopman = Nonlinear(
                self.feature_map, reduced_rank=True, rank=10, tikhonov_reg=1e-3
            ).fit(ctx)
            return koopman.eig()

def evaluate(model, dataloader, epoch=None):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if epoch is not None:
            plt.plot(model.conv)
            plt.savefig(f"imgs/conv{epoch}.png")
            plt.close()

            # eigs = model.eig(images)
            # plt.scatter(eigs.real, eigs.imag)
            # theta = np.linspace(0, 2*np.pi, 100)
            # plt.plot(np.cos(theta), np.sin(theta), 'k--')
            # plt.axis('equal')
            # plt.savefig(f"imgs/eigs{epoch}.png")
            # plt.close()

    return 100 * correct / total


# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="~/data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testset = torchvision.datasets.MNIST(
    root="~/data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training the model
for epoch in range(50):  # 5 epochs
    running_loss = 0.0
    model.train()
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # print(f"loss: {loss}")
        # print(f"koopman_loss: {model.koopman_loss}")
        koopman_loss = model.koopman_loss * 1000
        loss += koopman_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, koopman_loss: {koopman_loss}, classifier_loss: {loss - koopman_loss}")
            running_loss = 0.0

    acc = evaluate(model, testloader, epoch)
    # c = conv(torch.stack(model.deq_history))
    # plt.plot(c)
    # plt.show()
    # print(c[-1])
    print(f"Accuracy after epoch {epoch + 1}: {acc}%")

print("Finished Training")
