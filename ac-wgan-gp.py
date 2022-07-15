# this code is based on https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py, which is released under the MIT licesne

# imports
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import os
import sys
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 64
n_channels = 3
learning_rate = 0.0002
lambda_gp = 10
noise_dim = 128
dataset = "cifar10"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

generator_iters = 100000
critic_iter = 5

transforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(n_channels)], [0.5 for _ in range(n_channels)]
        ),
    ]
)

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for i, (x, labels) in enumerate(iterable):
            yield x, labels


# you may use cifar10 or stl10 datasets
if dataset == "cifar10":
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            "drive/My Drive/training/cifar10",
            train=True,
            download=True,
            transform=transforms,
        ),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    class_names = [
        "airplane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

# stl10 has larger images which are much slower to train on. You should develop your method with CIFAR-10 before experimenting with STL-10
if dataset == "stl10":
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.STL10(
            "drive/My Drive/training/stl10",
            split="train",
            download=True,
            transform=transforms,
        ),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )

    class_names = [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]  # these are slightly different to CIFAR-10


train_iterator = iter(cycle(train_loader))

# PyTorch (to my knowledge) has no implementation of conditional batch normalisation (unlike tensorflow).
# Therefore this third-party implementation from a github comment section was used and appears to work as expected.
# this code is taken from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775 which is unlisenced
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resample, num):
        super().__init__()

        if resample == "down":

            self.res = nn.Sequential(
                nn.LayerNorm(
                    [in_channels, (32 // (2 ** (num - 1))), (32 // (2 ** (num - 1)))]
                ),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.LayerNorm(
                    [out_channels, (32 // (2 ** (num - 1))), (32 // (2 ** (num - 1)))]
                ),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.AvgPool2d(2),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.AvgPool2d(2),
            )

        elif resample == None:

            self.res = nn.Sequential(
                nn.LayerNorm(
                    [in_channels, (32 // (2 ** (num - 1))), (32 // (2 ** (num - 1)))]
                ),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.LayerNorm(
                    [out_channels, (32 // (2 ** (num - 1))), (32 // (2 ** (num - 1)))]
                ),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
            )
        self.initialize()

    def initialize(self):
        for m in self.res.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):

        return self.res(x) + self.shortcut(x)


class OptDiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2),
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2), nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )
        self.initialize()

    def initialize(self):
        for m in self.res.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):

        return self.res(x) + self.shortcut(x)


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_labels):
        super().__init__()

        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_labels)
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_labels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
        )

    def forward(self, x, y):
        x1 = x.detach().clone()
        x2 = self.cbn1(x, y)
        x2 = F.relu(x2)
        x2 = self.conv1(self.up(x2))
        x2 = self.cbn2(x2, y)
        x2 = F.relu(x2)

        return (self.conv2(x2) + self.shortcut(x1)), y


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(noise_dim, 4 * 4 * 128)

        self.G1 = GenBlock(128, 128, 10)
        self.G2 = GenBlock(128, 128, 10)
        self.G3 = GenBlock(128, 128, 10)
        self.cbn = ConditionalBatchNorm2d(128, 10)

        self.conv = nn.Conv2d(128, 3, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.G1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        for m in self.G2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        for m in self.G3.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x, y):
        x = self.linear(x)
        x = x.view(-1, 128, 4, 4)

        x, _ = self.G1(x, y)
        x, _ = self.G2(x, y)
        x, _ = self.G3(x, y)

        x = self.cbn(x, y)

        x = F.relu(x)
        x = nn.Tanh()(self.conv(x))

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptDiscBlock(3, 128, num=1),
            DiscBlock(128, 128, "down", num=2),
            DiscBlock(128, 128, None, num=3),
            DiscBlock(128, 128, None, num=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Linear(128, 1)
        self.ac = nn.Sequential(nn.Linear(128, 10), nn.Softmax(dim=1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        wgan_out = self.linear(x)
        ac_out = self.ac(x)
        return wgan_out, ac_out


def gradient_penalty(D, real, fake, labels):
    alpha = torch.rand(batch_size, 1, 1, 1).to(real.device)
    alpha = alpha.expand(real.size())

    interpolates = real + (alpha * (fake - real))
    interpolates.requires_grad_(True)
    interpolates_d, _ = D(interpolates)
    gradients = torch.autograd.grad(
        inputs=interpolates,
        outputs=interpolates_d,
        grad_outputs=torch.ones_like(interpolates_d),
        create_graph=True,
        retain_graph=True,
    )[0]

    normalised_grad = torch.norm(torch.flatten(gradients, start_dim=1), dim=1)
    gp = torch.mean((normalised_grad - 1) ** 2)
    return gp


G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

learning_rate = 2e-4
b1 = 0.0
b2 = 0.999
batch_size = 64

g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(b1, b2))
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(b1, b2))

g_sched = optim.lr_scheduler.LambdaLR(
    g_optimizer, lambda step: 1 - step / generator_iters
)
d_sched = optim.lr_scheduler.LambdaLR(
    d_optimizer, lambda step: 1 - step / generator_iters
)

data = cycle(train_loader)

AC_D_SCALE = 1
AC_G_SCALE = 0.1

for g_iter in range(generator_iters):

    for p in D.parameters():
        p.requires_grad = True

    for d_iter in range(critic_iter):
        D.zero_grad()

        images, labels = data.__next__()

        if images.size()[0] != batch_size:
            continue

        z = torch.randn(batch_size, noise_dim)

        images, labels, z = images.to(device), labels.to(device), z.to(device)

        d_loss_real, pred_real_labels = D(images)
        d_loss_real = d_loss_real.mean()

        fake_images = G(z, labels)
        d_loss_fake, pred_fake_labels = D(fake_images)
        d_loss_fake = d_loss_fake.mean()

        gp = gradient_penalty(D, images, fake_images, labels)
        d_loss = d_loss_fake - d_loss_real + (lambda_gp * gp)

        aux_loss = F.cross_entropy(
            torch.cat((pred_real_labels, pred_fake_labels), 0),
            torch.cat((labels, labels), 0),
        )

        d_loss += AC_D_SCALE * aux_loss

        d_loss.backward()
        d_optimizer.step()

    for p in D.parameters():
        p.requires_grad = False

    G.zero_grad()

    z = torch.randn(batch_size * 2, noise_dim).to(device)
    labels = torch.randint(0, 10, (batch_size * 2,)).to(device)
    fake_images = G(z, labels)
    g_loss, pred_gen_labels = D(fake_images)
    g_loss = -g_loss.mean()

    aux_loss = F.cross_entropy(pred_gen_labels, labels)

    g_loss += AC_G_SCALE * aux_loss

    g_loss.backward()
    g_optimizer.step()

    g_sched.step()
    d_sched.step()

    if g_iter % 500 == 0:
        print(
            f"Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}, d_loss_fake: {d_loss_fake}, d_loss_real: {d_loss_real}"
        )
        with torch.no_grad():

            plt.imshow(
                torchvision.utils.make_grid(fake_images[:batch_size], normalize=True)
                .cpu()
                .data.permute(0, 2, 1)
                .contiguous()
                .permute(2, 1, 0)
            )
            plt.show()
            if not os.path.isdir(f"training_images"):
                os.makedirs(f"training_images")
            plt.savefig(f"training_images/g_iter{g_iter}.png")


with torch.no_grad():

    for i in range(10):

        z = torch.randn(batch_size, noise_dim).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)
        fake_images = G(z, labels)

        plt.imshow(
            torchvision.utils.make_grid(fake_images[:batch_size], normalize=True)
            .cpu()
            .data.permute(0, 2, 1)
            .contiguous()
            .permute(2, 1, 0)
        )
        plt.show()
        if not os.path.isdir(f"trained_images"):
            os.makedirs(f"trained_images")
        plt.savefig(f"trained_images/img{i}.png")

def generate_latent_points(noise_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(noise_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, noise_dim)
    return z_input

def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


with torch.no_grad():

    for iter in range(10):
        interpolated = torch.Tensor().to(device)

        for _ in range(10):

            pts = generate_latent_points(128, 100)

            i = torch.Tensor(interpolate_points(pts[0], pts[1])).to(device)

            interpolated = torch.cat((interpolated, i))

        interpolated = interpolated.reshape(100, 128)

        l = torch.tensor([]).type(torch.IntTensor).to(device)

        for i in range(10):

            l = torch.cat(
                [l, torch.tensor(10 * [int(i)]).type(torch.IntTensor).to(device)]
            )

        imgs = G(interpolated, l)

        plt.rcParams["figure.dpi"] = 175

        plt.imshow(
            torchvision.utils.make_grid(imgs, nrow=10, normalize=True)
            .cpu()
            .data.permute(0, 2, 1)
            .contiguous()
            .permute(2, 1, 0),
            cmap=plt.cm.binary,
        )
        plt.show()
        if not os.path.isdir(f"interpolations"):
            os.makedirs(f"interpolations")
        plt.savefig(f"interpolations/{iter}.png")
