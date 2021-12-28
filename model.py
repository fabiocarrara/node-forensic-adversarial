import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

class ODEfunc(nn.Module):

    def __init__(self, dim, norm, act):
        super(ODEfunc, self).__init__()

        self.activ = act()
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)
        self.norm3 = norm(dim)
        self.conv1 = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, padding=1)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        n, _, h, w = x.shape
        t = t.expand(n, 1, h, w)

        x = self.norm1(x)
        x = self.activ(x)

        x = torch.cat((x, t), dim=1)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)

        x = torch.cat((x, t), dim=1)
        x = self.conv2(x)
        x = self.norm3(x)

        return x

class ODENet(nn.Module):
    def __init__(
        self, 
        in_channels=3,
        num_outputs=1,
        num_filters=256,
        tol=1e-3,
        t_final=1,
        method='dopri5',
        adjoint=True,
        norm='group',
        activation='gelu',
        dropout=0.2,
    ):
        super(ODENet, self).__init__()

        self.tol = tol
        self.t = torch.tensor((0, t_final), dtype=torch.float32)
        self.method = method
        self.odeint = odeint_adjoint if adjoint else odeint

        norm = {
            'group': lambda dim: nn.GroupNorm(min(32, dim), dim),
            'batch': lambda dim: nn.BatchNorm2d(dim, track_running_stats=False),
        }[norm]

        activation = {
            'relu': lambda: nn.ReLU(),
            'gelu': lambda: nn.GELU()
        }[activation]

        self.stem = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1)
        self.odefunc = ODEfunc(num_filters, norm, activation)
        self.head = nn.Sequential(
            norm(num_filters),
            activation(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pool
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(num_filters, num_outputs)
        )
    
    def forward(self, x):
        # reset function evaluation counter at each forward pass
        self.odefunc.nfe = 0
        
        x = self.stem(x)
        _, x = self.odeint(self.odefunc, x, self.t, method=self.method, rtol=self.tol, atol=self.tol)
        x = self.head(x)

        return x
    
    @property
    def nfe(self):
        return self.odefunc.nfe


if __name__ == "__main__":
    device = 'cuda'
    net = ODENet(adjoint=True).to(device)
    x = torch.randn(2, 3, 64, 64).to(device)
    x.requires_grad = True
    y = net(x)
    y.mean().backward()
    print(x.grad)
    print(y, y.shape)
