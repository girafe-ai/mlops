import torch


class SimpleClassifier(torch.nn.Module):
    """Linear model for classification of images"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 96 * 96, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class ConvClassifier(torch.nn.Module):
    """Convolutional model for classification of images"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(6),
            # torch.nn.Dropout(0.3),
            torch.nn.Flatten(),
            torch.nn.Linear(4608, 96),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(96, num_classes, bias=False),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
