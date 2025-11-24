import torch.nn as nn

class FanSpectrogramCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # (N, 1, F, T) -> (N, 16, F/2, T/2)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)), # pool both freq & time

            # (N, 16, F/2, T/2) -> (N, 32, F/4, T/4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # (N, 32, F/4, T/4) -> (N, 64, F/8, T/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)), # pool freq only, since the time dim may be small
        )

        # Global average pooling: (N, 64, H, W) -> (N, 64, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(), #(N, 64)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_classes), # I don't use softmax here since CrossEntropyLoss does it internally, and I don't care about probabilities but only the scores
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
