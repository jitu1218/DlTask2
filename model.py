"""Shared CNN backbone with classification and occlusion heads."""

from torch import nn


class SignOcclusionCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        blocks = []
        incoming = 3
        for outgoing in (16, 32, 64):
            blocks += [
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            incoming = outgoing
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 4)))
        self.shared = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.25)
        )
        self.sign_head = nn.Linear(128, num_classes)
        self.occlusion_head = nn.Linear(128, 1)

    def forward(self, images):
        features = self.shared(self.features(images))
        return self.sign_head(features), self.occlusion_head(features).squeeze(1)
