import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Benutzerdefinierter Feature Extractor
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=311)

        # Bild-Teil des Observations
        self.cnn = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=3, stride=2, padding=1
            ),  # Erste Convolution-Schicht
            nn.ReLU(),
            nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # Zweite Convolution-Schicht
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # Zweite Convolution-Schicht
            nn.ReLU(),
            nn.Conv2d(
                64, 64, kernel_size=3, stride=2, padding=1
            ),  # Zweite Convolution-Schicht
            nn.ReLU(),
        )

        self.fc_channels = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU(),
        )

        # Gelenk-Teil des Observations
        self.fc_joints = nn.Sequential(
            nn.Linear(7, 32),  # Fully-Connected-Schicht für die Gelenke
            nn.ReLU(),
            nn.Linear(32, 64),  # Fully-Connected-Schicht für die Gelenke
            nn.ReLU(),
        )

    def forward(self, obs: dict):
        # Bild-Feature extrahieren
        image = torch.tensor(obs["image"]).transpose(-1, -3)  # Normalisieren
        x_image = self.cnn(image)

        x_channels = self.fc_channels(x_image.transpose(-1, -3)).flatten(start_dim=1)

        # Gelenk-Feature extrahieren
        joints = obs["joints"]
        x_joints = self.fc_joints(joints)

        # Zusammenführen der Bild- und Gelenk-Features
        x = torch.cat([x_channels, x_joints], dim=1)

        return x
