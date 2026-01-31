import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        """
        x: [batch, input_dim]
        returns: [batch, input_dim * (2*num_frequencies + 1)]
        """
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


# ---------- Custom Feature Extractor für Dict ----------
class DictPEFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        num_frequencies: int = 8,
    ):
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict()
        total_dim = 0

        for key, subspace in observation_space.spaces.items():
            if not isinstance(subspace, gym.spaces.Box):
                raise ValueError(f"Key {key} ist kein Box-Space!")

            input_dim = subspace.shape[0]
            pe = PositionalEncoding(input_dim, num_frequencies=num_frequencies)
            pe_output_dim = input_dim * (2 * num_frequencies + 1)

            # für jeden Key ein eigenes MLP nach dem Encoding
            self.extractors[key] = nn.Sequential(
                pe,
                nn.Linear(pe_output_dim, 128),
                nn.ReLU(),
            )
            total_dim += 128

        # gemeinsames MLP nach dem Concatenieren
        self.final_mlp = nn.Sequential(nn.Linear(total_dim, features_dim), nn.ReLU())

    def forward(self, observations):
        encoded = []
        for key, extractor in self.extractors.items():
            x = observations[key]
            encoded.append(extractor(x))
        concat = torch.cat(encoded, dim=-1)
        return self.final_mlp(concat)
