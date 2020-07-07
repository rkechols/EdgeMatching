import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np


def patch_pair_to_tensor(combined_patch: np.ndarray) -> torch.Tensor:
	return torch.tensor(np.transpose(combined_patch.copy(), (2, 0, 1)), dtype=torch.float)


class PatchPairBoolNet(nn.Module):
	def __init__(self):
		super(PatchPairBoolNet, self).__init__()
		self.fc_in_size = 960

		self.conv1 = nn.Conv2d(3, 6, 3)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 3)
		self.fc1 = nn.Linear(self.fc_in_size, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 17)
		self.fc4 = nn.Linear(17, 1)

	def forward(self, x):
		# first dimension is how many things are in the mini-batch
		# second dimension is channels
		x = self.pool(f.relu(self.conv1(x)))
		x = self.pool(f.relu(self.conv2(x)))
		x = x.view(-1, self.fc_in_size)
		x = f.elu(self.fc1(x))
		x = f.elu(self.fc2(x))
		x = self.fc3(x)
		x = torch.sigmoid(self.fc4(x))
		return x.view(-1)

	def forward_numpy(self, x: np.ndarray, device):
		x = patch_pair_to_tensor(x)
		x = x[None, :, :, :]
		x = x.to(device)
		x = self.forward(x)
		return x.item()
