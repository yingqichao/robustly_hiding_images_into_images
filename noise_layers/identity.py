import torch
import torch.nn as nn


class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the image
	"""

	def __init__(self):
		super(Identity, self).__init__()
		self.name = "Identity"

	def forward(self, image):
		# image, cover_image = image_and_cover
		return image
