import numpy as np
import matplotlib.pyplot as plt


class Image:
	"""2D grid of pixels used to genourate image"""

	def __init__(self, width, height):
		"""Initalize with a x in (-1, 1) and y in (-1/aspect, 1/aspect) in the z-y plane."""
		ratio = float(width) / height
		self.xspace = np.linspace(-1, 1, width)
		self.yspace = np.linspace(-1 / ratio, 1 / ratio, height)
		self.img_grid = np.zeros((width, height, 3))



	def set_pixel_color(i, j, color):
		"""set the pixel at img[i][j] to color"""
		self.img_grid[i][j][0] = color[0]
		self.img_grid[i][j][1] = color[1]
		self.img_grid[i][j][2] = color[2]


	def genourate(self, objects):
		"""Create image data given the objects in view"""
		for i, x in enumerate(self.xspace):
			for j, y in enumerate(self.yspace):
				self.set_pixel_color(i, j, (1, 1, 0))


		plt.imsave("image.png", self.img_grid)

