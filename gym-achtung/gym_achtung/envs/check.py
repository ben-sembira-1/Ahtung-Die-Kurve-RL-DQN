from scipy import ndimage, misc
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def print_mat(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(mat[i][j], end=" ")
        print()


n_x = 200
n_y = 200
img = np.zeros((n_x, n_y))
img[20:50, 20:40] = 1
imr = Image.fromarray(img)

fig = plt.figure(figsize=(10, 3))
ax1, ax2 = fig.subplots(1, 2)
print(type(imr))
img_45 = imr.rotate(45)
# full_img_45 = ndimage.rotate(img, 45, reshape=True)
ax1.imshow(img)
ax1.set_axis_off()
ax2.imshow(img_45)
ax2.set_axis_off()
fig.set_tight_layout(True)
plt.show()
