import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
plt.rcParams['font.family'] = ['serif', 'SimSun']
import time

path_images = Path("images")
fname = "222.png"
path = path_images.joinpath(fname)
image = Image.open(str(path)).convert('L')
x = np.array(image)
print("origin mean:", x.mean())

start_time = time.time()
f_trans = np.fft.fft2(x)#快速傅里叶变换
f_shift = np.fft.fftshift(f_trans)#将频率零点位于图像的中心

magnitude = np.abs(f_shift)
log_magnitude = np.log(magnitude)

r = 250
def set_center_zero(arr, r, out=False):
    center = (arr.shape[0] // 2, arr.shape[1] // 2)
    x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    distance = np.sqrt((x-center[1])**2 + (y-center[0])**2)
    return np.where((distance < r) ^ out, arr, 0.0)

f_shift = set_center_zero(f_shift, r, out=False)

f_ishift = np.fft.ifftshift(f_shift)

x_ = np.abs(np.fft.ifft2(f_ishift)).astype('uint8')
print("used time:", time.time() - start_time)

#x_ = ((x_ - x_.min()) * 255 / (x_.max() - x_.min())).astype('uint8')  # transform mean: 127.9349589223552
print("transform mean:", x_.mean())
rebuild_image = Image.fromarray(x_)
rebuild_image.save(f"{fname}withr={r}.png")

# fig = plt.figure(figsize=(15, 10))

# plt.subplot(221)
# plt.imshow(x, cmap='gray')
# plt.title("Origin")

# plt.subplot(222)
# plt.imshow(log_magnitude, cmap='gray')
# plt.title("Fourier transform")

# plt.subplot(223)
# plt.imshow(np.log(np.abs(f_shift)), cmap='gray')
# plt.title("Clip Fourier transform")

# plt.subplot(224)
# plt.imshow(x_, cmap='gray')
# plt.title("Rebuild")

# plt.savefig(f"images/grids_high{r}.png")
# plt.show()