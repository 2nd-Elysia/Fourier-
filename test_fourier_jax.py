import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from PIL import Image
from pathlib import Path
import time
plt.rcParams['font.family'] = ['serif', 'SimSun']

path_images = Path("origin_images")
fname = "B5.bmp"
path = path_images.joinpath(fname)
image = Image.open(str(path)).convert('L')
x = jnp.array(image)

start_time = time.time()
f_trans = jnp.fft.fft2(x)
f_shift = jnp.fft.fftshift(f_trans)

magnitude = jnp.abs(f_shift)
log_magnitude = jnp.log(magnitude)

r = 250
@jax.jit
def set_center_zero(arr, r, out=False):
    center = (arr.shape[0] // 2, arr.shape[1] // 2)
    x, y = jnp.meshgrid(jnp.arange(arr.shape[1]), jnp.arange(arr.shape[0]))
    distance = jnp.sqrt((x-center[1])**2 + (y-center[0])**2)
    return jnp.where((distance < r) ^ out, arr, 0.0)

f_shift = set_center_zero(f_shift, r, out=True)
f_shift = set_center_zero(f_shift, 500, out =False)
f_ishift = jnp.fft.ifftshift(f_shift)
print("used time:", time.time() - start_time)

x_ = jnp.abs(jnp.fft.ifft2(f_ishift))
# x_ = ((x_ - x_.min()) * 255 / (x_.max() - x_.min())).astype('uint8')
x_ = x_.astype('uint8')
x_ = Image.fromarray(jax.device_get(x_))
x_.save(f"{fname}withr={r}.png")
fig = plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.imshow(x, cmap='gray')
plt.title("Origin")
plt.subplot(222)
plt.imshow(log_magnitude, cmap='gray')
plt.title("Fourier transform")
plt.subplot(223)
plt.imshow(jnp.log(jnp.abs(f_shift)), cmap='gray')
plt.title("Clip Fourier transform")
plt.subplot(224)
plt.imshow(x_, cmap='gray')
plt.title("Rebuild")
plt.savefig(f"images/grids{r}500.png", dpi=200)
plt.show()

