# make fractals
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mandelbrot import MandelbrotSet

mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)

width, height = 512, 512
scale = 0.0075
GRAYSCALE = "L"

image = Image.new(mode=GRAYSCALE, size=(width,height))

for y in range(height):
    for x in range(width):
        c = scale * complex(x - width/2, height/2 - y)
        instability = 1 - mandelbrot_set.stability(c, smooth=True)
        image.putpixel((x,y),int(instability * 255))
image.show()
# def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
#     re = np.linspace(xmin, xmax, int(xmax-xmin)*pixel_density)
#     im = np.linspace(ymin, ymax, int(ymax-ymin)*pixel_density)
#     return re[np.newaxis,:] + im[:,np.newaxis] * 1j

# def is_stable(c,num_iterations):
#     z = 0
#     for _ in range(num_iterations):
#         z = z ** 2 + c
#         if abs(z) > 2:
#             return False;
#     return True

# def get_members(c, num_iterations):
#     mask = is_stable(c, num_iterations)
#     return c[mask]

# c = complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density=512)
# image = Image.fromarray(~is_stable(c, num_iterations=20))
# image.show()
# plt.gca().set_aspect("equal")
# plt.axis("off")
# plt.tight_layout
# plt.show()