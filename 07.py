import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps

image = Image.open('./images/noise.jpg')
min_filter = image.filter(ImageFilter.MinFilter(3))
max_filter = min_filter.filter(ImageFilter.MaxFilter(3))

# Apply Grayscale and Edge detection
grayscale = ImageOps.grayscale(max_filter)
edges = grayscale.filter(ImageFilter.FIND_EDGES)

# Show the original and processed images
plt.subplot(1, 2, 1), plt.imshow(image)
plt.subplot(1, 2, 2), plt.imshow(min_filter)
plt.show()

plt.subplot(1, 2, 1), plt.imshow(image)
plt.subplot(1, 2, 2), plt.imshow(max_filter)
plt.show()

plt.subplot(1, 2, 1), plt.imshow(image)
plt.subplot(1, 2, 2), plt.imshow(grayscale)
plt.show()

plt.subplot(1, 2, 1), plt.imshow(image)
plt.subplot(1, 2, 2), plt.imshow(edges)
plt.show()