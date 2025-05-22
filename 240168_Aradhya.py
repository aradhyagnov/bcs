import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and convert to grayscale
image_path = "input.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Manual convolution function
def convolve(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float32)

# Scharr kernels
scharr_x = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]], dtype=np.float32)
scharr_y = np.array([[-3, -10, -3],
                     [0,   0,  0],
                     [3,  10,  3]], dtype=np.float32)

# Feldman operator (a variant of Roberts)
feldman_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)
feldman_y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

# Compute edge magnitudes
def edge_magnitude(img, kx, ky):
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

sobel_edges = edge_magnitude(img, sobel_x, sobel_y)
scharr_edges = edge_magnitude(img, scharr_x, scharr_y)
feldman_edges = edge_magnitude(img, feldman_x, feldman_y)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.title("Sobel")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(scharr_edges, cmap='gray')
plt.title("Scharr")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(feldman_edges, cmap='gray')
plt.title("Feldman")
plt.axis("off")

plt.tight_layout()
plt.show()

# 1. Gaussian Blur
def gaussian_kernel(size=5, sigma=1.4):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def convolve(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return output


blurred = convolve(img, gaussian_kernel(5, 1.4))

# 2. Sobel Gradient
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

Gx = convolve(blurred, sobel_x)
Gy = convolve(blurred, sobel_y)
magnitude = np.hypot(Gx, Gy)
magnitude = magnitude / magnitude.max() * 255
angle = np.arctan2(Gy, Gx) * 180 / np.pi
angle[angle < 0] += 180


# 3. Non-Maximum Suppression
def non_max_suppression(mag, ang):
    Z = np.zeros_like(mag, dtype=np.float32)
    h, w = mag.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 255
            r = 255
            angle = ang[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif (22.5 <= angle < 67.5):
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif (67.5 <= angle < 112.5):
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif (112.5 <= angle < 157.5):
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0
    return Z


nms = non_max_suppression(magnitude, angle)


# 4. Double Threshold
def threshold1(img, low_ratio=0.05, high_ratio=0.1):
    high = img.max() * high_ratio
    low = high * low_ratio
    strong = 255
    weak = 75

    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def threshold(img, low_ratio=0.05, high_ratio=0.15):
    high = img.max() * high_ratio
    low = high * low_ratio
    strong = 255
    weak = 75

    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong
thresholded, weak, strong = threshold1(nms)
thresholded1, weak1, strong1 = threshold(nms)

# 5. Edge Tracking by Hysteresis
def hysteresis(img, weak, strong=255):
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == weak:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


canny_edges = hysteresis(thresholded, weak, strong)
canny_edges1 = hysteresis(thresholded1, weak1, strong1)
# Display result
plt.figure(figsize=(8, 5))
plt.imshow(thresholded, cmap='gray')
plt.title("Canny Edge Detection (without hysterisis)")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny Edge Detection (low high threshold)")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(canny_edges1, cmap='gray')
plt.title("Canny Edge Detection (higher high threshold)")
plt.axis("off")
plt.show()
#we can see that things like the eye shape are not clearly visible in canny with high
# threshold whereas it is seen in canny with low high threshhold
