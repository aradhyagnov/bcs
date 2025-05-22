
# Edge Detection Report
## Methods
### 1. Sobel Operator
The Sobel operator uses a pair of 3x3 convolution kernels to compute gradients in the
horizontal and vertical directions. The combined gradient magnitude highlights strong
edges.
### 2. Scharr Operator
An enhanced version of Sobel, optimized for higher accuracy with less noise sensitivity.
It uses larger weights in the gradient direction.
### 3. Feldman Operator
A simple 2x2 approximation of the Roberts cross operator. It detects diagonal edges but
with less detail due to its small kernel size.
### 4. Canny Edge Detection
This method includes:
- Gaussian Blurring
- Gradient Calculation using Sobel
- Non-Maximum Suppression
- Double Thresholding
- Edge Tracking by Hysteresis
It is known for its accuracy and clean edge maps.
---
## Results
### Sobel
- Highlights general edges clearly.
- Sensitive to noise if not blurred beforehand.
### Scharr
- Stronger edge response than Sobel.
- Better accuracy at the cost of some smoothing.
### Feldman
- Fast and simple, but less detail.
- Not suitable for complex edge maps.
### Canny
- Best visual quality with fine edge details.
- Threshold tuning is essential.
