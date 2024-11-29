import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def denoise_image(image):
    return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)


def apply_lpf(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    return cv2.filter2D(image, -1, kernel)


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())


def pooling_method(ssim_map):
    center_weight = 2
    height, width = ssim_map.shape
    center_x, center_y = width // 2, height // 2
    weights = np.ones_like(ssim_map)
    weights[center_y - 10:center_y + 10, center_x - 10:center_x + 10] = center_weight
    return np.average(ssim_map, weights=weights)


# Load original image
image_path = 'images/original_image.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noisy_image = add_gaussian_noise(original_image)
cv2.imwrite('images/noisy_image.png', noisy_image)

# Denoise image
denoised_image = denoise_image(noisy_image)
cv2.imwrite('images/denoised_image.png', denoised_image)

# Apply LPF
lpf_image = apply_lpf(noisy_image)
cv2.imwrite('images/lpf_image.png', lpf_image)

# Calculate PSNR and SSIM
psnr_value = calculate_psnr(original_image, denoised_image)
ssim_value, ssim_map = calculate_ssim(original_image, denoised_image, full=True)
pooled_ssim = pooling_method(ssim_map)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
print(f"Pooled SSIM: {pooled_ssim}")

# Display SSIM map and absolute error map
cv2.imshow("SSIM Map", ssim_map)
cv2.imshow("Absolute Error Map", cv2.absdiff(original_image, denoised_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
