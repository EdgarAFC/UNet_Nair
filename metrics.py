import numpy as np
from scipy.optimize import curve_fit

# Metricas: Contraste, gcnr, SNR
def compute_metrics(cx, cz, r, bmode_output, grid):
    env_output = 10 ** (bmode_output / 20)

    r0 = r - 1 / 1000
    r1 = r + 1 / 1000
    r2 = np.sqrt(r0 ** 2 + r1 ** 2)

    dist = np.sqrt((grid[:, :, 0] - cx) ** 2 + (grid[:, :, 2] - cz) ** 2)
    roi_i = dist <= r0
    roi_o = (r1 <= dist) * (dist <= r2)

    # Compute metrics
    env_inner = env_output[roi_i]
    env_outer = env_output[roi_o]

    contrast_value = contrast(env_inner, env_outer)
    snr_value = snr(env_outer)
    gcnr_value = gcnr(env_inner, env_outer)
    cnr_value = cnr(env_inner, env_outer)

    # Calculate row brightness excluding cysts
    brightness1 = calculate_row_brightness_excluding_cysts(bmode_output, cx,cz, r)
    # Normalize the brightness values
    brightness1_normalized = (brightness1 - brightness1.min()) / (brightness1.max() - brightness1.min())
    # Fit the brightness decay and get decay rates
    decay_param = fit_brightness_decay(brightness1_normalized)
    return contrast_value, cnr_value, gcnr_value, snr_value, decay_param

# Function to calculate the average brightness of each row excluding cyst regions
def calculate_row_brightness_excluding_cysts(image, cx, cz, cyst_radii):
    cyst_centers = []
    cyst_centers.append(cx)
    cyst_centers.append(cz)
    rows, cols = image.shape
    brightness = np.zeros(rows)
    counts = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            exclude = False
            for center, radius in zip(cyst_centers, cyst_radii):
                if np.sqrt((i - center[0])**2 + (j - center[1])**2) <= radius:
                    exclude = True
                    break
            if not exclude:
                brightness[i] += image[i, j]
                counts[i] += 1
    # Avoid division by zero
    counts[counts == 0] = 1
    return brightness / counts

# Function to fit an exponential decay function
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# Function to fit the brightness decay curve
def fit_brightness_decay(brightness):
    rows = np.arange(len(brightness))
    popt, _ = curve_fit(exp_decay, rows, brightness, p0=(1, 0.001))
    return popt[1]

def contrast(img1, img2):
    return 20 * np.log10(img1.mean() / img2.mean())

# Compute contrast-to-noise ratio
def cnr(img1, img2):
    return (img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())

# Compute the generalized contrast-to-noise ratio
def gcnr(img1, img2):
    _, bins = np.histogram(np.concatenate((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))

def snr(img):
    return img.mean() / img.std()