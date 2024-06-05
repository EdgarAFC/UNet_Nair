import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Metricas: Contraste, gcnr, SNR
def compute_metrics(cx, cz, r, bmode_output, grid, region):
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
    brightness1 = calculate_row_brightness_excluding_cysts(bmode_output, cx,cz, r, region)
    # print(brightness1)
    # Normalize the brightness values
    brightness1_normalized = (brightness1 - brightness1.min()) / (brightness1.max() - brightness1.min())
    
    # # Crear el eje x con números de píxeles de 0 a 799
    # x = list(range(800))
    # # Crear la figura y los ejes
    # fig, ax = plt.subplots()
    # # Plotear los datos
    # ax.plot(x, brightness1, label='Valores de la lista')
    # # Etiquetas para los ejes
    # ax.set(xlabel='Número de píxeles', ylabel='Valores', title='Gráfico de 800 elementos')
    # # Añadir una leyenda
    # ax.legend()
    # # Mostrar la grilla (opcional)
    # ax.grid(True)
    # # Mostrar el gráfico
    # plt.show()
    
    # Fit the brightness decay and get decay rates
    decay_param = fit_brightness_decay(brightness1_normalized)

    value_contrast_att = contrast_att(region, env_output)
    return contrast_value, cnr_value, gcnr_value, snr_value, decay_param, value_contrast_att

# Function to calculate the average brightness of each row excluding cyst regions
def calculate_row_brightness_excluding_cysts(image, cx, cz, radius, region):
    center = [cx,cz]
    rows, cols = image.shape
    brightness = np.zeros(rows)
    counts = np.zeros(rows)
    cols = region
    for i in range(rows):
        for j in cols:
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
    # return (img1.mean() / img2.mean())

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

def contrast_att(region, array):
    
    contrast_n = []

    # Define the dimensions of the rectangles
    row_start1 = 300         # starting row for the first rectangle
    row_end1 = 400         # ending row for the first rectangle
    width = 4             # width of the rectangles
    column_step = 5       # step size for moving the columns

    # Loop through the array to select the rectangles
    for col_start in range(region[0], region[-1] - width + 1, column_step):
        col_end = col_start + width

        # Define the second rectangle's starting and ending rows
        row_start2 = row_end1 + 200
        row_end2 = row_start2 + 100

        # Ensure that the row indices do not exceed the array dimensions
        if row_end2 > 800:
            break

        # Extract the first region of interest (ROI1)
        roi1 = array[row_start1:row_end1, col_start:col_end]

        # Extract the second region of interest (ROI2)
        roi2 = array[row_start2:row_end2, col_start:col_end]

        contrast_value = contrast(roi2, roi1)
        contrast_n.append(contrast_value)

    contrast_final = sum(contrast_n)/len(contrast_n)

    print('Contrast:', contrast_final)

    return contrast_final