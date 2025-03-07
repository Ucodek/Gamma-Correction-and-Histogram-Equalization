import cv2
import numpy as np 
import matplotlib.pyplot as plt


def gammaDüzeltme(image, gamma): # içeriye argüman olarak bir resim ve kullanıcının belirleyeceği bir gamma değeri alacak
    inv_gamma = 1.0 / gamma     # gamma değerinin tersini alıyoruz
    # Normalizasyon işlemi için bir lookup table oluşturuyoruz
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Lookup table kullanarak resmi dönüştürüyoruz
    return cv2.LUT(image, table)
    
def custom_histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape).astype(np.uint8)
    return equalized_image

def apply_operations(image, gamma_value, order="gamma_first"):
    if order == "gamma_first":
        img_gamma = gammaDüzeltme(image, gamma_value)
        result = custom_histogram_equalization(img_gamma)
    else:
        img_hist_eq = custom_histogram_equalization(image)
        result = gammaDüzeltme(img_hist_eq, gamma_value)
    return result

def display_results(original, gamma_corrected, hist_equalized, combined1, combined2):
    images = [original, gamma_corrected, hist_equalized, combined1, combined2]
    titles = ["Original", "Gamma Corrected", "Histogram Equalized", 
              "Gamma -> Hist Equalized", "Hist Equalized -> Gamma"]
    
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
        plt.subplot(2, 5, i+6)
        plt.hist(images[i].ravel(), 256, [0, 256])
        plt.title(titles[i])
    plt.show()



# Main block to read the image, apply gamma correction, and display results
if __name__ == "__main__":
    image_path = "Photo.jpeg"  # Change this to your image path
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gamma_value = 2.0
    gamma_corrected = gammaDüzeltme(original_image, gamma_value)
    hist_equalized = custom_histogram_equalization(original_image)
    combined1 = apply_operations(original_image, gamma_value, "gamma_first")
    combined2 = apply_operations(original_image, gamma_value, "hist_first")
    
    display_results(original_image, gamma_corrected, hist_equalized, combined1, combined2)