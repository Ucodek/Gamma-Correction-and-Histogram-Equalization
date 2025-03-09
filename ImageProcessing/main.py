import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math


def gammaDüzeltme(image, gamma): # içeriye argüman olarak bir resim ve kullanıcının belirleyeceği bir gamma değeri alacak
    inv_gamma = gamma     # gamma değerini doğrudan kullanıyoruz
    # Normalizasyon işlemi için bir lookup table oluşturuyoruz
    table = np.array([(math.pow((i / 255.0), inv_gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Lookup table kullanarak resmi dönüştürüyoruz
    return cv2.LUT(image, table)
        
def custom_histogram_equalization(image):       # Histogram eşitleme işlemi
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])       # Histogramı hesaplıyoruz
    cdf = hist.cumsum()     # Kumulatif dağılım fonksiyonunu hesaplıyoruz
    cdf_normalized = cdf * 255 / cdf[-1]    # Normalizasyon işlemi
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape).astype(np.uint8)   # Eşitlenmiş resmi oluşturuyoruz
    return equalized_image  # Eşitlenmiş resmi döndürüyoruz

def apply_operations(image, gamma_value, order="gamma_first"):  # İki işlemi birleştirme işlemi
    if order == "gamma_first":  # Eğer kullanıcı gamma düzeltme işleminden önce histogram eşitleme işlemi yapılmasını isterse
        img_gamma = gammaDüzeltme(image, gamma_value)   # Gamma düzeltme işlemi uygulanıyor
        result = custom_histogram_equalization(img_gamma)   # Histogram eşitleme işlemi uygulanıyor
    else:   # Eğer kullanıcı histogram eşitleme işleminden önce gamma düzeltme işlemi yapılmasını isterse
        img_hist_eq = custom_histogram_equalization(image)  # Histogram eşitleme işlemi uygulanıyor
        result = gammaDüzeltme(img_hist_eq, gamma_value)    # Gamma düzeltme işlemi uygulanıyor
    return result

def display_results(original, gamma_corrected, hist_equalized, combined1, combined2):   # Sonuçları görselleştirme işlemi
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
    image_path = "Photo2.jpeg"  # Change this to your image path
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gamma_value = 1.5
    gamma_corrected = gammaDüzeltme(original_image, gamma_value)
    hist_equalized = custom_histogram_equalization(original_image)
    combined1 = apply_operations(original_image, gamma_value, "gamma_first")
    combined2 = apply_operations(original_image, gamma_value, "hist_first")
    
    display_results(original_image, gamma_corrected, hist_equalized, combined1, combined2)