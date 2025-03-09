import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def display_results(original, gamma_corrected_images, gamma_values):
    plt.figure(figsize=(15, 10))
    titles = ["Original"] + [f"(γ={gamma})" for gamma in gamma_values]
    
    images = [original] + gamma_corrected_images
    for i in range(len(images)):
        plt.subplot(2, len(images), i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
        plt.subplot(2, len(images), i+len(images)+1)
        plt.hist(images[i].ravel(), 256, [0, 256])
        plt.title(titles[i] + " Histogram")
    
    plt.show()

if __name__ == "__main__":
    image_path = "Photo2.jpeg"  # Change this to your image path
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    gamma_corrected_images = [gamma_correction(original_image, gamma) for gamma in gamma_values]
    
    display_results(original_image, gamma_corrected_images, gamma_values)