import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

# Fungsi untuk memuat gambar dari folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                labels.append(label)  # Tambahkan label: "Segar" atau "Busuk"
            else:
                print(f"File {filename} tidak dapat dimuat!")
    return images, labels

# Fungsi untuk ekstraksi fitur GLCM
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

# Fungsi untuk ekstraksi fitur warna (HSV)
def extract_color_features(image):
    rgb_mean = np.mean(image, axis=(0, 1))  # Rata-rata RGB
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv, axis=(0, 1))  # Rata-rata HSV
    return list(rgb_mean) + list(hsv_mean)

# Fungsi untuk menambahkan teks pada gambar
def add_text_to_image(image, text, position=(10, 20), font_scale=0.4, color=(0, 0, 0)):
    return cv2.putText(
        image.copy(),
        text,
        position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=color,
        thickness=1
    )

# Fungsi untuk memproses gambar dan mengekstrak fitur
def preprocess_and_extract_features(image, label, original_size, resize_dim):
    # Ekstraksi fitur GLCM
    glcm_features = extract_glcm_features(image)
    glcm_text = f"GLCM: Ctr={glcm_features[0]:.2f}, Dsm={glcm_features[1]:.2f}"

    # Ekstraksi fitur warna
    color_features = extract_color_features(image)
    color_text = f"RGB-HSV: {', '.join(f'{val:.1f}' for val in color_features)}"

    # Tambahkan label, ukuran asli, dan ukuran hasil resize ke gambar
    label_text = f"Label: {label}"
    original_size_text = f"Original Size: {original_size[0]}x{original_size[1]}"
    resize_text = f"Resized to: {resize_dim[0]}x{resize_dim[1]}"

    return glcm_features, color_features, glcm_text, color_text, label_text, original_size_text, resize_text

def main():
    # Menentukan path folder gambar
    folder_segar = 'data/Segar'
    folder_busuk = 'data/Busuk'

    print(f"Memuat gambar dari folder: {folder_segar}")
    images_segar, labels_segar = load_images_from_folder(folder_segar, label="Segar")
    print(f"Jumlah gambar yang dimuat dari folder Segar: {len(images_segar)}")

    print(f"Memuat gambar dari folder: {folder_busuk}")
    images_busuk, labels_busuk = load_images_from_folder(folder_busuk, label="Busuk")
    print(f"Jumlah gambar yang dimuat dari folder Busuk: {len(images_busuk)}")

    # Gabungkan semua gambar dan label dari kedua folder
    all_images = images_segar + images_busuk
    all_labels = labels_segar + labels_busuk

    # Buat list untuk menyimpan fitur yang diekstraksi
    features_data = []

    # Tentukan ukuran gambar yang akan digunakan untuk resize
    target_size = (224, 224)  # Ukuran target untuk semua gambar

    # Proses semua gambar
    for index, image in enumerate(all_images):
        label = all_labels[index]
        original_size = image.shape[:2]  # Ukuran asli gambar (height, width)

        # Ekstraksi fitur
        glcm_features, color_features, glcm_text, color_text, label_text, original_size_text, resize_text = preprocess_and_extract_features(
            image, label, original_size, target_size)

        # Resize gambar asli dan crop untuk diproses
        h, w, _ = image.shape
        cropped_image = image[h//4:3*h//4, w//4:3*w//4]  # Crop tengah 50%
        resized_image = cv2.resize(image, target_size)  # Resize ke 224x224
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Grayscale

        # Resize cropped image dan grayscale image agar ukurannya konsisten
        cropped_image_resized = cv2.resize(cropped_image, target_size)
        gray_image_resized = cv2.resize(gray_image, target_size)

        # Menambahkan teks ke gambar
        original_image_with_text = add_text_to_image(resized_image, label_text)
        original_image_with_text = add_text_to_image(original_image_with_text, original_size_text, position=(10, 40))
        original_image_with_text = add_text_to_image(original_image_with_text, resize_text, position=(10, 60))
        resized_image_with_text = add_text_to_image(resized_image, "Resized")
        cropped_image_with_text = add_text_to_image(cropped_image_resized, "Cropped")
        gray_image_with_text = add_text_to_image(cv2.cvtColor(gray_image_resized, cv2.COLOR_GRAY2BGR), "Grayscale")

        # Tambahkan teks GLCM dan RGB-HSV ke gambar
        original_image_with_text = add_text_to_image(original_image_with_text, glcm_text, position=(10, 80), font_scale=0.3)
        original_image_with_text = add_text_to_image(original_image_with_text, color_text, position=(10, 100), font_scale=0.3)

        # Gabungkan gambar menjadi satu (resize semua gambar untuk memiliki ukuran yang sama)
        combined_image = np.hstack((
            original_image_with_text,
            resized_image_with_text,
            cropped_image_with_text,
            gray_image_with_text
        ))

        # Tampilkan gambar yang sudah diproses
        cv2.imshow(f'Processed Image: {label}', combined_image)

        # Simpan fitur untuk Excel
        features_data.append([label] + list(original_size) + glcm_features + color_features)

        # Tunggu input keyboard
        key = cv2.waitKey(0)
        if key == ord('q'):  # Tekan 'q' untuk keluar
            break

    # Membuat DataFrame dengan pandas
    columns = ['Label', 'Original Height', 'Original Width', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'R_mean', 'G_mean', 'B_mean', 'H_mean', 'S_mean', 'V_mean']
    df = pd.DataFrame(features_data, columns=columns)

    # Simpan ke file Excel
    output_file = 'glcm_features_with_sizes.xlsx'
    df.to_excel(output_file, index=False)

    print(f"Data fitur GLCM dan warna telah disimpan di {output_file}")

    cv2.destroyAllWindows()  # Tutup semua jendela gambar

if __name__ == "__main__":
    main()
