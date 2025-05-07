import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd  # Untuk ekspor ke Excel

# Path ke folder data
fresh_folder = 'data/Segar'   # Gambar kol segar
rotten_folder = 'data/Busuk'  # Gambar kol busuk
test_folder = 'data/Testing'  # Gambar untuk pengujian

# 1. Preprocessing Data: Ekstraksi Fitur
def extract_glcm_features(image):
    """Ekstrak fitur GLCM dari gambar grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_color_features(image):
    """Ekstrak fitur RGB dan HSV"""
    rgb_mean = np.mean(image, axis=(0, 1))  # Rata-rata RGB
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv, axis=(0, 1))  # Rata-rata HSV
    return list(rgb_mean) + list(hsv_mean)

def extract_features(image):
    """Gabungkan fitur GLCM dan warna"""
    glcm_features = extract_glcm_features(image)
    color_features = extract_color_features(image)
    return glcm_features + color_features

# 2. Load Data Training dan Testing
def load_images(folder, label=None):
    """Load gambar dari folder dan ekstrak fitur"""
    features, labels, filenames = [], [], []
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        image = cv2.imread(filepath)
        if image is not None:
            features.append(extract_features(image))
            filenames.append(file)
            if label is not None:  # Untuk data training
                labels.append(label)
            elif 'segar' in file.lower():  # Jika gambar mengandung 'segar' (untuk testing)
                labels.append(0)  # Kol segar
            elif 'busuk' in file.lower():  # Jika gambar mengandung 'busuk' (untuk testing)
                labels.append(1)  # Kol busuk
    return features, labels, filenames

# 3. Main Program
if __name__ == '__main__':
    # Load data training
    fresh_features, fresh_labels, _ = load_images(fresh_folder, 0)  # 0 untuk segar
    rotten_features, rotten_labels, _ = load_images(rotten_folder, 1)  # 1 untuk busuk

    # Gabungkan data training
    X_train = np.array(fresh_features + rotten_features)
    y_train = np.array(fresh_labels + rotten_labels)

    # Normalisasi fitur
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3)  # K=3
    knn.fit(X_train, y_train)

    # Load data testing
    test_features, test_labels, test_filenames = load_images(test_folder)

    if not test_features:
        print("Tidak ada gambar pada folder test!")
        exit()

    X_test = np.array(test_features)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)  # Mengubah menjadi 2D jika perlu
    X_test = scaler.transform(X_test)
    y_test = np.array(test_labels)

    # Cek jika data pengujian valid
    if len(y_test) == 0:
        print("Tidak ada label pada data pengujian!")
        exit()

    # Prediksi
    y_pred = knn.predict(X_test)

    # Evaluasi
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Segar', 'Busuk'])
    print(report)

    # Ekspor hasil pengujian ke Excel
    results_file = 'testing_results.xlsx'
    results_df = pd.DataFrame({
        'Filename': test_filenames,
        'True Label': ['Segar' if label == 0 else 'Busuk' for label in y_test],
        'Predicted Label': ['Segar' if pred == 0 else 'Busuk' for pred in y_pred]
    })

    # Simpan ke Excel
    results_df.to_excel(results_file, index=False)
    print(f"Hasil pengujian diekspor ke file: {results_file}")
