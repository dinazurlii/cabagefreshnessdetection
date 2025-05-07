import os
import cv2
import numpy as np
import pandas as pd

# Path ke folder Segar dan Busuk
fresh_folder = 'data/Segar'
rotten_folder = 'data/Busuk'
output_excel = 'hsv_values.xlsx'

def extract_hsv_values(folder, label):
    """Ekstrak nilai HSV rata-rata untuk semua gambar dalam folder"""
    data = []
    images = []  # List untuk menyimpan gambar-gambar untuk ditampilkan
    
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        image = cv2.imread(filepath)
        if image is not None:
            # Konversi gambar ke citra biner (Thresholding)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Ubah citra biner menjadi 3 kanal (RGB)
            binary_image_3channel = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

            # Konversi gambar ke HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_display = hsv.copy()
            
            # Meningkatkan kontras dengan menyesuaikan S dan V jika perlu
            hsv_display[..., 1] = np.clip(hsv_display[..., 1] * 1.3, 0, 255)  # Meningkatkan Saturasi
            hsv_display[..., 2] = np.clip(hsv_display[..., 2] * 1.1, 0, 255)  # Meningkatkan Value

            # Menyimpan gambar asli, biner, dan HSV dalam satu list
            images.append({
                'Nama Gambar': file,
                'Label': label,
                'Original': image,
                'Binary': binary_image_3channel,  # Gambar biner sekarang 3 kanal
                'HSV': hsv_display
            })

            # Hitung nilai HSV rata-rata
            hsv_mean = np.mean(hsv, axis=(0, 1))
            data.append({
                'Nama Gambar': file,
                'Label': label,
                'H': hsv_mean[0],
                'S': hsv_mean[1],
                'V': hsv_mean[2]
            })
    return data, images

def show_images(images, idx):
    """Tampilkan gambar bersebelahan dalam satu jendela dengan keterangan"""
    original_image = images[idx]['Original']
    binary_image = images[idx]['Binary']
    hsv_image = images[idx]['HSV']
    filename = images[idx]['Nama Gambar']
    label = images[idx]['Label']

    # Menambahkan teks pada gambar dengan ukuran font yang lebih kecil
    font_scale = 2.0  # Ukuran font lebih kecil
    thickness = 2

    # Menambahkan keterangan gambar
    original_image = cv2.putText(original_image, f"Original ({filename})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    binary_image = cv2.putText(binary_image, f"Binary ({filename})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    hsv_image = cv2.putText(hsv_image, f"HSV ({filename})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Menambahkan keterangan label pada gambar
    label_text = "Segar" if label == 'Segar' else "Busuk"
    original_image = cv2.putText(original_image, f"Label: {label_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    binary_image = cv2.putText(binary_image, f"Label: {label_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    hsv_image = cv2.putText(hsv_image, f"Label: {label_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Menambahkan posisi teks yang lebih rapi dengan jarak yang cukup
    original_image = cv2.putText(original_image, f"Image: {filename}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    binary_image = cv2.putText(binary_image, f"Image: {filename}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    hsv_image = cv2.putText(hsv_image, f"Image: {filename}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Gabungkan gambar-gambar menjadi satu gambar bersebelahan
    combined_image = np.hstack((original_image, binary_image, hsv_image))

    # Tampilkan gambar
    cv2.imshow('Gambar Segar dan Busuk (Asli, Biner, HSV)', combined_image)

if __name__ == '__main__':
    # Ekstrak data dari folder Segar dan Busuk
    fresh_data, fresh_images = extract_hsv_values(fresh_folder, 'Segar')
    rotten_data, rotten_images = extract_hsv_values(rotten_folder, 'Busuk')

    # Gabungkan data
    all_data = fresh_data + rotten_data
    all_images = fresh_images + rotten_images

    # Simpan ke Excel
    df = pd.DataFrame(all_data)
    df.to_excel(output_excel, index=False)
    print(f"Data HSV telah diekspor ke file: {output_excel}")

    # Navigasi untuk menampilkan gambar-gambar
    idx = 0
    while True:
        show_images(all_images, idx)

        # Menunggu input tombol untuk navigasi
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # Tombol 'n' untuk next gambar
            idx = (idx + 1) % len(all_images)  # Pindah ke gambar berikutnya, looping ke awal
        elif key == ord('p'):  # Tombol 'p' untuk previous gambar
            idx = (idx - 1) % len(all_images)  # Pindah ke gambar sebelumnya, looping ke akhir
        elif key == ord(' '):  # Tombol spasi untuk next gambar
            idx = (idx + 1) % len(all_images)  # Pindah ke gambar berikutnya
        elif key == ord('q'):  # Tombol 'q' untuk keluar
            break

    # Tutup semua jendela
    cv2.destroyAllWindows()
