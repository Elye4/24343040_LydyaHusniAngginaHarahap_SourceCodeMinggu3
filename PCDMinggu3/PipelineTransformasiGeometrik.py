# =====================================================
# PIPELINE TRANSFORMASI GEOMETRIK UNTUK REGISTRASI CITRA
# =====================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log10, sqrt

# ==============================
# LOAD IMAGE
# ==============================
ref = cv2.imread("buku_lurus.jpg")
img = cv2.imread("buku_miring.jpg")

if ref is None or img is None:
    print("Gambar tidak ditemukan!")
    exit()

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = ref.shape[:2]
img = cv2.resize(img, (w, h))

# ==============================
# METRIK EVALUASI
# ==============================
def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float"))**2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return 100
    return 20 * log10(255.0 / sqrt(m))

# =====================================================
# 1. TRANSLASI
# =====================================================
T = np.float32([[1,0,80],[0,1,40]])
translated = cv2.warpAffine(img, T, (w,h))

# =====================================================
# 2. ROTASI
# =====================================================
center = (w//2, h//2)
R = cv2.getRotationMatrix2D(center, 15, 1.0)
rotated = cv2.warpAffine(img, R, (w,h))

# =====================================================
# 3. SCALING
# =====================================================
scaled = cv2.resize(img, None, fx=1.2, fy=1.2,
                    interpolation=cv2.INTER_LINEAR)
scaled = cv2.resize(scaled, (w,h))

# =====================================================
# 4. AFFINE (3 TITIK)
# =====================================================
pts1 = np.float32([[50,50],[300,50],[50,300]])
pts2 = np.float32([[10,120],[300,50],[120,350]])

M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M_affine, (w,h))

# =====================================================
# 5. PERSPEKTIF (4 TITIK)
# =====================================================
src_pts = np.float32([
    [60,60],
    [420,40],
    [40,350],
    [430,360]
])

dst_pts = np.float32([
    [0,0],
    [w,0],
    [0,h],
    [w,h]
])

M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

# =====================================================
# 6. INTERPOLASI PADA PERSPEKTIF
# =====================================================
methods = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

results = {}

for name, method in methods.items():

    start = time.time()

    warped = cv2.warpPerspective(
        img,
        M_persp,
        (w,h),
        None,
        method
    )

    elapsed = time.time() - start

    results[name] = {
        "image": warped,
        "MSE": mse(ref, warped),
        "PSNR": psnr(ref, warped),
        "Time": elapsed
    }

# =====================================================
# CETAK HASIL EVALUASI
# =====================================================
print("\n===== HASIL INTERPOLASI =====")

for k,v in results.items():
    print(f"\nMetode : {k}")
    print("MSE   :", v["MSE"])
    print("PSNR  :", v["PSNR"])
    print("Waktu :", v["Time"], "detik")

# =====================================================
# VISUALISASI
# =====================================================
plt.figure(figsize=(14,8))

images = [
    ref,
    img,
    translated,
    rotated,
    affine,
    results["Bicubic"]["image"]
]

titles = [
    "Referensi",
    "Original",
    "Translasi",
    "Rotasi",
    "Affine",
    "Perspective (Bicubic)"
]

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================================
# VISUALISASI INTERPOLASI
# =====================================================
plt.figure(figsize=(12,4))

i = 1
for name,data in results.items():
    plt.subplot(1,3,i)
    plt.imshow(data["image"])
    plt.title(name)
    plt.axis("off")
    i += 1

plt.tight_layout()
plt.show()

# ======================================================
#  INTERPOLASI EVALUASI KUALITAS
# ======================================================

import time
from math import log10, sqrt

# ---------- Fungsi Evaluasi ----------
def hitung_mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def hitung_psnr(img1, img2):
    mse = hitung_mse(img1, img2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

print("\n===== EVALUASI INTERPOLASI =====")

# gunakan matriks perspektif yang SUDAH ADA sebelumnya
methods = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

hasil_interpolasi = {}

for nama, metode in methods.items():

    start_time = time.time()

    hasil = cv2.warpPerspective(
        img,
        M_persp,     # pakai matriks perspektif sebelumnya
        (w, h),
        flags=metode
    )

    waktu = time.time() - start_time

    mse = hitung_mse(img, hasil)
    psnr = hitung_psnr(img, hasil)

    hasil_interpolasi[nama] = hasil

    print(f"\nMetode : {nama}")
    print("MSE    :", mse)
    print("PSNR   :", psnr, "dB")
    print("Waktu  :", waktu, "detik")

    cv2.imshow(f"Perspektif {nama}", hasil)