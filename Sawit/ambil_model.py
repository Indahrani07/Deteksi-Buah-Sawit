import os
import torch

# Definisikan jalur ke file model
model_path = 'best.pt'

# Periksa apakah file model ada
if os.path.isfile(model_path):
    print(f"File model ditemukan: {model_path}")
    # Muat model menggunakan torch.hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.eval()
    print("Model berhasil dimuat:")
    print(model)
else:
    # Jika file tidak ditemukan, beri tahu pengguna
    print(f"File model tidak ditemukan: {model_path}")
