import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

# ====== CONFIGURE AQUI ======
PASTA_EXAME = "/Storage/jerogalsky-2024/0006950G/1.2.840.113704.1.111.1348.1440592299.1/1.2.840.113704.1.111.1348.1440592406.11/"
IMG_SIZE = (224, 224)
# ============================

def aplicar_janela(img_hu, center, width):
    low = center - width / 2
    high = center + width / 2
    img = np.clip(img_hu, low, high)
    img = (img - low) / (high - low + 1e-6)
    return img.astype(np.float32)

def carregar_fatia_central_e_preprocessar_janelas(pasta_ou_arquivo: str, dim=(224,224)):
    """
    Replica a nova lógica:
    - pega fatia central se for pasta
    - lê pixel_array
    - converte para HU
    - aplica 3 janelamentos
    - resize para 224x224
    - stack 3 canais
    """
    arquivo_para_ler = pasta_ou_arquivo

    if os.path.isdir(pasta_ou_arquivo):
        files = sorted([f for f in os.listdir(pasta_ou_arquivo) if f.endswith(".dcm")])
        if not files:
            raise RuntimeError(f"Pasta sem .dcm: {pasta_ou_arquivo}")
        meio = len(files) // 2
        arquivo_para_ler = os.path.join(pasta_ou_arquivo, files[meio])
    else:
        if not pasta_ou_arquivo.endswith(".dcm"):
            raise RuntimeError("Passe uma pasta com DICOMs ou um arquivo .dcm")

    ds = pydicom.dcmread(arquivo_para_ler)

    img_original = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    img_hu = img_original * slope + intercept

    img_pulmao = aplicar_janela(img_hu, center=-600, width=1500)
    img_mediastino = aplicar_janela(img_hu, center=40, width=400)
    img_extra = aplicar_janela(img_hu, center=100, width=700)

    img_pulmao_r = cv2.resize(img_pulmao, dim, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img_mediastino_r = cv2.resize(img_mediastino, dim, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img_extra_r = cv2.resize(img_extra, dim, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    img_3c = np.stack([img_pulmao_r, img_mediastino_r, img_extra_r], axis=-1).astype(np.float32)

    return img_original, img_hu, img_pulmao_r, img_mediastino_r, img_extra_r, img_3c, arquivo_para_ler

def main():
    img_original, img_hu, img_p, img_m, img_e, img_3c, arquivo_lido = carregar_fatia_central_e_preprocessar_janelas(
        PASTA_EXAME, dim=IMG_SIZE
    )

    print("=== CONFIRMAÇÃO DA NOVA ENTRADA DO MODELO ===")
    print(f"Arquivo DICOM lido: {arquivo_lido}")
    print(f"Shape original: {img_original.shape}")
    print(f"Shape final: {img_3c.shape}  (esperado: (224,224,3))")
    print(f"Dtype final: {img_3c.dtype}")
    print(f"Range canal pulmonar: {img_p.min():.6f} / {img_p.max():.6f}")
    print(f"Range canal mediastinal: {img_m.min():.6f} / {img_m.max():.6f}")
    print(f"Range canal extra: {img_e.min():.6f} / {img_e.max():.6f}")

    plt.figure(figsize=(6, 6))
    plt.imshow(img_original, cmap="gray")
    plt.title("Imagem original (pixel_array)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_p, cmap="gray")
    plt.title("Canal 1 - Pulmonar")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_m, cmap="gray")
    plt.title("Canal 2 - Mediastinal")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_e, cmap="gray")
    plt.title("Canal 3 - Extra")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(img_3c)
    plt.title("Entrada do modelo (3 janelamentos)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.imsave("janela_pulmonar.png", img_p, cmap="gray")
    plt.imsave("janela_mediastinal.png", img_m, cmap="gray")
    plt.imsave("janela_extra.png", img_e, cmap="gray")
    print("Imagens salvas: janela_pulmonar.png, janela_mediastinal.png, janela_extra.png")

if __name__ == "__main__":
    main()