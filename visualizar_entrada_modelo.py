import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

# ====== CONFIGURE AQUI ======
PASTA_EXAME = "/Storage/jerogalsky-2024/0006950G/1.2.840.113704.1.111.1348.1440592299.1/1.2.840.113704.1.111.1348.1440592406.11/"  
IMG_SIZE = (224, 224)
# ============================

def carregar_fatia_central_e_preprocessar(pasta_ou_arquivo: str, dim=(224,224)):
    """
    Replica fielmente o que o MedicalDataGenerator faz:
    - pega fatia central se for pasta
    - lê pixel_array
    - normaliza min-max
    - resize para 224x224
    - stack 3 canais (RGB fake)

    Retorna:
    - img_original: imagem original lida do DICOM
    - img_3c: imagem final (224,224,3)
    - arquivo_lido
    - mn, mx
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

    # imagem original
    img_original = ds.pixel_array.astype(np.float32)

    # min-max
    mn = float(np.min(img_original))
    mx = float(np.max(img_original))
    img_norm = (img_original - mn) / (mx - mn + 1e-6)

    # resize 224x224
    img_resized = cv2.resize(img_norm, dim, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # 3 canais (RGB fake)
    img_3c = np.stack((img_resized,)*3, axis=-1).astype(np.float32)

    return img_original, img_3c, arquivo_para_ler, mn, mx

def main():
    img_original, img_3c, arquivo_lido, mn, mx = carregar_fatia_central_e_preprocessar(PASTA_EXAME, dim=IMG_SIZE)

    print("=== CONFIRMAÇÃO DO QUE O MODELO RECEBE ===")
    print(f"Arquivo DICOM lido: {arquivo_lido}")
    print(f"Shape original: {img_original.shape}")
    print(f"Shape final: {img_3c.shape}  (esperado: (224,224,3))")
    print(f"Dtype final: {img_3c.dtype}        (esperado: float32)")
    print(f"Range final (min/max): {img_3c.min():.6f} / {img_3c.max():.6f}  (esperado ~ 0..1)")
    print(f"Min/Max originais (antes do min-max): {mn:.3f} / {mx:.3f}")

    # 1) Mostrar imagem original
    plt.figure(figsize=(6, 6))
    plt.imshow(img_original, cmap="gray")
    plt.title("Imagem original (DICOM)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 2) Mostrar imagem final em RGB fake
    plt.figure(figsize=(6, 6))
    plt.imshow(img_3c)
    plt.title("Entrada do modelo (RGB replicado)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 3) Mostrar um canal da imagem final
    plt.figure(figsize=(6, 6))
    plt.imshow(img_3c[:, :, 0], cmap="gray")
    plt.title("Entrada do modelo (canal 0 em cinza)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 4) Checar se os canais são mesmo iguais
    dif01 = np.max(np.abs(img_3c[:, :, 0] - img_3c[:, :, 1]))
    dif12 = np.max(np.abs(img_3c[:, :, 1] - img_3c[:, :, 2]))
    print(f"Máx diferença canal0-canal1: {dif01:.8f}")
    print(f"Máx diferença canal1-canal2: {dif12:.8f}")

    # 5) Salvar imagens separadas
    plt.imsave("imagem_original_dicom.png", img_original, cmap="gray")
    plt.imsave("exemplo_entrada_modelo.png", img_3c[:, :, 0], cmap="gray")
    print("Imagens salvas para relatório: imagem_original_dicom.png e exemplo_entrada_modelo.png")