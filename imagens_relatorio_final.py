import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

# ============================
PASTA_EXAME = "/Storage/jerogalsky-2024/0006950G/1.2.840.113704.1.111.1348.1440592299.1/1.2.840.113704.1.111.1348.1440592406.11/"
IMG_SIZE = (224, 224)
# ============================


def aplicar_janela(img_hu, center, width):
    """
    Aplica janelamento em imagem já convertida para HU.
    Depois normaliza os valores da janela para 0-1.
    """
    low = center - width / 2
    high = center + width / 2

    img = np.clip(img_hu, low, high)
    img = (img - low) / (high - low + 1e-6)

    return img.astype(np.float32)


def carregar_fatia_central(pasta_ou_arquivo):
    """
    Lê uma pasta de DICOMs ou um arquivo .dcm.
    Se for pasta, seleciona a fatia central.
    Retorna:
    - imagem original do pixel_array
    - imagem convertida para HU
    - caminho do arquivo lido
    """
    arquivo_para_ler = pasta_ou_arquivo

    if os.path.isdir(pasta_ou_arquivo):
        files = sorted([f for f in os.listdir(pasta_ou_arquivo) if f.endswith(".dcm")])

        if not files:
            raise RuntimeError(f"Pasta sem arquivos .dcm: {pasta_ou_arquivo}")

        meio = len(files) // 2
        arquivo_para_ler = os.path.join(pasta_ou_arquivo, files[meio])

    else:
        if not pasta_ou_arquivo.endswith(".dcm"):
            raise RuntimeError("Passe uma pasta com DICOMs ou um arquivo .dcm")

    ds = pydicom.dcmread(arquivo_para_ler)

    # Imagem original do DICOM
    img_original = ds.pixel_array.astype(np.float32)

    # Conversão para HU
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img_hu = img_original * slope + intercept

    return img_original, img_hu, arquivo_para_ler


def gerar_mediastinal_3x(img_hu):
    """
    Gera entrada do modelo com janela mediastinal repetida em 3 canais.
    """
    img_mediastino = aplicar_janela(img_hu, center=40, width=400)
    img_mediastino = cv2.resize(img_mediastino, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    img_3c = np.stack(
        [img_mediastino, img_mediastino, img_mediastino],
        axis=-1
    ).astype(np.float32)

    return img_3c


def gerar_tres_janelamentos(img_hu):
    """
    Gera entrada do modelo com 3 janelamentos distintos:
    - canal 1: pulmonar
    - canal 2: mediastinal
    - canal 3: extra
    """
    img_pulmao = aplicar_janela(img_hu, center=-600, width=1500)
    img_mediastino = aplicar_janela(img_hu, center=40, width=400)
    img_extra = aplicar_janela(img_hu, center=100, width=700)

    img_pulmao = cv2.resize(img_pulmao, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    img_mediastino = cv2.resize(img_mediastino, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    img_extra = cv2.resize(img_extra, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    img_3j = np.stack(
        [img_pulmao, img_mediastino, img_extra],
        axis=-1
    ).astype(np.float32)

    return img_3j, img_pulmao, img_mediastino, img_extra


def salvar_figuras(img_original, img_mediastinal_3x, img_3j, img_pulmao, img_mediastino, img_extra):
    """
    Salva as figuras para inserir no relatório.
    """

    # Figura 1: imagem original do DICOM
    plt.figure(figsize=(6, 6))
    plt.imshow(img_original, cmap="gray")
    plt.title("Imagem original extraída do DICOM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("imagem_original_dicom.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 2: janela mediastinal repetida 3x
    # Como os 3 canais são iguais, mostramos um canal em escala de cinza.
    plt.figure(figsize=(6, 6))
    plt.imshow(img_mediastinal_3x[:, :, 0], cmap="gray")
    plt.title("Janela mediastinal repetida em 3 canais")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("janela_mediastinal_3x.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 3: composição RGB com 3 janelamentos distintos
    plt.figure(figsize=(6, 6))
    plt.imshow(img_3j)
    plt.title("Composição com 3 janelamentos distintos")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("tres_janelamentos_rgb.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 4: canais separados dos 3 janelamentos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_pulmao, cmap="gray")
    axes[0].set_title("Canal 1: janela pulmonar")
    axes[0].axis("off")

    axes[1].imshow(img_mediastino, cmap="gray")
    axes[1].set_title("Canal 2: janela mediastinal")
    axes[1].axis("off")

    axes[2].imshow(img_extra, cmap="gray")
    axes[2].set_title("Canal 3: janela extra")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("tres_janelamentos_canais_separados.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    img_original, img_hu, arquivo_lido = carregar_fatia_central(PASTA_EXAME)

    img_mediastinal_3x = gerar_mediastinal_3x(img_hu)
    img_3j, img_pulmao, img_mediastino, img_extra = gerar_tres_janelamentos(img_hu)

    print("=== CONFIRMAÇÃO DAS IMAGENS GERADAS ===")
    print(f"Arquivo DICOM lido: {arquivo_lido}")
    print(f"Shape original: {img_original.shape}")
    print(f"Shape HU: {img_hu.shape}")
    print(f"Shape mediastinal 3x: {img_mediastinal_3x.shape}")
    print(f"Shape 3 janelamentos: {img_3j.shape}")
    print(f"Range mediastinal 3x: {img_mediastinal_3x.min():.6f} / {img_mediastinal_3x.max():.6f}")
    print(f"Range 3 janelamentos: {img_3j.min():.6f} / {img_3j.max():.6f}")

    dif01 = np.max(np.abs(img_mediastinal_3x[:, :, 0] - img_mediastinal_3x[:, :, 1]))
    dif12 = np.max(np.abs(img_mediastinal_3x[:, :, 1] - img_mediastinal_3x[:, :, 2]))

    print("\n=== CHECAGEM DA MEDIASTINAL 3X ===")
    print(f"Máx diferença canal0-canal1: {dif01:.8f}")
    print(f"Máx diferença canal1-canal2: {dif12:.8f}")

    salvar_figuras(
        img_original,
        img_mediastinal_3x,
        img_3j,
        img_pulmao,
        img_mediastino,
        img_extra
    )

    print("\nImagens salvas:")
    print("- imagem_original_dicom.png")
    print("- janela_mediastinal_3x.png")
    print("- tres_janelamentos_rgb.png")
    print("- tres_janelamentos_canais_separados.png")


if __name__ == "__main__":
    main()