import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Defina o caminho para a pasta da série DICOM de UM paciente

path_da_serie = 'C:/Users/reyno/USP2025/pacientes/0002265C/1.2.277.1.2.18545002.13862.20160808143815056.1/1.3.12.2.1107.5.1.4.63802.30000016080509244947000016114'

# 2. Ler todos os arquivos DICOM da pasta
print(f"Lendo arquivos da pasta: {path_da_serie}")
fatias = []
for nome_arquivo in os.listdir(path_da_serie):
    caminho_completo = os.path.join(path_da_serie, nome_arquivo)
    try:
        dcm = pydicom.dcmread(caminho_completo)
        if hasattr(dcm, 'pixel_array'):
            fatias.append(dcm)
    except Exception:
        pass # Ignora arquivos que não são DICOM válidos

print(f"Total de {len(fatias)} fatias DICOM lidas.")

# 3. Ordenar as fatias (pelo número da instância/imagem)
fatias.sort(key=lambda x: int(x.InstanceNumber))

# 4. Empilhar as fatias em um volume 3D (array NumPy)
# (Isso só funciona se todas as fatias tiverem o mesmo tamanho)
try:
    volume_3d_bruto = np.stack([fatia.pixel_array for fatia in fatias])
    print(f"Volume 3D empilhado! Shape: {volume_3d_bruto.shape}")
except:
    print("Erro: As fatias têm tamanhos diferentes e não puderam ser empilhadas.")


# 5. "ABRIR" (VISUALIZAR) UMA FATIA
# Vamos pegar a fatia do meio do volume 3D para exibir

if 'volume_3d_bruto' in locals():
    fatia_central_idx = volume_3d_bruto.shape[0] // 2
    fatia_central = volume_3d_bruto[fatia_central_idx, :, :]
    
    print(f"Exibindo a fatia central (Índice: {fatia_central_idx}).")
    
    plt.imshow(fatia_central, cmap='gray')
    plt.title(f"Fatia {fatia_central_idx}")
    plt.axis('off') # Desliga os eixos X e Y
    plt.show()


