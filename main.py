import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt

def processar_uma_serie(path_da_serie, nomes_dos_arquivos_dcm, pasta_resultados):
    """
    Esta função processa UMA série:
    1. Lê os arquivos DICOM.
    2. Ordena e empilha em um volume 3D.
    3. SALVA a fatia central como uma imagem PNG, sem mostrar na tela.
    """
    print(f"--- Processando Série: {path_da_serie} ---")
    
    # 2. Lendo todos os arquivos DICOM da pasta
    fatias = []
    for nome_arquivo in nomes_dos_arquivos_dcm:
        caminho_completo = os.path.join(path_da_serie, nome_arquivo)
        try:
            dcm = pydicom.dcmread(caminho_completo)
            
            if hasattr(dcm, 'pixel_array'):
                fatias.append(dcm)
        except Exception:
            pass # Ignora arquivos que não são DICOM

    if not fatias:
        print("Nenhuma fatia DICOM válida encontrada nesta pasta.\n")
        return # Pula para a próxima série

    print(f"Total de {len(fatias)} fatias DICOM lidas.")

    # 3. Ordena as fatias
    try:
        fatias.sort(key=lambda x: int(x.InstanceNumber))
    except Exception as e:
        print(f"Aviso: Não foi possível ordenar fatias (pode faltar 'InstanceNumber'): {e}")
        # O código continua, mas pode estar fora de ordem

    # 4. Empilha as fatias
    volume_3d_bruto = None  
    try:
        volume_3d_bruto = np.stack([fatia.pixel_array for fatia in fatias])
        print(f"Volume 3D empilhado! Shape: {volume_3d_bruto.shape}")
    except Exception as e:
        print(f"Erro: As fatias têm tamanhos diferentes e não puderam ser empilhadas. {e}")
        return # Pula esta série (muito comum em séries mistas)


    # 5. "SALVAR" UMA FATIA 
    if volume_3d_bruto is not None:
        fatia_central_idx = volume_3d_bruto.shape[0] // 2
        fatia_central = volume_3d_bruto[fatia_central_idx, :, :]
        
        print(f"Fatia central (Índice: {fatia_central_idx}) encontrada.")
        
        # --- Lógica de Salvamento ---
        plt.figure() # Cria a figura "na memória"
        plt.imshow(fatia_central, cmap='gray')
        
            
        nome_da_serie = os.path.basename(path_da_serie)
        # Limpa o nome da série para criar um nome de arquivo seguro
        nome_da_serie_limpo = "".join(c for c in nome_da_serie if c.isalnum() or c in ('-', '_'))[:50]
        
        # Cria um nome de arquivo único
        nome_do_arquivo = f"{nome_da_serie_limpo}_fatia_{fatia_central_idx}.png"
        caminho_para_salvar = os.path.join(pasta_resultados, nome_do_arquivo)
        
        try:
            # 1. Salva a figura no disco
            plt.savefig(caminho_para_salvar, bbox_inches='tight')
            print(f"Imagem salva em: {caminho_para_salvar}")
            
            # 2. Fecha a figura (MUITO IMPORTANTE para não vazar memória)
            plt.close()
        except Exception as e:
            print(f"Erro ao salvar imagem: {e}")
            plt.close() # Garante que a figura feche mesmo se der erro
        # --- Fim da lógica de Salvamento ---
    
    print("\n" + "="*50 + "\n")


# --- INÍCIO DO SCRIPT PRINCIPAL ---

# 1. Definindo o caminho RAIZ para TODOS os pacientes
path_raiz_pacientes = 'C:/Users/reyno/USP2025/pacientes/'

# 1b. Definir e criar a pasta de resultados
# 'os.getcwd()' pega a pasta atual do seu projeto (ex: 'imagens_medicas')
path_dos_resultados = os.path.join(os.getcwd(), "resultados_fatias")
os.makedirs(path_dos_resultados, exist_ok=True) # Cria a pasta se ela não existir
print(f"Resultados serão salvos em: {path_dos_resultados}\n")


print("Iniciando varredura de todas as séries de pacientes...")
print(f"Pasta raiz: {path_raiz_pacientes}\n")

# 2. O "Explorador" (os.walk)
for root, dirs, files in os.walk(path_raiz_pacientes):
    
    # Filtra a lista de arquivos para pegar APENAS os .dcm
    arquivos_dcm = [f for f in files if f.endswith('.dcm')]
    
    # Se encontramos arquivos .dcm nesta pasta, ela é uma "série"
    if arquivos_dcm:
        # 3. Chama a nossa função para processar esta pasta (série)
        processar_uma_serie(root, arquivos_dcm, path_dos_resultados)

print(f"--- Varredura concluída. Verifique os arquivos em '{path_dos_resultados}'. ---")