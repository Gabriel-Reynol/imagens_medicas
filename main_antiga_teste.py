import pydicom
import os
import numpy as np
import pandas as pd # Biblioteca para ler o CSV

# --- PASSO 0: CONFIGURAÇÃO ---
NOME_DO_ARQUIVO_CSV = 'patIDStudy_contrast.csv' # Arquivo CSV com os labels
NOME_COLUNA_ID_ESTUDO = 'UID dicom' # Coluna no CSV com o ID do Estudo
NOME_COLUNA_LABEL = 'Contraste' # Coluna no CSV com o label (0 ou 1)


def carregar_mapa_de_labels(arquivo_csv, coluna_id, coluna_label):
    """
    Carrega a planilha CSV e a transforma em um dicionário (mapa)
    para busca rápida.
    """
    try:
        # 'latin-1' previne erros de encoding
        df = pd.read_csv(arquivo_csv, encoding='latin-1') 
        
        # Garante IDs únicos no mapa
        df = df.drop_duplicates(subset=[coluna_id])
        
        # Converte as colunas do CSV em um dicionário para performance
        # Ex: {'id_estudo_123': 0, 'id_estudo_456': 1}
        mapa_de_labels = df.set_index(coluna_id)[coluna_label].to_dict()
        print(f"Planilha '{arquivo_csv}' lida. {len(mapa_de_labels)} IDs únicos de estudo carregados.")
        return mapa_de_labels
    except FileNotFoundError:
        print(f"ERRO: Planilha não encontrada em: {arquivo_csv}")
        print("Verificar se o .csv está na mesma pasta do main.py")
        return None
    except KeyError:
        print(f"ERRO: Colunas '{coluna_id}' ou '{coluna_label}' não encontradas na planilha.")
        return None
    except Exception as e:
        print(f"Erro inesperado ao ler o CSV: {e}")
        return None

def e_uma_serie_valida(lista_de_headers):
    """
    Verifica se a série é uma imagem de TC válida e não um "lixo" (como localizador).
    Esta é a lógica de "Estudo -> Série -> Imagem" (O Filtro).
    """
    if not lista_de_headers:
        return False

    # Pega o cabeçalho da primeira fatia para checar
    header = lista_de_headers[0]

    try:
        # 1. Filtro por Número de Fatias: Séries de localizador/scout têm < 10 fatias.
        #    Este número (10) pode ser ajustado se necessário.
        if len(lista_de_headers) < 10:
            print("Filtro: Rejeitado (pocas fatias, provável localizador)")
            return False

        # 2. Filtro por Tipo de Imagem (ImageType):
        #    O tipo 'AXIAL' é o tipo de corte de interesse. 
        if 'AXIAL' not in header.ImageType:
            print("Filtro: Rejeitado (não é AXIAL)")
            return False
            
        # 3. Filtro de "Lixo" explícito
        if 'LOCALIZER' in header.ImageType:
            print("Filtro: Rejeitado (é LOCALIZER)")
            return False
            
        print("Filtro: Série APROVADA.")
        return True

    except Exception as e:
        print(f"Erro no filtro de série: {e}")
        return False

def processar_e_salvar_serie(path_da_serie, nomes_dos_arquivos_dcm, mapa_de_labels, pasta_0, pasta_1):
    """
    Função principal de processamento.
    1. Lê os cabeçalhos das fatias (otimizado).
    2. Encontra o 'StudyInstanceUID' (ID do Exame/Estudo).
    3. Procura o 'label' (0 ou 1) no mapa_de_labels (carregado do CSV).
    4. Se houver "match", aplica o FILTRO DE SÉRIE.
    5. Se o filtro aprovar, lê os pixels, empilha e salva o volume 3D (.npy) na pasta correta.
    """
    print(f"--- Processando Série: {path_da_serie} ---")
    
    fatias_headers = [] # Lista para guardar os cabeçalhos
    for nome_arquivo in nomes_dos_arquivos_dcm:
        caminho_completo = os.path.join(path_da_serie, nome_arquivo)
        try:
            # Otimização: stop_before_pixels=True faz a leitura ser MUITO mais rápida
            dcm = pydicom.dcmread(caminho_completo, stop_before_pixels=True) # Lê só cabeçalho
            fatias_headers.append(dcm)
        except Exception:
            pass 

    if not fatias_headers:
        print("Nenhuma fatia DICOM válida encontrada.")
        return # Pula

    # --- PASSO 2: O "MATCH" (A "Ponte") ---
    # O "ID de Estudo" no DICOM chama-se 'StudyInstanceUID'
    try:
        # Pega o ID do cabeçalho da primeira fatia
        study_id_dcm = fatias_headers[0].StudyInstanceUID
    except AttributeError:
        print("Aviso: Não foi possível ler o StudyInstanceUID. Pulando série.")
        return

    # Procura o ID do Estudo no mapa carregado do CSV
    label = mapa_de_labels.get(study_id_dcm)

    # Se o ID não foi encontrado no CSV, a série é pulada
    if label is None:
        print(f"Aviso: ID {study_id_dcm[:10]}... não encontrado no CSV. Pulando série.")
        print("\n" + "="*50 + "\n")
        return
    
    # Se o label não for 0 ou 1, a série é pulada
    if label not in [0, 1]:
        print(f"Aviso: Label '{label}' para o ID {study_id_dcm[:10]}... não é 0 ou 1. Pulando.")
        print("\n" + "="*50 + "\n")
        return

    print(f"MATCH! ID {study_id_dcm[:10]}... encontrado com Label = {label}")

    # --- NOVO PASSO: FILTRO DE SÉRIE ---
    if not e_uma_serie_valida(fatias_headers):
        print("Aviso: A série foi encontrada no CSV, mas não é uma série de imagem válida (ex: localizador). Pulando.")
        print("\n" + "="*50 + "\n")
        return
    # --- FIM DO NOVO PASSO ---


    # --- Agora que houve "MATCH" E "FILTRO", os pixels são processados ---
    fatias_com_pletas = []
    for dcm_header in fatias_headers:
        try:
            dcm_full = pydicom.dcmread(dcm_header.filename) # Agora lê os pixels
            if hasattr(dcm_full, 'pixel_array'):
                fatias_com_pletas.append(dcm_full)
        except Exception:
            pass
    
    if not fatias_com_pletas:
        print("Erro ao ler pixels das fatias encontradas. Pulando.")
        return

    try:
        fatias_com_pletas.sort(key=lambda x: int(x.InstanceNumber))
    except Exception as e:
        print(f"Aviso: Não foi possível ordenar fatias: {e}")

    try:
        volume_3d_bruto = np.stack([f.pixel_array for f in fatias_com_pletas])
        print(f"Volume 3D criado! Shape: {volume_3d_bruto.shape}")
    except Exception as e:
        print(f"Erro: As fatias têm tamanhos diferentes e não puderam ser empilhadas. {e}")
        return

    # --- PASSO 3: SALVAR O DATASET ---
    # O nome do arquivo .npy será o ID da Série (SeriesInstanceUID), que é único para cada "pilha" de fatias
    nome_do_arquivo = f"{fatias_com_pletas[0].SeriesInstanceUID}.npy" 
    
    if label == 0:
        caminho_para_salvar = os.path.join(pasta_0, nome_do_arquivo)
    else: # label == 1
        caminho_para_salvar = os.path.join(pasta_1, nome_do_arquivo)

    try:
        # Salva o array NumPy 3D completo
        np.save(caminho_para_salvar, volume_3d_bruto)
        print(f"SUCESSO! Volume salvo em: {caminho_para_salvar}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo .npy: {e}")

    print("\n" + "="*50 + "\n")


# --- INÍCIO DO SCRIPT PRINCIPAL ---

# 1. Carregar o "mapa" da planilha (CSV) UMA VEZ
mapa_de_labels = carregar_mapa_de_labels(NOME_DO_ARQUIVO_CSV, NOME_COLUNA_ID_ESTUDO, NOME_COLUNA_LABEL)

# Se o mapa não for carregado (ex: planilha não encontrada), o script é encerrado.
if mapa_de_labels is None:
    print("Encerrando o script devido a erro na leitura da planilha.")
    exit() # Encerra o programa

# 2. Definir pastas de Saída para o DATASET
path_raiz_pacientes = 'C:/Users/reyno/USP2025/pacientes/'
path_dataset = os.path.join(os.getcwd(), "dataset")
path_sem_contraste = os.path.join(path_dataset, "0_sem_contraste")
path_com_contraste = os.path.join(path_dataset, "1_com_contraste")

# Cria as pastas de saída (0 e 1) se elas não existirem
os.makedirs(path_sem_contraste, exist_ok=True)
os.makedirs(path_com_contraste, exist_ok=True)
print(f"Dataset será salvo em: {path_dataset}\n")

# 3. O "Explorador" (os.walk)
print("Iniciando varredura de arquivos DICOM...")
for root, dirs, files in os.walk(path_raiz_pacientes):
    arquivos_dcm = [f for f in files if f.endswith('.dcm')]
    
    if arquivos_dcm:
        # 4. Chama a função de processamento para esta série
        processar_e_salvar_serie(root, arquivos_dcm, mapa_de_labels, path_sem_contraste, path_com_contraste)

print(f"--- Processamento concluído. ---")
print(f"Dataset criado em: \n{path_sem_contraste}\n{path_com_contraste}")