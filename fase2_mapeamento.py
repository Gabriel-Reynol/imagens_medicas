import pydicom
import os
import numpy as np
import pandas as pd

# --- FUNÇÕES DE APOIO (O "Trabalho Pesado") ---

def carregar_mapa_de_labels(arquivo_csv, coluna_id, coluna_label):
    """Lê o CSV e cria um dicionário ID -> Label"""
    try:
        # encoding latin-1 ajuda a ler arquivos feitos no Excel 
        df = pd.read_csv(arquivo_csv, encoding='latin-1') 
        
        # Garante que IDs duplicados não atrapalhem
        df = df.drop_duplicates(subset=[coluna_id])
        
        # Cria o dicionário
        mapa = df.set_index(coluna_id)[coluna_label].to_dict()
        print(f"Planilha lida! {len(mapa)} pacientes no gabarito.")
        return mapa
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return None

def e_uma_serie_valida(lista_de_headers):
    """Filtra se é uma imagem de tomografia axial útil"""
    if not lista_de_headers: return False
    
    header = lista_de_headers[0]
    try:
        # Regra 1: Tem que ter bastante fatia (evita Scout/Raio-X)
        if len(lista_de_headers) < 10: return False 
        
        # Regra 2: Tem que ser AXIAL (corte transversal)
        if 'AXIAL' not in header.ImageType: return False
        
        # Regra 3: Não pode ser LOCALIZER
        if 'LOCALIZER' in header.ImageType: return False
        
        return True
    except Exception:
        return False

def processar_e_salvar_serie(path_da_serie, nomes_dos_arquivos_dcm, mapa_de_labels, pasta_0, pasta_1):
    """
    Lê os DICOMs, transforma em Volume 3D e salva como .npy
    """
    fatias_headers = []
    
    # 1. Leitura Rápida (Só Headers)
    for nome_arquivo in nomes_dos_arquivos_dcm:
        caminho_completo = os.path.join(path_da_serie, nome_arquivo)
        try:
            dcm = pydicom.dcmread(caminho_completo, stop_before_pixels=True)
            fatias_headers.append(dcm)
        except Exception: pass 

    if not fatias_headers: return

    # 2. O MATCH (Verifica se está na planilha)
    try:
        study_id = fatias_headers[0].StudyInstanceUID
    except AttributeError: return
    
    label = mapa_de_labels.get(study_id)
    
    # Se não achou na planilha ou o label não é 0/1, ignora
    if label not in [0, 1]: return

    # 3. FILTRO DE QUALIDADE
    if not e_uma_serie_valida(fatias_headers): return 

    # 4. LEITURA DOS PIXELS (Só agora gasta memória)
    # print(f"Processando ID: {study_id[:15]}... (Label {label})")
    fatias_full = []
    for h in fatias_headers:
        try:
            d = pydicom.dcmread(h.filename)
            if hasattr(d, 'pixel_array'): fatias_full.append(d)
        except: pass
    
    if not fatias_full: return
    
    try:
        # Ordena as fatias pela posição
        fatias_full.sort(key=lambda x: int(x.InstanceNumber))
        
        # Empilha em 3D
        vol = np.stack([f.pixel_array for f in fatias_full])
        
        # Define onde salvar
        nome_arq = f"{fatias_full[0].SeriesInstanceUID}.npy"
        path_destino = pasta_1 if label == 1 else pasta_0
        arquivo_final = os.path.join(path_destino, nome_arq)
        
        # Salva
        np.save(arquivo_final, vol)
        print(f"Salvo: {nome_arq} (Label {label})")
        
    except Exception as e:
        print(f"Erro ao salvar volume: {e}")


# --- A FUNÇÃO PRINCIPAL (Que a Main vai chamar) ---

def executar_fase2(caminho_imagens, caminho_csv):
    """
    Recebe os caminhos da MAIN e executa o processamento.
    """
    print(f"\n" + "="*40)
    print(f"FASE 2: PREPARAÇÃO DE DADOS")
    print(f"Imagens: {caminho_imagens}")
    print(f"CSV: {caminho_csv}")
    print("="*40 + "\n")
    
    # 1. Carregar Gabarito
    mapa = carregar_mapa_de_labels(caminho_csv, 'UID dicom', 'Contraste')
    if mapa is None: return

    # 2. Criar Pastas de Saída (Sempre locais, onde o script roda)
    path_ds = os.path.join(os.getcwd(), "dataset")
    p0 = os.path.join(path_ds, "0_sem_contraste")
    p1 = os.path.join(path_ds, "1_com_contraste")
    
    os.makedirs(p0, exist_ok=True)
    os.makedirs(p1, exist_ok=True)

    print(f"Iniciando varredura e conversão para .npy...")
    print(f"Os arquivos serão salvos em: {path_ds}\n")

    count = 0
    # 3. O Loop (Walk)
    for root, dirs, files in os.walk(caminho_imagens):
        dcms = [f for f in files if f.endswith('.dcm')]
        if dcms:
            processar_e_salvar_serie(root, dcms, mapa, p0, p1)
            count += 1
            # Um print a cada 50 séries para você saber que não travou
            if count % 50 == 0: print(f"... {count} pastas analisadas ...")

    print(f"\nFASE 2 CONCLUÍDA! Verifique a pasta 'dataset'.")


# --- MODO DE TESTE LOCAL ---
# Isso permite rodar esse arquivo sozinho no PC sem a main
if __name__ == "__main__":
    # Caminhos do PC para teste
    teste_img = 'C:/Users/reyno/USP2025/pacientes/'
    teste_csv = 'C:/Users/reyno/USP2025/IC_imagens/imagens_medicas/patIDStudy_contrast.csv'
    
    if os.path.exists(teste_img):
        executar_fase2(teste_img, teste_csv)
    else:
        print("Caminho de teste não encontrado no PC.")