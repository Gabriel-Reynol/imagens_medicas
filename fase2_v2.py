import os
import pandas as pd
import pydicom
import cv2
import tensorflow as tf
import numpy as np
import math

def aplicar_janela(img_hu, center, width):
    low = center - width / 2
    high = center + width / 2
    img = np.clip(img_hu, low, high)
    img = (img - low) / (high - low + 1e-6)
    return img.astype(np.float32)

def unificar_planilhas(csv_com, csv_sem):
    """
    Lê os dois arquivos CSV e cria um dicionário de IDs de ESTUDOS (Exames).
    Retorna: { 'UID_Estudo_123': 1, 'UID_Estudo_456': 0, ... }
    """
    print(" Lendo planilhas de exames (Study UIDs)...")
    
    # Procura pela coluna que identifica o EXAME (Study)
    possiveis_colunas = ['Study Instance UID', 'UID dicom', 'StudyInstanceUID', 'StudyID', '_id']
    
    def carregar(caminho, label):
        try:
            df = pd.read_csv(caminho, encoding='latin-1', sep=None, engine='python')
        except FileNotFoundError:
            print(f"ERRO: Arquivo não encontrado: {caminho}")
            return {}

        col = next((c for c in possiveis_colunas if c in df.columns), None)
        if not col: 
            print(f"AVISO: Coluna de UID de Estudo não encontrada em {caminho}")
            return {}
        
        # Limpa sujeira e cria dicionário
        df[col] = df[col].astype(str).str.strip().str.replace('"', '').str.replace("'", "")
        return dict.fromkeys(df[col], label)

    dict_com = carregar(csv_com, 1) 
    dict_sem = carregar(csv_sem, 0)
    
    mapa_total = {**dict_sem, **dict_com} 
    print(f"   -> Total de UIDs (Exames) nas planilhas: {len(mapa_total)}")
    return mapa_total

def scan_dataset_logica(caminho_imagens, csv_com, csv_sem):
    
    # --- PASSO 1: CARREGAR LISTA DE EXAMES ALVO ---
    mapa_labels = unificar_planilhas(csv_com, csv_sem)
    if not mapa_labels:
        print("Erro crítico: Nenhum exame carregado.")
        return [], {}

    exames_candidatos = {} # Dicionário: { StudyUID: [ (caminho_serie1, qtd_arquivos), ... ] }
    
    print(f"\n Iniciando varredura de pastas em: {caminho_imagens}")
    
    # --- PASSO 2: VARRER O DISCO ---
    for root, _, files in os.walk(caminho_imagens):
        dcms = [f for f in files if f.endswith('.dcm')]
        
        # Filtro de ruído: Ignora pastas com poucos arquivos (Scouts/Dose Report)
        if len(dcms) >= 10:
            try:
                # Lê cabeçalho para identificar a qual EXAME aquela pasta pertence
                first = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
                study_uid = str(first.StudyInstanceUID).strip().replace('\x00', '')
                
                # Se esse exame está na nossa lista (CSV)
                if study_uid in mapa_labels:
                    if study_uid not in exames_candidatos: 
                        exames_candidatos[study_uid] = []
                    
                    # Adiciona essa série como candidata para representar o exame
                    exames_candidatos[study_uid].append( (root, len(dcms)) )
                    
            except Exception:
                continue 

    # --- PASSO 3: SELEÇÃO DA MELHOR SÉRIE POR EXAME ---
    final_paths = []
    final_labels = {}
    stats_descartados_incompletos = 0
    
    print(f"  Analisando pastas de {len(exames_candidatos)} exames encontrados no disco...")

    for study_uid, lista_de_series in exames_candidatos.items():
        
        # Se o exame tem poucas séries no total, pode estar incompleto/corrompido
        if len(lista_de_series) < 2:
            stats_descartados_incompletos += 1
            continue 

        # Seleciona a série com MAIOR número de fatias para representar o exame
        melhor_serie = max(lista_de_series, key=lambda x: x[1])
        
        caminho_vencedor = melhor_serie[0] # Isso é uma PASTA
        label_vencedor = mapa_labels[study_uid]
        
        final_paths.append(caminho_vencedor)
        final_labels[caminho_vencedor] = label_vencedor

    # --- RELATÓRIO ---
    print("\n" + "="*45)
    print(" RESUMO DA SELEÇÃO (Por Exame/Study)")
    print("="*45)
    print(f"1. Exames localizados no disco: {len(exames_candidatos)}")
    print(f"2. Exames descartados (incompletos): {stats_descartados_incompletos}")
    print(f"3. EXAMES SELECIONADOS PARA TREINO:  {len(final_paths)}")
    print("="*45)
    
    final_paths = sorted(final_paths)
    return final_paths, final_labels

# --- CLASSE GERADORA DE DADOS (CORRIGIDA PARA LER PASTAS) ---
class MedicalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        cur_bs = len(list_IDs_temp)
        X = np.empty((cur_bs, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((cur_bs,), dtype=np.int32)

        for i, ID in enumerate(list_IDs_temp):
            # ID aqui é o caminho da PASTA (root) vindo do scan_dataset_logica
            try:
                arquivo_para_ler = ID # Caso já seja arquivo
                
                # Se for pasta, pega o arquivo do meio (middle slice)
                if os.path.isdir(ID):
                    files = sorted([f for f in os.listdir(ID) if f.endswith('.dcm')])
                    if len(files) > 0:
                        meio = len(files) // 2
                        arquivo_para_ler = os.path.join(ID, files[meio])
                    else:
                        raise Exception("Pasta vazia")

                # 1. Ler DICOM
                ds = pydicom.dcmread(arquivo_para_ler)
                img = ds.pixel_array.astype(np.float32)

                # 2. Converter para HU
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                img_hu = img * slope + intercept

                # 3. Aplicar 3 janelamentos no mesmo corte
                img_pulmao = aplicar_janela(img_hu, center=-600, width=1500)
                img_mediastino = aplicar_janela(img_hu, center=40, width=400)
                img_extra = aplicar_janela(img_hu, center=100, width=700)

                # 4. Resize
                img_pulmao = cv2.resize(img_pulmao, self.dim)
                img_mediastino = cv2.resize(img_mediastino, self.dim)
                img_extra = cv2.resize(img_extra, self.dim)

                # 5. Empilhar canais
                img = np.stack([img_pulmao, img_mediastino, img_extra], axis=-1)
                
                X[i,] = img
                y[i] = self.labels[ID] # Label associado à pasta
                
            except Exception as e:
                print(f"Erro ao ler {ID}: {e}")
                X[i] = np.zeros((*self.dim, self.n_channels), dtype=np.float32)
                y[i] = self.labels.get(ID, 0)

        return X, y

# --- BLOCO DE TESTE ---
if __name__ == "__main__":
    
    PASTA_IMAGENS = '/Storage/jerogalsky-2025'
    PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'
    
    ARQUIVO_COM = os.path.join(PASTA_CSVS, 'hc-2025_exames_contraste_novo.csv')
    ARQUIVO_SEM = os.path.join(PASTA_CSVS, 'hc-2025_exames_semcontraste_novo.csv')
    
    if os.path.exists(ARQUIVO_COM) and os.path.exists(ARQUIVO_SEM):
        paths, labels = scan_dataset_logica(PASTA_IMAGENS, ARQUIVO_COM, ARQUIVO_SEM)
        
        if len(paths) > 0:
            print("\n  SUCESSO! Código rodou perfeitamente.")
            print(f"   Exemplo de Exame: {paths[0]}")
            print(f"   Classificação: {labels[paths[0]]}")
        else:
            print("\n  Nenhum exame encontrado. Verifique os UIDs.")
    else:
        print("\n  ERRO: Arquivos CSV não encontrados.")