import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from tensorflow.keras.utils import Sequence

# --- CLASSE GERADORA (Fica aqui porque a Main chama fase2.MedicalDataGenerator) ---
class MedicalDataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.IMG_SIZE = 224
        self.CHANNELS = 3
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

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
        X = np.empty((self.batch_size, self.IMG_SIZE, self.IMG_SIZE, self.CHANNELS))
        y = np.empty((self.batch_size), dtype=int)

        for i, folder_path in enumerate(list_IDs_temp):
            try:
                # 1. Ler arquivos DICOM
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
                slices = []
                for f in files:
                    try:
                        ds = pydicom.dcmread(f, stop_before_pixels=False)
                        if hasattr(ds, 'pixel_array'): slices.append(ds)
                    except: pass
                
                if not slices: raise ValueError("Pasta vazia")

                # Ordena (protegido contra falta de InstanceNumber)
                slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
                
                # 2. Pegar a FATIA CENTRAL
                meio_idx = len(slices) // 2
                img = slices[meio_idx].pixel_array.astype(float)

                # 3. Processamento ResNet
                img_resized = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                
                # Normalizar (0 a 1)
                img_norm = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized) + 1e-8)
                
                # RGB Fake
                img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

                X[i,] = img_rgb
                y[i] = self.labels[folder_path]

            except Exception as e:
                # Imagem preta se der erro
                X[i,] = np.zeros((self.IMG_SIZE, self.IMG_SIZE, self.CHANNELS))
                y[i] = 0

        return X, y

# --- FUNÇÃO DE MAPEAMENTO (A lógica nova de agrupar por Study) ---
def scan_dataset(caminho_imagens, caminho_csv):
    print(" Iniciando indexação por EXAME (StudyInstanceUID)...")
    
    df = pd.read_csv(caminho_csv, encoding='latin-1')
    if 'UID dicom' in df.columns:
        df['UID dicom'] = df['UID dicom'].astype(str).str.strip()
    
    df = df.drop_duplicates(subset=['UID dicom'])
    mapa_labels = df.set_index('UID dicom')['Contraste'].to_dict()
    
    exames_candidatos = {} 

    print(" Varrendo pastas...")
    for root, _, files in os.walk(caminho_imagens):
        dcms = [f for f in files if f.endswith('.dcm')]
        if len(dcms) > 10:
            try:
                first = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
                study_uid = str(first.StudyInstanceUID).strip().replace('\x00', '')
                
                if study_uid in mapa_labels:
                    if mapa_labels[study_uid] in [0, 1]:
                        if study_uid not in exames_candidatos: exames_candidatos[study_uid] = []
                        exames_candidatos[study_uid].append( (root, len(dcms)) )
            except: continue

    final_paths = []
    final_labels = {}
    
    print(f" Selecionando melhores séries de {len(exames_candidatos)} exames...")
    
    for study_uid, lista_de_series in exames_candidatos.items():

        if len(lista_de_series) < 2:
             continue  # Pula para o próximo exame, ignorando este
        # Quem tem mais arquivos ganha
        melhor_serie = max(lista_de_series, key=lambda x: x[1])
        caminho = melhor_serie[0]
        final_paths.append(caminho)
        final_labels[caminho] = mapa_labels[study_uid]

    print(f" Indexado: {len(final_paths)} EXAMES únicos.")
    return final_paths, final_labels