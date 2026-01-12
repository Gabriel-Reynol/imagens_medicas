import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from tensorflow.keras.utils import Sequence

# --- CONFIGURAÇÕES PARA RESNET50 ---
IMG_SIZE = 224  # ResNet exige 224x224
CHANNELS = 3    # ResNet exige 3 canais (RGB)

class MedicalDataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
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
        # Cria arrays vazios no formato da ResNet (Batch, 224, 224, 3)
        X = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, CHANNELS))
        y = np.empty((self.batch_size), dtype=int)

        for i, folder_path in enumerate(list_IDs_temp):
            try:
                # 1. Ler arquivos DICOM
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
                slices = []
                for f in files:
                    try:
                        ds = pydicom.dcmread(f)
                        if hasattr(ds, 'pixel_array'): slices.append(ds)
                    except: pass
                
                # Se der erro ou pasta vazia, retorna imagem preta
                if not slices: raise ValueError("Pasta vazia")

                slices.sort(key=lambda x: int(x.InstanceNumber))
                
                # 2. Pegar a FATIA CENTRAL (Lógica 2D)
                # Sua lógica original: transformar 3D em 2D pegando o meio
                meio_idx = len(slices) // 2
                img = slices[meio_idx].pixel_array

                # 3. Processamento para ResNet
                # Resize para 224x224
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Normalizar (0 a 1)
                img_norm = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized) + 1e-8)
                
                # Empilhar 3 vezes para virar RGB (Simulado)
                img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

                X[i,] = img_rgb
                y[i] = self.labels[folder_path]

            except Exception as e:
                # print(f"Erro em {folder_path}: {e}") # Descomente para debugar
                X[i,] = np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS))
                y[i] = 0

        return X, y

def scan_dataset(caminho_imagens, caminho_csv):
    """Varre as pastas e cria a lista de pacientes validos"""
    print(" Indexando dataset (lendo CSV e pastas)...")
    
    df = pd.read_csv(caminho_csv, encoding='latin-1')
    df = df.drop_duplicates(subset=['UID dicom'])
    mapa = df.set_index('UID dicom')['Contraste'].to_dict()
    
    paths = []
    labels = {}
    count = 0
    
    for root, _, files in os.walk(caminho_imagens):
        dcms = [f for f in files if f.endswith('.dcm')]
        if len(dcms) > 10:
            try:
                # Validação rápida pelo header
                first = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
                uid = first.StudyInstanceUID
                
                # Filtros
                if 'AXIAL' not in first.ImageType: continue
                if uid not in mapa: continue
                if mapa[uid] not in [0, 1]: continue
                
                paths.append(root)
                labels[root] = mapa[uid]
                count += 1
            except: continue
            
            
    print(f" Indexado: {len(paths)} pacientes encontrados.")
    return paths, labels