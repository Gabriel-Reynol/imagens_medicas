import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import fase2_mapeamento
import fase3_server_dl

# --- 1. DETECÇÃO DE AMBIENTE ---
SERVER_PATH = '/Storage/jerogalsky-2024'
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

if os.path.exists(SERVER_PATH):
    print(" MODO SERVIDOR")
    CAMINHO_IMG = SERVER_PATH
    CAMINHO_CSV = '/home/jerogalsky/IC_PAOLA/patIDStudy_contrast_VPaola.csv'
    EPOCHS = 30
    BATCH_SIZE = 32
else:
    print(" MODO PC LOCAL")
    CAMINHO_IMG = PC_PATH
    CAMINHO_CSV = 'C:/Users/reyno/USP2025/IC_imagens/imagens_medicas/patIDStudy_contrast.csv'
    EPOCHS = 1
    BATCH_SIZE = 4 # Batch menor no PC

# --- 2. PREPARAÇÃO (Fase 2) ---
paths, labels_dict = fase2_mapeamento.scan_dataset(CAMINHO_IMG, CAMINHO_CSV)

if not paths:
    raise ValueError("Nenhum paciente encontrado!")

# Separar Treino/Validação
labels_list = [labels_dict[p] for p in paths]
X_train, X_val, y_train, y_val = train_test_split(paths, labels_list, test_size=0.2, stratify=labels_list, random_state=42)

from collections import Counter
contagem = Counter(y_train)
print(f"\n RELATÓRIO DO DATASET (Treino):")
print(f"    Sem Contraste (0): {contagem[0]} exames")
print(f"    Com Contraste (1): {contagem[1]} exames")
print(f"   ∑  Total: {len(X_train)}\n")
# ----------------------------------------------

print(f" Dados: {len(X_train)} treino | {len(X_val)} validação")

# --- 3. CÁLCULO DE PESOS (Balanceamento) ---
# Cálculo aqui na Main porque aqui tem a lista completa y_train
unique_classes = np.unique(y_train)
pesos = class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
class_weights_dict = dict(zip(unique_classes, pesos))
print(f"⚖️ Pesos das classes: {class_weights_dict}")

# --- 4. INSTANCIAR GERADORES ---
# O Gerador agora vai transformar DICOM -> 2D RGB na hora
train_gen = fase2_mapeamento.MedicalDataGenerator(X_train, labels_dict, batch_size=BATCH_SIZE)
val_gen = fase2_mapeamento.MedicalDataGenerator(X_val, labels_dict, batch_size=BATCH_SIZE)

# --- 5. RODAR FASE 3 ---
fase3_server_dl.executar_treino(train_gen, val_gen, EPOCHS, class_weights_dict)