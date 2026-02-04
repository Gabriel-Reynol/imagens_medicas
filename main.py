import os
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# --- IMPORTS DOS ARQUIVOS ---
# Importa do arquivo novo (V2)
import fase2_v2 as fase2 
import fase3_server_dl

# --- 1. DETECÇÃO DE AMBIENTE ---
SERVER_PATH = '/Storage/jerogalsky-2024'
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

if os.path.exists(SERVER_PATH):
    print(" MODO SERVIDOR")
    CAMINHO_IMG = SERVER_PATH
    # Caminhos ATUALIZADOS para os seus 2 CSVs
    CAMINHO_CSV_COM = '/home/jerogalsky/tabelasSeparadas/planilha_uid_contraste.csv'
    CAMINHO_CSV_SEM = '/home/jerogalsky/tabelasSeparadas/planilha_uid_SC.csv'
    EPOCHS = 30
    BATCH_SIZE = 32
else:
    print(" MODO PC LOCAL (Teste)")
    # Caminhos PC
    CAMINHO_IMG = PC_PATH
    CAMINHO_CSV_COM = 'C:/Users/reyno/USP2025/planilha_uid_contraste.csv'
    CAMINHO_CSV_SEM = 'C:/Users/reyno/USP2025/planilha_uid_SC.csv'
    EPOCHS = 1
    BATCH_SIZE = 4

# --- 2. PREPARAÇÃO (Fase 2 - Nova Lógica) ---
print("\n Iniciando varredura e cruzamento de dados...")

# Chama a função nova 
paths, labels_dict = fase2.scan_dataset_logica(CAMINHO_IMG, CAMINHO_CSV_COM, CAMINHO_CSV_SEM)

if not paths:
    raise ValueError(" Nenhum exame encontrado! Verifique os caminhos.")

print(f" Total de exames válidos para treino: {len(paths)}")

# Separar Treino/Validação
labels_list = [labels_dict[p] for p in paths]
X_train, X_val, y_train, y_val = train_test_split(paths, labels_list, test_size=0.2, stratify=labels_list, random_state=42)

from collections import Counter
contagem = Counter(y_train)
print(f"\n RELATÓRIO DO DATASET (Treino):")
print(f"   Sem Contraste (0): {contagem[0]} exames")
print(f"   Com Contraste (1): {contagem[1]} exames")
print(f"   Total Treino: {len(X_train)}")
print(f"   Total Validação: {len(X_val)}\n")

# --- 3. CÁLCULO DE PESOS (Balanceamento) ---
unique_classes = np.unique(y_train)
pesos = class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
class_weights_dict = dict(zip(unique_classes, pesos))
print(f"  Pesos das classes calculados: {class_weights_dict}")

# --- 4. INSTANCIAR GERADORES ---
print("  Criando geradores de imagem...")
# Usa a classe adicionada no fase2_v2
train_gen = fase2.MedicalDataGenerator(X_train, labels_dict, batch_size=BATCH_SIZE)
val_gen = fase2.MedicalDataGenerator(X_val, labels_dict, batch_size=BATCH_SIZE)

# --- 5. RODAR FASE 3 (Deep Learning) ---
fase3_server_dl.executar_treino(train_gen, val_gen, EPOCHS, class_weights_dict)