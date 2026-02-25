import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import Counter

import fase2_v2 as fase2
import fase3_server_dl


# --- 1. DETECÇÃO DE AMBIENTE ---
SERVER_PATH = '/Storage/jerogalsky-2024'
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

if os.path.exists(SERVER_PATH):
    print(" MODO SERVIDOR")
    CAMINHO_IMG = SERVER_PATH

    PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'
    CAMINHO_CSV_COM = os.path.join(PASTA_CSVS, 'planilha_uid_contraste.csv')
    CAMINHO_CSV_SEM = os.path.join(PASTA_CSVS, 'planilha_uid_SC.csv')

    EPOCHS = 30
    BATCH_SIZE = 32
else:
    print(" MODO PC LOCAL (Teste)")
    CAMINHO_IMG = PC_PATH
    CAMINHO_CSV_COM = 'C:/Users/reyno/USP2025/planilha_uid_contraste.csv'
    CAMINHO_CSV_SEM = 'C:/Users/reyno/USP2025/planilha_uid_SC.csv'

    EPOCHS = 1
    BATCH_SIZE = 4


# --- 2. PREPARAÇÃO (Fase 2) ---
print("\n Iniciando varredura e cruzamento de dados...")
paths, labels_dict = fase2.scan_dataset_logica(CAMINHO_IMG, CAMINHO_CSV_COM, CAMINHO_CSV_SEM)

if not paths:
    raise ValueError("Nenhum exame encontrado! Verifique os caminhos.")

# CRÍTICO: ordem determinística para split reprodutível
paths = sorted(paths)

print(f" Total de exames válidos: {len(paths)}")

# separar treino e validação
labels_list = [labels_dict[p] for p in paths]

X_train, X_val, y_train, y_val = train_test_split(
    paths, labels_list,
    test_size=0.2,
    stratify=labels_list,
    random_state=42
)

# ===============================
# OVERSAMPLING (SÓ NO TREINO)
# ===============================
print("\n[OVERSAMPLING] Balanceando classe minoritária no treino...")

ids_pos = [p for p in X_train if labels_dict[p] == 1]
ids_neg = [p for p in X_train if labels_dict[p] == 0]

print(f"  Antes: Negativos={len(ids_neg)} | Positivos={len(ids_pos)}")

if len(ids_pos) > 0:
    mult = len(ids_neg) // len(ids_pos)   # ex: 932//48 = 19
    X_train_bal = ids_neg + ids_pos * mult
    np.random.shuffle(X_train_bal)
    print(f"  Depois: Negativos={len(ids_neg)} | Positivos={len(ids_pos)*mult} | Total={len(X_train_bal)}")
else:
    print("  AVISO: Nenhum positivo encontrado no treino.")
    X_train_bal = X_train

c_train_bal = Counter([labels_dict[p] for p in X_train_bal])
print(f"   Treino BAL - Sem Contraste (0): {c_train_bal[0]} | Com Contraste (1): {c_train_bal[1]} | Total: {len(X_train_bal)}")

# Relatório de distribuição
c_train = Counter(y_train)
c_val = Counter(y_val)

print("\n RELATÓRIO DO DATASET")
print(f"   Treino - Sem Contraste (0): {c_train[0]} | Com Contraste (1): {c_train[1]} | Total: {len(X_train)}")
print(f"   Val    - Sem Contraste (0): {c_val[0]} | Com Contraste (1): {c_val[1]} | Total: {len(X_val)}\n")

# --- 3. PESOS DE CLASSE ---
unique_classes = np.unique(y_train)
pesos = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train
)
class_weights_dict = dict(zip(unique_classes, pesos))
print(f"  Pesos das classes calculados: {class_weights_dict}")

# --- 4. GERADORES ---
print("  Criando geradores de imagem...")

# Boa prática:
train_gen = fase2.MedicalDataGenerator(X_train_bal, labels_dict, batch_size=BATCH_SIZE, shuffle=True) #ordem dos exames muda a cada época
val_gen   = fase2.MedicalDataGenerator(X_val, labels_dict, batch_size=BATCH_SIZE, shuffle=False) #ordem fixa

# --- 5. TREINO ---
fase3_server_dl.executar_treino(train_gen, val_gen, EPOCHS, class_weights_dict)
