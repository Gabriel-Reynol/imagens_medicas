import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

import fase2_v2 as fase2
import fase3_server_dl


# --- 1. DETECÇÃO DE AMBIENTE ---
SERVER_PATH = '/Storage/jerogalsky-2024'
#SERVER_PATH = '/Storage/jerogalsky-2025' #Base nova

PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

if os.path.exists(SERVER_PATH):
    print(" MODO SERVIDOR")
    CAMINHO_IMG = SERVER_PATH

    PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'
    CAMINHO_CSV_COM = os.path.join(PASTA_CSVS, 'planilha_uid_contraste.csv')
    CAMINHO_CSV_SEM = os.path.join(PASTA_CSVS, 'planilha_uid_SC.csv')
    
    #base nova
    #PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'
    #CAMINHO_CSV_COM = os.path.join(PASTA_CSVS, 'hc-2025_exames_contraste_novo.csv')
    #CAMINHO_CSV_SEM = os.path.join(PASTA_CSVS, 'hc-2025_exames_semcontraste_novo.csv')

    EPOCHS = 30
    BATCH_SIZE = 16
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

paths = sorted(paths)
labels_list = [labels_dict[p] for p in paths]

print(f" Total de exames válidos: {len(paths)}")

# ===============================
# SPLIT 70/20/10 (Treino/Val/Teste)
# ===============================

# 1) separa TESTE (10%)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    paths, labels_list,
    test_size=0.10,
    stratify=labels_list,
    random_state=42
)

# 2) separa TREINO/VAL dentro do restante (90%)
# Queremos VAL = 20% do total => 20/90 = 0.2222...
val_ratio = 0.20 / 0.90

X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp,
    test_size=val_ratio,
    stratify=y_tmp,
    random_state=42
)

print("\n RELATÓRIO DO DATASET")
c_train = Counter(y_train)
c_val   = Counter(y_val)
c_test  = Counter(y_test)

print(f"   Treino (70%) - Sem (0): {c_train[0]} | Com (1): {c_train[1]} | Total: {len(X_train)}")
print(f"   Val   (20%)  - Sem (0): {c_val[0]}   | Com (1): {c_val[1]}   | Total: {len(X_val)}")
print(f"   Teste (10%)  - Sem (0): {c_test[0]}  | Com (1): {c_test[1]}  | Total: {len(X_test)}")
print("\nExemplos do TESTE:")
for p in X_test[:5]:
    print(p)

# ===============================
# OVERSAMPLING (SÓ NO TREINO)
# ===============================
#print("\n[OVERSAMPLING] Balanceando classe minoritária no TREINO...")
#
#ids_pos = [p for p in X_train if labels_dict[p] == 1]
#ids_neg = [p for p in X_train if labels_dict[p] == 0]
#
#print(f"  Antes: Negativos={len(ids_neg)} | Positivos={len(ids_pos)}")
#
#if len(ids_pos) > 0:
#    mult = len(ids_neg) // len(ids_pos)  # ex: 816//42 = 19
#    X_train_bal = ids_neg + ids_pos * mult
#    np.random.shuffle(X_train_bal)
#else:
#    print("  AVISO: Nenhum positivo encontrado no treino.")
#    X_train_bal = X_train
#
#c_train_bal = Counter([labels_dict[p] for p in X_train_bal])
#print(f"  Depois: Sem (0)={c_train_bal[0]} | Com (1)={c_train_bal[1]} | Total={len(X_train_bal)}")

# --- 4. GERADORES ---
print("\n Criando geradores...")
train_gen = fase2.MedicalDataGenerator(X_train, labels_dict, batch_size=BATCH_SIZE, shuffle=True)
val_gen   = fase2.MedicalDataGenerator(X_val,       labels_dict, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. CLASS WEIGHTS + TREINO ---
n_neg = c_train[0]
n_pos = c_train[1]
n_total = n_neg + n_pos

class_weights = {
    0: n_total / (2 * n_neg),
    1: n_total / (2 * n_pos)
}

print("\n CLASS WEIGHTS")
print(f"   Classe 0 (Sem Contraste): {class_weights[0]:.4f}")
print(f"   Classe 1 (Com Contraste): {class_weights[1]:.4f}")

pasta_resultado = fase3_server_dl.executar_treino(
    train_gen,
    val_gen,
    EPOCHS,
    class_weights=class_weights
)

# (Opcional) salvar split do teste para a fase4 usar exatamente o mesmo
np.save(os.path.join(pasta_resultado, "X_test.npy"), np.array(X_test, dtype=object))
np.save(os.path.join(pasta_resultado, "y_test.npy"), np.array(y_test, dtype=np.int32))

print("\n Split de teste salvo em:")
print(f"  {os.path.join(pasta_resultado, 'X_test.npy')}")
print(f"  {os.path.join(pasta_resultado, 'y_test.npy')}")