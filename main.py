import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

import fase2_v2 as fase2
import fase3_server_dl


# --- 1. DETECÇÃO DE AMBIENTE ---

PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

SERVER_PATH_2024 = '/Storage/jerogalsky-2024'
SERVER_PATH_2025 = '/Storage/jerogalsky-2025'

if os.path.exists(SERVER_PATH_2024) or os.path.exists(SERVER_PATH_2025):
    print(" MODO SERVIDOR")

    PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'

    BASES = [
        {
            "nome": "Base 2024",
            "caminho_img": SERVER_PATH_2024,
            "csv_com": os.path.join(PASTA_CSVS, 'planilha_uid_contraste.csv'),
            "csv_sem": os.path.join(PASTA_CSVS, 'planilha_uid_SC.csv')
        },
        {
            "nome": "Base 2025",
            "caminho_img": SERVER_PATH_2025,
            "csv_com": os.path.join(PASTA_CSVS, 'hc-2025_exames_contraste_novo.csv'),
            "csv_sem": os.path.join(PASTA_CSVS, 'hc-2025_exames_semcontraste_novo.csv')
        }
    ]

    EPOCHS = 30
    BATCH_SIZE = 16

else:
    print(" MODO PC LOCAL (Teste)")

    BASES = [
        {
            "nome": "Base local",
            "caminho_img": PC_PATH,
            "csv_com": 'C:/Users/reyno/USP2025/planilha_uid_contraste.csv',
            "csv_sem": 'C:/Users/reyno/USP2025/planilha_uid_SC.csv'
        }
    ]

    EPOCHS = 1
    BATCH_SIZE = 4


# --- 2. PREPARAÇÃO (Fase 2) ---
print("\n Iniciando varredura e cruzamento de dados...")

paths = []
labels_dict = {}

for base in BASES:
    print(f"\n Lendo {base['nome']}...")

    paths_base, labels_base = fase2.scan_dataset_logica(
        base["caminho_img"],
        base["csv_com"],
        base["csv_sem"]
    )

    print(f"   Exames válidos encontrados em {base['nome']}: {len(paths_base)}")

    paths.extend(paths_base)
    labels_dict.update(labels_base)

if not paths:
    raise ValueError("Nenhum exame encontrado! Verifique os caminhos das bases.")

paths = sorted(paths)
labels_list = [labels_dict[p] for p in paths]

print(f"\n Total de exames válidos: {len(paths)}")

c_total = Counter(labels_list)
print(f"   Total geral - Sem (0): {c_total[0]} | Com (1): {c_total[1]} | Total: {len(paths)}")

# ===============================
# SPLIT 70/20/10 (Treino/Val/Teste)
# ===============================

# 1) separa TESTE (10%)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    paths, labels_list,
    test_size=0.10,
    stratify=labels_list,
    random_state=7
)

# 2) separa TREINO/VAL dentro do restante (90%)
# Queremos VAL = 20% do total => 20/90 = 0.2222...
val_ratio = 0.20 / 0.90

X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp,
    test_size=val_ratio,
    stratify=y_tmp,
    random_state=7
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

X_train_bal = X_train

c_train_bal = Counter([labels_dict[p] for p in X_train_bal])
print(f"  Treino usado: Sem (0)={c_train_bal[0]} | Com (1)={c_train_bal[1]} | Total={len(X_train_bal)}")

# --- 4. GERADORES ---
print("\n Criando geradores...")
train_gen = fase2.MedicalDataGenerator(X_train_bal, labels_dict, batch_size=BATCH_SIZE, shuffle=True)
val_gen   = fase2.MedicalDataGenerator(X_val,       labels_dict, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. SEM PESOS e SEM OVERSAMPLING ---
# Note: class_weights=None
pasta_resultado = fase3_server_dl.executar_treino(train_gen, val_gen, EPOCHS, class_weights=None)

# salvar split do teste para a fase4 usar exatamente o mesmo
np.save(os.path.join(pasta_resultado, "X_test.npy"), np.array(X_test, dtype=object))
np.save(os.path.join(pasta_resultado, "y_test.npy"), np.array(y_test, dtype=np.int32))

print("\n Split de teste salvo em:")
print(f"  {os.path.join(pasta_resultado, 'X_test.npy')}")
print(f"  {os.path.join(pasta_resultado, 'y_test.npy')}")