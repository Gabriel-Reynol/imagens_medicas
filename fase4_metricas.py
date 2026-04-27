import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
import fase2_v2 as fase2

plt.switch_backend('Agg')

# --- CONFIGURAÇÕES ---
PASTA_RESULTADO = '/home/jerogalsky/projeto_gabriel/imagens_medicas/resultados_treinos/treino_basesJuntas'
#a pasta do treino atual

# Prefere o melhor modelo, se existir
ARQUIVO_MODELO_MELHOR = os.path.join(PASTA_RESULTADO, 'modelo_melhor_val_auc.h5')
ARQUIVO_MODELO_FINAL  = os.path.join(PASTA_RESULTADO, 'modelo_final.h5')

if os.path.exists(ARQUIVO_MODELO_MELHOR):
    ARQUIVO_MODELO = ARQUIVO_MODELO_MELHOR
elif os.path.exists(ARQUIVO_MODELO_FINAL):
    ARQUIVO_MODELO = ARQUIVO_MODELO_FINAL
else:
    raise FileNotFoundError("Não encontrei modelo melhor nem modelo final na pasta.")

# --- Detecta ambiente e caminhos ---
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

SERVER_PATH_2024 = '/Storage/jerogalsky-2024'
SERVER_PATH_2025 = '/Storage/jerogalsky-2025'

if os.path.exists(SERVER_PATH_2024) or os.path.exists(SERVER_PATH_2025):
    print("--- MODO SERVIDOR DETECTADO ---")

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

    BATCH_SIZE = 32

else:
    print("--- MODO PC LOCAL DETECTADO ---")

    BASES = [
        {
            "nome": "Base local",
            "caminho_img": PC_PATH,
            "csv_com": 'C:/Users/reyno/USP2025/planilha_uid_contraste.csv',
            "csv_sem": 'C:/Users/reyno/USP2025/planilha_uid_SC.csv'
        }
    ]

    BATCH_SIZE = 4

# --- 1) Recria labels_dict juntando as bases ---
print("\n Carregando lista de exames (para obter labels_dict)...")

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

print(f"\n Total de exames válidos juntando as bases: {len(paths)}")

# --- 2) Carrega o TESTE salvo pela main (70/20/10) ---
X_TEST_PATH = os.path.join(PASTA_RESULTADO, "X_test.npy")
Y_TEST_PATH = os.path.join(PASTA_RESULTADO, "y_test.npy")

if not (os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH)):
    raise FileNotFoundError(
        f"Não achei X_test.npy / y_test.npy em {PASTA_RESULTADO}. "
        "Confirme que sua main salvou esses arquivos."
    )

X_test = np.load(X_TEST_PATH, allow_pickle=True).tolist()
y_test = np.load(Y_TEST_PATH, allow_pickle=True).astype(np.int32)

print(f" Total de Exames no TESTE: {len(X_test)}")

test_gen = fase2.MedicalDataGenerator(X_test, labels_dict, batch_size=BATCH_SIZE, shuffle=False)

# --- 3) Carregar modelo ---
print(f"\n Carregando modelo: {ARQUIVO_MODELO}")
model = load_model(ARQUIVO_MODELO)
print(" Modelo carregado com sucesso!")

# --- 4) Predições ---
print("\n Realizando inferência no TESTE...")
predictions = model.predict(test_gen, verbose=1).ravel()
predicted_classes = (predictions > 0.5).astype(np.int32)

true_classes = y_test

if len(true_classes) != len(predicted_classes):
    raise RuntimeError(
        f"ERRO: Tamanhos diferentes! true={len(true_classes)} pred={len(predicted_classes)}."
    )

# --- 5) Relatórios ---
print("\n" + "="*40)
print(" RELATÓRIO FINAL DE PERFORMANCE (TESTE)")
print("="*40)

cm = confusion_matrix(true_classes, predicted_classes)
print("\nMatriz de Confusão:")
print(cm)

# --- Métricas derivadas da matriz de confusão ---
tn, fp, fn, tp = cm.ravel()

sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\nMétricas adicionais:")
print(f"Sensibilidade (Com Contraste): {sensibilidade:.4f}")
print(f"Especificidade (Sem Contraste): {especificidade:.4f}")

# Matriz
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Sem Contraste', 'Com Contraste'],
    yticklabels=['Sem Contraste', 'Com Contraste']
)
plt.xlabel('Predição do Modelo')
plt.ylabel('Realidade (Gabarito)')
plt.title('Matriz de Confusão - TESTE')
plt.savefig(f"{PASTA_RESULTADO}/matriz_confusao_teste.png")
plt.close()
print(f" Gráfico salvo em: {PASTA_RESULTADO}/matriz_confusao_teste.png")

# Classification report
report = classification_report(
    true_classes, predicted_classes,
    target_names=['Sem Contraste', 'Com Contraste'],
    zero_division=0
)
print("\nResumo Detalhado:")
print(report)

with open(f"{PASTA_RESULTADO}/relatorio_metrics_teste.txt", "w") as f:
    f.write(report)
    f.write(f"\n\nMatriz de Confusão:\n{cm}")
    f.write("\n\nMétricas adicionais:")
    f.write(f"\nSensibilidade (Com Contraste): {sensibilidade:.4f}")
    f.write(f"\nEspecificidade (Sem Contraste): {especificidade:.4f}")

# ROC / AUC
fpr, tpr, thresholds = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('ROC - TESTE')
plt.legend(loc="lower right")
plt.savefig(f"{PASTA_RESULTADO}/curva_roc_teste.png")
plt.close()
print(f" Curva ROC salva em: {PASTA_RESULTADO}/curva_roc_teste.png")