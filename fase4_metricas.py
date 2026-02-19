import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
import fase2_v2 as fase2

plt.switch_backend('Agg')

# --- CONFIGURAÇÕES ---
PASTA_RESULTADO = '/home/jerogalsky/projeto_gabriel/imagens_medicas/resultados_treinos/treino_20260216_184905'

# Preferir o melhor modelo, se existir
ARQUIVO_MODELO_MELHOR = os.path.join(PASTA_RESULTADO, 'modelo_melhor_val_auc.keras')
ARQUIVO_MODELO_FINAL  = os.path.join(PASTA_RESULTADO, 'modelo_final.keras')
ARQUIVO_MODELO_ANTIGO = os.path.join(PASTA_RESULTADO, 'modelo_resnet.keras')

if os.path.exists(ARQUIVO_MODELO_MELHOR):
    ARQUIVO_MODELO = ARQUIVO_MODELO_MELHOR
elif os.path.exists(ARQUIVO_MODELO_FINAL):
    ARQUIVO_MODELO = ARQUIVO_MODELO_FINAL
else:
    ARQUIVO_MODELO = ARQUIVO_MODELO_ANTIGO

SERVER_PATH = '/Storage/jerogalsky-2024'
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

if os.path.exists(SERVER_PATH):
    print("--- MODO SERVIDOR DETECTADO ---")
    CAMINHO_IMG = SERVER_PATH
    PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'
    CAMINHO_CSV_COM = os.path.join(PASTA_CSVS, 'planilha_uid_contraste.csv')
    CAMINHO_CSV_SEM = os.path.join(PASTA_CSVS, 'planilha_uid_SC.csv')
    BATCH_SIZE = 32
else:
    print("--- MODO PC LOCAL DETECTADO ---")
    CAMINHO_IMG = PC_PATH
    CAMINHO_CSV_COM = 'C:/Users/reyno/USP2025/planilha_uid_contraste.csv'
    CAMINHO_CSV_SEM = 'C:/Users/reyno/USP2025/planilha_uid_SC.csv'
    BATCH_SIZE = 4

# --- 1) CARREGAR DADOS ---
print("\n Carregando lista de exames...")
paths, labels_dict = fase2.scan_dataset_logica(CAMINHO_IMG, CAMINHO_CSV_COM, CAMINHO_CSV_SEM)

# CRÍTICO: mesma regra da main para split reprodutível
paths = sorted(paths)

from sklearn.model_selection import train_test_split
labels_list = [labels_dict[p] for p in paths]
_, X_val, _, y_val = train_test_split(
    paths, labels_list,
    test_size=0.2,
    stratify=labels_list,
    random_state=42
)

print(f" Total de Exames na Validação: {len(X_val)}")

val_gen = fase2.MedicalDataGenerator(X_val, labels_dict, batch_size=BATCH_SIZE, shuffle=False)

# --- 2) CARREGAR MODELO ---
print(f"\n Carregando modelo: {ARQUIVO_MODELO}")
if not os.path.exists(ARQUIVO_MODELO):
    raise FileNotFoundError(f"ERRO: Não achei o modelo em: {ARQUIVO_MODELO}")

model = load_model(ARQUIVO_MODELO)
print(" Modelo carregado com sucesso!")

# --- 3) PREDIÇÕES ---
print("\n Realizando inferência...")
predictions = model.predict(val_gen, verbose=1).ravel()
predicted_classes = (predictions > 0.5).astype(np.int32)

#  o generator agora não descarta batch
true_classes = np.array([labels_dict[x] for x in X_val], dtype=np.int32)

if len(true_classes) != len(predicted_classes):
    raise RuntimeError(
        f"ERRO: Tamanhos diferentes! true={len(true_classes)} pred={len(predicted_classes)}. "
        "Verifique se o generator está atualizado (ceil + batch dinâmico)."
    )

# --- 4) RELATÓRIOS ---
print("\n" + "="*40)
print(" RELATÓRIO FINAL DE PERFORMANCE")
print("="*40)

#matriz de confusão
cm = confusion_matrix(true_classes, predicted_classes)
print("\nMatriz de Confusão:")
print(cm)

#plotar matriz
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Sem Contraste', 'Com Contraste'],
    yticklabels=['Sem Contraste', 'Com Contraste']
)
plt.xlabel('Predição do Modelo')
plt.ylabel('Realidade (Gabarito)')
plt.title('Matriz de Confusão - Validação')
plt.savefig(f"{PASTA_RESULTADO}/matriz_confusao.png")
print(f" Gráfico salvo em: {PASTA_RESULTADO}/matriz_confusao.png")

#relatório texto
report = classification_report(true_classes, predicted_classes, target_names=['Sem Contraste', 'Com Contraste'])
print("\nResumo Detalhado:")
print(report)

#salvar relatório em texto
with open(f"{PASTA_RESULTADO}/relatorio_metrics.txt", "w") as f:
    f.write(report)
    f.write(f"\n\nMatriz de Confusão:\n{cm}")

fpr, tpr, thresholds = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(f"{PASTA_RESULTADO}/curva_roc.png")
print(f" Curva ROC salva em: {PASTA_RESULTADO}/curva_roc.png")
