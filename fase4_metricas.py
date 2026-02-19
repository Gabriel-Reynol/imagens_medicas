import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
import fase2_v2 as fase2 

# --- CORREÇÃO DE VÍDEO ---
# Sem isso, o código trava ao tentar criar gráficos
plt.switch_backend('Agg')

# --- CONFIGURAÇÕES ---
# caminho onde o treino salvou o modelo
PASTA_RESULTADO = '/home/jerogalsky/projeto_gabriel/imagens_medicas/resultados_treinos/treino_20260216_184905'
ARQUIVO_MODELO = os.path.join(PASTA_RESULTADO, 'modelo_resnet.keras')

# Caminhos
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

# --- 1. CARREGAR DADOS ---
print("\n Carregando lista de imagens...")
paths, labels_dict = fase2.scan_dataset_logica(CAMINHO_IMG, CAMINHO_CSV_COM, CAMINHO_CSV_SEM)

# Recriar a divisão
from sklearn.model_selection import train_test_split
labels_list = [labels_dict[p] for p in paths]
_, X_val, _, y_val = train_test_split(paths, labels_list, test_size=0.2, stratify=labels_list, random_state=42)

print(f" Total de Imagens para Teste (Validação): {len(X_val)}")

# Criar o Gerador
val_gen = fase2.MedicalDataGenerator(X_val, labels_dict, batch_size=BATCH_SIZE, shuffle=False) 
#shuffle=False é crucial para bater a ordem das previsões com as respostas reais!

# --- 2. CARREGAR O MODELO ---
print(f"\n Carregando o modelo treinado: {ARQUIVO_MODELO}")
if not os.path.exists(ARQUIVO_MODELO):
    raise FileNotFoundError("ERRO: Não achei o arquivo do modelo.")

model = load_model(ARQUIVO_MODELO)
print(" Modelo carregado com sucesso!")

# --- 3. FAZER PREVISÕES E AJUSTE TÉCNICO ---
print("\n Realizando inferência (Isso pode demorar um pouquinho)...")

predictions = model.predict(val_gen, verbose=1)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# Pega o gabarito original COMPLETO
true_classes_full = np.array([labels_dict[x] for x in X_val])

# Lógica de CORREÇÃO DO TAMANHO 
n_pred = len(predicted_classes)
n_true = len(true_classes_full)

if n_pred != n_true:
    print(f"\n AJUSTE DE TAMANHO: O gerador processou {n_pred} imagens, mas tínhamos {n_true}.")
    print(f"   Cortando o gabarito para bater com as previsões...")
    true_classes = true_classes_full[:n_pred]
else:
    true_classes = true_classes_full

print(f" Tamanhos sincronizados: {len(true_classes)} imagens.")

# --- 4. GERAR RELATÓRIOS ---
print("\n" + "="*40)
print(" RELATÓRIO FINAL DE PERFORMANCE")
print("="*40)

#matrix de confusão
cm = confusion_matrix(true_classes, predicted_classes)
print("\nMatriz de Confusão:")
print(cm)

# Plotar Matriz Bonita
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sem Contraste', 'Com Contraste'], 
            yticklabels=['Sem Contraste', 'Com Contraste'])
plt.xlabel('Predição do Modelo')
plt.ylabel('Realidade (Gabarito)')
plt.title('Matriz de Confusão - Validação')
plt.savefig(f"{PASTA_RESULTADO}/matriz_confusao.png")
print(f" Gráfico salvo em: {PASTA_RESULTADO}/matriz_confusao.png")

# Relatório de Texto
report = classification_report(true_classes, predicted_classes, target_names=['Sem Contraste', 'Com Contraste'])
print("\nResumo Detalhado:")
print(report)

#salvar relatório em texto
with open(f"{PASTA_RESULTADO}/relatorio_metrics.txt", "w") as f:
    f.write(report)
    f.write(f"\n\nMatriz de Confusão:\n{cm}")

# Curva ROC 
fpr, tpr, thresholds = roc_curve(true_classes, predictions.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(f"{PASTA_RESULTADO}/curva_roc.png")
print(f" Curva ROC salva em: {PASTA_RESULTADO}/curva_roc.png")