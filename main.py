import os
import fase2_mapeamento
import fase3_server_dl

# --- PAINEL DE CONTROLE ---
RODAR_FASE_2 = True   # True na primeira vez para criar o dataset
RODAR_FASE_3 = True   # True para treinar

# --- 1. DETECÇÃO DE AMBIENTE ---
SERVER_PATH = '/Storage/jerogalsky-2024'
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/'

if os.path.exists(SERVER_PATH):
    print("MODO SERVIDOR")
    CAMINHO_IMG = SERVER_PATH
    CAMINHO_CSV = '/home/jerogalsky/IC_PAOLA/patIDStudy_contrast_VPaola'
    EPOCHS = 30 # Treino Valendo!
else:
    print("MODO PC LOCAL")
    CAMINHO_IMG = PC_PATH
    CAMINHO_CSV = 'C:/Users/reyno/USP2025/IC_imagens/imagens_medicas/patIDStudy_contrast.csv'
    EPOCHS = 1  # Teste rápido

# --- 2. EXECUÇÃO ---
if RODAR_FASE_2:
    fase2_mapeamento.executar_fase2(CAMINHO_IMG, CAMINHO_CSV)

if RODAR_FASE_3:
    fase3_server_dl.executar_treino(EPOCHS)

print("\nFIM DO PIPELINE.")