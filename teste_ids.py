import os
import pandas as pd
import pydicom
import fase2_mapeamento  # usando a lógica REAL do projeto

# --- CONFIGURAÇÃO DE CAMINHOS (Igual à Main) ---
SERVER_PATH = '/Storage/jerogalsky-2024'
PC_PATH = 'C:/Users/reyno/USP2025/pacientes/' # caminho local

if os.path.exists(SERVER_PATH):
    print(" MODO SERVIDOR DETECTADO")
    CAMINHO_IMG = SERVER_PATH
    CAMINHO_CSV = '/home/jerogalsky/IC_PAOLA/patIDStudy_contrast_VPaola.csv'
else:
    print(" MODO PC LOCAL DETECTADO")
    CAMINHO_IMG = PC_PATH
    CAMINHO_CSV = 'C:/Users/reyno/USP2025/IC_imagens/imagens_medicas/patIDStudy_contrast.csv' 

def gerar_relatorio():
    print("\n" + "="*50)
    print(" INICIANDO AUDITORIA DO DATASET")
    print("="*50)

    # 1. Pega EXATAMENTE o que o treino vai usar
    # Isso garante que estamos auditando a lógica real, não uma simulação
    paths, labels_dict = fase2_mapeamento.scan_dataset(CAMINHO_IMG, CAMINHO_CSV)

    if not paths:
        print(" Nenhum paciente encontrado. Verifique os caminhos.")
        return

    print(f"\n Gerando relatório para {len(paths)} exames selecionados...")
    
    dados_relatorio = []

    for i, folder_path in enumerate(paths):
        try:
            # Pega o primeiro DCM da pasta só para ler o UID
            arquivos = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
            qtd_arquivos = len(arquivos)
            
            if qtd_arquivos > 0:
                first_dcm = pydicom.dcmread(os.path.join(folder_path, arquivos[0]), stop_before_pixels=True)
                uid_dicom = str(first_dcm.StudyInstanceUID).strip()
            else:
                uid_dicom = "ERRO_PASTA_VAZIA"

            # Busca o label que o sistema atribuiu
            label = labels_dict[folder_path]
            classe_nome = "COM CONTRASTE" if label == 1 else "SEM CONTRASTE"

            # Adiciona na lista
            dados_relatorio.append({
                'Index': i + 1,
                'UID_no_DICOM': uid_dicom,
                'Classe_Atribuida': label,
                'Descricao': classe_nome,
                'Qtd_Imagens': qtd_arquivos,
                'Pasta_Escolhida': folder_path
            })
            
            # Barra de progresso visual simples
            if i % 50 == 0:
                print(f"   Processado: {i}/{len(paths)}...")

        except Exception as e:
            print(f" Erro ao ler pasta {folder_path}: {e}")

    # 2. Salva em CSV para abrir no Excel
    nome_arquivo = "auditoria_dataset_final.csv"
    df_resultado = pd.DataFrame(dados_relatorio)
    df_resultado.to_csv(nome_arquivo, index=False, sep=';')

    print("\n" + "="*50)
    print(f" RELATÓRIO CONCLUÍDO!")
    print(f" Arquivo salvo: {nome_arquivo}")
    print("="*50)
    print("Abra este arquivo e confira se os UIDs batem com o esperado.")
    
    # Mostra uma amostra grátis no terminal
    print("\n--- Amostra dos 5 primeiros ---")
    print(df_resultado[['UID_no_DICOM', 'Classe_Atribuida', 'Qtd_Imagens']].head().to_string())

if __name__ == "__main__":
    gerar_relatorio()