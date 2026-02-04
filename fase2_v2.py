import os
import pandas as pd
import pydicom

def unificar_planilhas(csv_com, csv_sem):
    """
    Lê os dois arquivos CSV e cria um dicionário de IDs de ESTUDOS (Exames).
    Retorna: { 'UID_Estudo_123': 1, 'UID_Estudo_456': 0, ... }
    """
    print("📋 Lendo planilhas de exames (Study UIDs)...")
    
    # Procura pela coluna que identifica o EXAME (Study)
    possiveis_colunas = ['Study Instance UID', 'UID dicom', 'StudyInstanceUID', 'StudyID']
    
    def carregar(caminho, label):
        try:
            df = pd.read_csv(caminho, encoding='latin-1')
        except FileNotFoundError:
            print(f"ERRO: Arquivo não encontrado: {caminho}")
            return {}

        col = next((c for c in possiveis_colunas if c in df.columns), None)
        if not col: 
            print(f"AVISO: Coluna de UID de Estudo não encontrada em {caminho}")
            return {}
        
        # Limpa sujeira e cria dicionário
        df[col] = df[col].astype(str).str.strip().str.replace('"', '').str.replace("'", "")
        return dict.fromkeys(df[col], label)

    dict_com = carregar(csv_com, 1) 
    dict_sem = carregar(csv_sem, 0)
    
    mapa_total = {**dict_sem, **dict_com} 
    print(f"   -> Total de UIDs (Exames) nas planilhas: {len(mapa_total)}")
    return mapa_total

def scan_dataset_logica(caminho_imagens, csv_com, csv_sem):
    
    # --- PASSO 1: CARREGAR LISTA DE EXAMES ALVO ---
    mapa_labels = unificar_planilhas(csv_com, csv_sem)
    if not mapa_labels:
        print("Erro crítico: Nenhum exame carregado.")
        return [], {}

    exames_candidatos = {} # Dicionário: { StudyUID: [ (caminho_serie1, qtd_arquivos), ... ] }
    
    print(f"\n Iniciando varredura de pastas em: {caminho_imagens}")
    
    # --- PASSO 2: VARRER O DISCO ---
    for root, _, files in os.walk(caminho_imagens):
        dcms = [f for f in files if f.endswith('.dcm')]
        
        # Filtro de ruído: Ignora pastas com poucos arquivos (Scouts/Dose Report)
        if len(dcms) >= 10:
            try:
                # Lê cabeçalho para identificar a qual EXAME aquela pasta pertence
                first = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
                study_uid = str(first.StudyInstanceUID).strip().replace('\x00', '')
                
                # Se esse exame está na nossa lista (CSV)
                if study_uid in mapa_labels:
                    if study_uid not in exames_candidatos: 
                        exames_candidatos[study_uid] = []
                    
                    # Adiciona essa série como candidata para representar o exame
                    exames_candidatos[study_uid].append( (root, len(dcms)) )
                    
            except Exception:
                continue 

    # --- PASSO 3: SELEÇÃO DA MELHOR SÉRIE POR EXAME ---
    final_paths = []
    final_labels = {}
    stats_descartados_incompletos = 0
    
    print(f"  Analisando pastas de {len(exames_candidatos)} exames encontrados no disco...")

    for study_uid, lista_de_series in exames_candidatos.items():
        
        # Se o exame tem poucas séries no total, pode estar incompleto/corrompido
        if len(lista_de_series) < 2:
            stats_descartados_incompletos += 1
            continue 

        # Seleciona a série com MAIOR número de fatias para representar o exame
        melhor_serie = max(lista_de_series, key=lambda x: x[1])
        
        caminho_vencedor = melhor_serie[0]
        label_vencedor = mapa_labels[study_uid]
        
        final_paths.append(caminho_vencedor)
        final_labels[caminho_vencedor] = label_vencedor

    # --- RELATÓRIO ---
    print("\n" + "="*45)
    print("📊 RESUMO DA SELEÇÃO (Por Exame/Study)")
    print("="*45)
    print(f"1. Exames localizados no disco: {len(exames_candidatos)}")
    print(f"2. Exames descartados (incompletos): {stats_descartados_incompletos}")
    print(f"3. EXAMES SELECIONADOS PARA TREINO:  {len(final_paths)}")
    print("="*45)
    
    return final_paths, final_labels

# --- BLOCO DE TESTE ---
if __name__ == "__main__":
    
    PASTA_IMAGENS = '/Storage/jerogalsky-2024'
    PASTA_CSVS = '/home/jerogalsky/tabelasSeparadas'
    
    ARQUIVO_COM = os.path.join(PASTA_CSVS, 'planilha_uid_contraste.csv')
    ARQUIVO_SEM = os.path.join(PASTA_CSVS, 'planilha_uid_SC.csv')
    
    if os.path.exists(ARQUIVO_COM) and os.path.exists(ARQUIVO_SEM):
        paths, labels = scan_dataset_logica(PASTA_IMAGENS, ARQUIVO_COM, ARQUIVO_SEM)
        
        if len(paths) > 0:
            print("\n SUCESSO! Código rodou perfeitamente.")
            print(f"   Exemplo de Exame: {paths[0]}")
            print(f"   Classificação: {labels[paths[0]]}")
        else:
            print("\n Nenhum exame encontrado. Verifique os UIDs.")
    else:
        print("\n ERRO: Arquivos CSV não encontrados.")