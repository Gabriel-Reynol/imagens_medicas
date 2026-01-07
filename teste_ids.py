import os
import pydicom
import pandas as pd

PATH_PACIENTES = 'C:/Users/reyno/USP2025/pacientes/'
PATH_PLANILHA = 'C:/Users/reyno/USP2025/IC_imagens/imagens_medicas/patIDStudy_contrast.csv'
COLUNA_ID = 'UID dicom' 

print("--- DIAGNÓSTICO DE IDS ---")

# 1. Mostra como estão os IDs na Planilha
try:
    df = pd.read_csv(PATH_PLANILHA)
    df[COLUNA_ID] = df[COLUNA_ID].astype(str).str.strip() # Limpa espaços extras
    exemplos_csv = df[COLUNA_ID].head(3).tolist()
    print(f"\n[PLANILHA] Primeiros 3 IDs encontrados na coluna '{COLUNA_ID}':")
    for ex in exemplos_csv:
        print(f"   -> {ex}")
except Exception as e:
    print(f"Erro ao ler planilha: {e}")

# 2. Mostra como estão os IDs no DICOM (entrando na primeira pasta que achar)
print(f"\n[DICOM] Procurando um arquivo DICOM para ler...")
encontrou = False

for root, dirs, files in os.walk(PATH_PACIENTES):
    for f in files:
        if f.endswith('.dcm'):
            path_dcm = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(path_dcm, stop_before_pixels=True)
                
                print(f"   Arquivo lido: {f}")
                print(f"   -> SeriesInstanceUID: {dcm.SeriesInstanceUID}")
                print(f"   -> StudyInstanceUID:  {dcm.StudyInstanceUID}")
                print("\nCOMPARAR: Qual desses dois (Series ou Study) se parece com o da Planilha?")
                
                encontrou = True
                break # Para tudo, só precisamos ver um
            except:
                pass
    if encontrou:
        break

if not encontrou:
    print("Não achei nenhum arquivo .dcm para testar.")