import os
import numpy as np
import pandas as pd  # Adicionado para salvar o histórico
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import glob

# --- 1. GERADOR DE DADOS (Lê os arquivos .npy rápidos) ---
class NpyGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, img_size, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x_paths = self.file_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        images = []
        for path in batch_x_paths:
            try:
                # Carrega o arquivo .npy (já processado na Fase 2)
                vol = np.load(path)
                
                # Se for 3D, pega a fatia central
                if vol.ndim == 3:
                    meio = vol.shape[0] // 2
                    img = vol[meio]
                else:
                    img = vol 

                # Normalização (0 a 1)
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                
                # Ajuste para ResNet (precisa de 3 canais)
                # Redimensiona para 224x224
                img_resized = tf.image.resize(img[..., np.newaxis], (self.img_size, self.img_size)).numpy()
                # Empilha 3 vezes para simular RGB
                img_final = np.concatenate([img_resized, img_resized, img_resized], axis=-1)

                images.append(img_final)
            except Exception as e:
                print(f"Erro no arquivo {path}: {e}")
                images.append(np.zeros((self.img_size, self.img_size, 3)))

        return np.array(images), np.array(batch_y)

# --- 2. FUNÇÃO PRINCIPAL DE TREINO (Controlada pela Main) ---

def executar_treino(epochs_num):
    print(f"\n" + "="*40)
    print(f"FASE 3: TREINAMENTO (Epochs: {epochs_num})")
    print("="*40 + "\n")

    # Configurações
    BATCH_SIZE = 16 
    IMG_SIZE = 224
    LR = 0.0001
    DATA_DIR = os.path.join(os.getcwd(), 'dataset')

    # 1. Listar arquivos .npy nas pastas criadas pela Fase 2
    print("Mapeando arquivos .npy...")
    sem_contraste = glob.glob(os.path.join(DATA_DIR, "0_sem_contraste", "*.npy"))
    com_contraste = glob.glob(os.path.join(DATA_DIR, "1_com_contraste", "*.npy"))

    arquivos = sem_contraste + com_contraste
    labels = [0] * len(sem_contraste) + [1] * len(com_contraste)

    print(f"   - Sem Contraste (0): {len(sem_contraste)}")
    print(f"   - Com Contraste (1): {len(com_contraste)}")
    print(f"   - Total: {len(arquivos)}")

    if len(arquivos) == 0:
        print("ERRO: Nenhum arquivo .npy encontrado. Rode a Fase 2 (preparação) primeiro!")
        return

    # 2. Split Treino/Validação
    X_train, X_val, y_train, y_val = train_test_split(
        arquivos, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 3. Pesos (Balanceamento Automático)
    pesos = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), pesos))
    print(f"Pesos calculados para balancear as classes: {class_weights}")

    # 4. Criar Geradores
    train_gen = NpyGenerator(X_train, y_train, BATCH_SIZE, IMG_SIZE, augment=True)
    val_gen = NpyGenerator(X_val, y_val, BATCH_SIZE, IMG_SIZE, augment=False)

    # 5. Modelo ResNet50
    print("Construindo ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False # Congela o aprendizado da base

    # Cabeçalho Personalizado (lógica original)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # 6. Rodar Treino
    print("Iniciando FIT...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_num,
        class_weight=class_weights,
        verbose=1
    )

    # 7. Salvar Modelo e Histórico
    model.save("modelo_final_contraste.h5")
    pd.DataFrame(history.history).to_csv('historico_treino.csv') # Salva a tabela de acurácia
    print("\nTreino concluído!")
    print("Modelo salvo: 'modelo_final_contraste.h5'")
    print("Histórico salvo: 'historico_treino.csv'")

# Teste local rápido (Só roda se der play neste arquivo direto)
if __name__ == "__main__":
    executar_treino(1)