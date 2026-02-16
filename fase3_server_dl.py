import pandas as pd
import tensorflow as tf
import datetime
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def executar_treino(train_gen, val_gen, epochs, class_weights=None):
    # 1. Configuração de Pastas e Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_resultado = f"resultados_treinos/treino_{timestamp}"
    os.makedirs(pasta_resultado, exist_ok=True) # Cria a pasta se não existir

    print(f"\n" + "="*40)
    print(f" SALVANDO TUDO EM: {pasta_resultado}")
    print("="*40 + "\n")

    IMG_SIZE = 224
    LR = 0.00001

    # -- SALVAR CONFIGURAÇÕES (Para o Relatório) --
    with open(f"{pasta_resultado}/config_run.txt", "w") as f:
        f.write(f"DATA: {timestamp}\n")
        f.write(f"MODELO: ResNet50 (Transfer Learning)\n")
        f.write(f"EPOCHS: {epochs}\n")
        f.write(f"LEARNING RATE: {LR}\n")
        f.write(f"IMG SIZE: {IMG_SIZE}\n")
        f.write(f"PESOS DAS CLASSES: {class_weights}\n")
        f.write(f"BATCH SIZE: {train_gen.batch_size}\n")
        f.write(f"QTD TREINO: {len(train_gen.list_IDs)}\n")
    
    # 2. Construir Modelo com Data Augmentation
    print("  Construindo ResNet50 com Data Augmentation...")
    
    # Baixa a ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True 
    
    # --- A MÁGICA DO AUGMENTATION AQUI ---
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # A cada época, a imagem sofre pequenas alterações aleatórias:
    x = RandomFlip("horizontal")(inputs) # Espelha horizontalmente às vezes
    x = RandomRotation(0.1)(x)           # Gira até 10% para os lados
    x = RandomZoom(0.1)(x)               # Dá um leve zoom de até 10%
    
    # Passa a imagem "mexida" para a ResNet estudar
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    # --------------------------------------

    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # 3. Rodar Treino
    print(" Iniciando FIT...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        verbose=1
    )

    # 4. SALVAMENTO COMPLETO
    print("\n Salvando artefatos...")
    
    # A) Salvar Modelo
    model.save(f"{pasta_resultado}/modelo_resnet.keras")
    
    # B) Salvar Histórico CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f"{pasta_resultado}/historico_metrics.csv", index=False)

    # C) GERAR GRÁFICOS 
    # Backend Agg é necessário para salvar gráficos em servidor sem tela
    plt.switch_backend('Agg') 
    
    # Gráfico de Acurácia
    plt.figure()
    plt.plot(hist_df['accuracy'], label='Treino')
    plt.plot(hist_df['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {timestamp}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(f"{pasta_resultado}/grafico_acuracia.png")
    plt.close()

    # Gráfico de Loss (Erro)
    plt.figure()
    plt.plot(hist_df['loss'], label='Treino')
    plt.plot(hist_df['val_loss'], label='Validação')
    plt.title(f'Loss (Erro) - {timestamp}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{pasta_resultado}/grafico_loss.png")
    plt.close()
    
    print(f" TUDO SALVO COM SUCESSO EM: {pasta_resultado}")