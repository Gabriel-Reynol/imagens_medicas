import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def executar_treino(train_gen, val_gen, epochs, class_weights=None):
    print(f"\n" + "="*40)
    print(f"FASE 3: TREINAMENTO COM RESNET50")
    print("="*40 + "\n")

    IMG_SIZE = 224
    LR = 0.0001

    # 1. Construir ResNet50
    print("Carregando ResNet50 (ImageNet weights)...")
    
    # Input Shape: (224, 224, 3)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)) # Transfer learning
    base_model.trainable = False  # Congela a base para não destruir os pesos pré-treinados

    # 2. Cabeçalho Personalizado 
    x = base_model.output                   # Pega o que a resnet enxergou
    x = GlobalAveragePooling2D()(x)         # Resume a informação
    x = Dense(128, activation='relu')(x)    
    x = Dropout(0.5)(x)                     # Evita vício
    output = Dense(1, activation='sigmoid')(x) 

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # 3. Rodar Treino
    print("Iniciando FIT...")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights, # Recebe os pesos calculados na Main
        verbose=1
    )

    # 4. Salvar com DATA e HORA (Segurança para não sobrescrever)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nome_modelo = f"modelo_resnet_contraste_{timestamp}.keras"
    nome_csv = f"historico_treino_{timestamp}.csv"
    
    print("\n Salvando resultados...")
    model.save(nome_modelo)
    pd.DataFrame(history.history).to_csv(nome_csv)
    
    print(f"Treino concluído!")
    print(f"   --> Modelo salvo: {nome_modelo}")
    print(f"   --> Histórico salvo: {nome_csv}")