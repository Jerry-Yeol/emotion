from keras import layers, models, applications, losses
from keras import backend as K

def EmoNet():
    input_tensor = layers.Input(shape=(None, None, 1)) # 48 x 48
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    x = layers.MaxPooling2D(2)(x) # 24 x 24
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.MaxPooling2D(2)(x) # 12 x 12
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.MaxPooling2D(2)(x) # 6 x 6
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)

    x = layers.MaxPooling2D(2)(x) # 3 x 3
    x = layers.GlobalAveragePooling2D()(x) # 1 x 1
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(7, activation='softmax')(x)

    return models.Model(inputs = input_tensor, outputs = x)
