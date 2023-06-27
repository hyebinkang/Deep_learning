import tensorflow as tf
def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if i == 0:          # 이전 stage에서 가져온 텐서의 dimension을 증가 시켜야 함
            x = tf.keras.layers.Conv2D(128, (1, 1), strides=(2,2))(x)  # 논문 3.3 1*1 convolutions 는 stride2
            x = tf.keras.layers.BatchNormalization()(x)                #배치
            x = tf.keras.layers.Activation('relu')(x)                  #

            x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(512, (1, 1))(x)  # input 의 차원이 1x1 convolution과 stride=2에 의해 바뀌게 된다.
            shortcut = tf.keras.layers.Conv2D(512, (1, 1), strides=(2,2))(shortcut)  # 그러므로 skip connection도 차원 변화를 처리해줘야 한다.
            x = tf.keras.layers.BatchNormalization()(x)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

            x = tf.keras.layers.Add()([x, shortcut])   #residual connection
            x = tf.keras.layers.Activation('relu')(x)

            shortcut = x
        else:
            x = tf.keras.layers.Conv2D(128, (1, 1))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(512, (1, 1))(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)

            shortcut = x

    return x