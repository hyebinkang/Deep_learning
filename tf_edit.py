import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):     # (100,1) (1,10), 10
      angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
      return position * angle_rates #(100,1)

    # tf.newaxis는 별도의 포스팅을 해두었습니다. 그부분을 참고하시면 됩니다.
    # shape 맞지 않는 행렬을 연산할 때 사용합니다.
    def positional_encoding(self, position, d_model): #100 10

      angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],  # 1차원(리스트) -> 2차원 (100,1)
                              np.arange(d_model)[np.newaxis, :],        # 1차원(리스트) -> 2차원 (1,10)
                              d_model)                                  # 10

      angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])         #sin 적용 -> 짝수

      angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])         #cos 적용-> 홀수

      pos_encoding = angle_rads[np.newaxis, ...]                # embedding 값과 더해주어야하기 때문에 다시 리스트형태(1차원)로 전환하기


      return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# 문장의 길이 50, 임베딩 벡터의 차원 128
sample_pos_encoding = PositionalEncoding(50, 128)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()

def scaled_dot_product_attention(query, key, value, mask):
  # 우리는 배치단위로 나누어서 학습을 시킬 것이기 때문에 batch_size가 맨앞에 들어오게 됩니다. 
  # 일반적으로 맨 앞쪽에 적어준다고 하는군요 
  # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
  # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
  # padding_mask : (batch_size, 1, 1, key의 문장 길이)

  # Q와 K의 곱. 어텐션 스코어 행렬.
  # tf에서는 transpose를 이렇게 우하한 방법으로 하더군요. 
  # transpose_a는 query를 transpose_b는 key를 가르킵니다.
    matmul_qk = tf.matmul(query, key, transpose_b=True)         #query,key곱, attention score


  # 여기서 d는 d_model/num_heads를 의미합니다. 
    depth = tf.cast(tf.shape(key)[-1], tf.float32)                #d_k 루트 값 == (key)[-1]
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)                                     #masking, -1e9(아주 작은 음수값-> softmax를 지난 후 0이 되는 값)

  # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
  # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logits, axis=-1)                  #softmax를 활용 (query와 key)

  # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)                        #(query,key를 구한것 + value)matmul을 통해 output출력

    return output, attention_weights

# np.set_printoptions(suppress=True)
# temp_k = tf.constant([[10,0,0],
#                       [0,10,0],
#                       [0,0,10],
#                       [0,0,10]], dtype=tf.float32)  # (4, 3)
#
# temp_v = tf.constant([[   1,0],
#                       [  10,0],
#                       [ 100,5],
#                       [1000,6]], dtype=tf.float32)  # (4, 2)
# temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
#
# temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
# print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
# print(temp_out) # 어텐션 값

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads                      # d_model을 num_heads로 나눈 값(논문: 64)

    # WQ, WK, WV에 해당하는 밀집층 정의
    # 1. WQ, WK, WV에 해당하는 밀집층 지나기
    # q : (batch_size, query의 문장 길이, d_model)
    # k : (batch_size, key의 문장 길이, d_model)
    # v : (batch_size, value의 문장 길이, d_model)
    # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
        self.query = tf.keras.layers.Dense(units=d_model)                   #DenseLayer 지나기
        self.key = tf.keras.layers.Dense(units=d_model)
        self.value = tf.keras.layers.Dense(units=d_model)

    # Wo에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units=d_model)

      # num_heads 개수만큼 q, k, v를 split하는 함수
  # 여기서 perm을 통해 순서를 변경하는 이유는 scaled dot-product Attention 연산시 
  # scaled_dot_product_attention func이 데이터를 입력받는 순서가  batch_szie, num_heads, query문장길이, key문장길이 순서 이기 때문입니다. 
    def split_heads(self, inputs, batch_size):                                              # num_heads 개수만큼 q, k, v를 split
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

    # 2. 헤드 나누기
    # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)                                 #head 수 만큼 나누기
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

    # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
    # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
    # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # 4. 헤드 연결(concatenate)하기 : (batch_size, query의 문장 길이, d_model)
    # (batch_size, query의 문장 길이, num_heads, d_model/num_heads) -> (batch_size, query의 문장 길이, d_model) 만드는 과정 
    # 원래 d_model을 num_heads로 나누어서 d_model/num_heads를 구했으니 역으로 합치는 과정입니다. 
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))

    # 5. WO에 해당하는 밀집층 지나기
    # concat을 하고 나면 나오는 shape가 (seq_len, d_model)인데, 우리가 필요한 shape는 (d_model, d_model)이기에 Wo를 곱해준다. 
    # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)                 #x와 0이 같으면 True(1)반환, 다르면 False(0)반환

    return mask[:, tf.newaxis, tf.newaxis, :]                       # (batch_size, 1, 1, key의 문장 길이)

# print(create_padding_mask(tf.constant([[1, 21, 777, 0, 0]])))


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")   # 인코더는 패딩 마스크 사용
 
    attention = MultiHeadAttention(                                          # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs,                   # Q = K = V
          'mask': padding_mask                                               # 패딩 마스크 사용
      })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)            # 드롭아웃(오버피팅 방지) + 잔차 연결과 층 정규화
    attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)                                     #epslion: 분모가 0이 되는것을 방지하는 파라미터

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)    # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)                # 드롭아웃 + residual connection과 층 정규화
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):

    inputs = tf.keras.Input(shape=(None,), name="inputs")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")    # 인코더는 패딩 마스크 사용

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)        # 들어오는 vocab_size와 d_model만큼 Embedding                           
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)          # Positionalencoding+dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):                                                # 인코더를 num_layers개 쌓기
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

#디코더의 첫번째 서브층에서 masked 함수, 마스킹 위치에 1, 하지 않는 위치는 0
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]  # (1,seq_len,1,none) != (1,5)
    print(seq_len,"dd")
    look_ahead_mask = 1-tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)           #삼각 대각행렬(tensor, -1, 0): 하삼각행렬??

    padding_mask = create_padding_mask(x)                                                   # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)

print(create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]])))

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(                                          # 첫번째 서브층
      shape=(1, None, None), name="look_ahead_mask")

    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')    # 패딩 마스크(두번째 서브층)

    attention1 = MultiHeadAttention(                                          # 첫번째 서브층 / 마스크드 셀프 어텐션
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask  # 룩어헤드 마스크
      })

    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)      # layerNormalization

    attention2 = MultiHeadAttention(                                          # 두번째 서브층 / 디코더-인코더 어텐션
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs,      # Q != K = V
          'mask': padding_mask})                                                # 패딩 마스크


    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)              # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)   # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)                    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],outputs=outputs, name=name)


def decoder(vocab_size, num_layers, dff,d_model, num_heads, dropout,name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask = tf.keras.Input(                                                # look_ahead_mask + padding mask 둘 다 사용
      shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)              # 포지셔널 인코딩 + 드롭아웃
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):                                                     # 디코더를 num_layers개 쌓기
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def transformer(vocab_size, num_layers, dff,d_model, num_heads, dropout,name="transformer"):

    inputs = tf.keras.Input(shape=(None,), name="inputs")                  # 인코더의 입력

    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")          # 디코더의 입력

    enc_padding_mask = tf.keras.layers.Lambda(       # (1, none)           # 인코더의 패딩 마스크
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)           # (1,1,1, none)

    look_ahead_mask = tf.keras.layers.Lambda(                               # look_ahead_mask
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(                              # Decoder Padding mask
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,       # encoder output-> decoder로 전달
      d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])                                               # padding mask, input sequence encoder입력값

    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,      # decoder 출력
      d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)    # 다음 단어 예측을 위한 output

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

small_transformer = transformer(
    vocab_size = 9000,
    num_layers = 4,
    dff = 512,
    d_model = 128,
    num_heads = 4,
    dropout = 0.3,
    name="small_transformer")

# tf.keras.utils.plot_model(
#     small_transformer, to_file='small_transformer.png', show_shapes=True)

# def loss_function(y_true, y_pred):
#   y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
#
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(
#       from_logits=True, reduction='none')(y_true, y_pred)
#
#   mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#   loss = tf.multiply(loss, mask)
#
#   return tf.reduce_mean(loss)