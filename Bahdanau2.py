import warnings
warnings.simplefilter('ignore')
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time


# ### 데이터셋 다운로드하고 준비하기
# 출처 사이트 : http://www.manythings.org/anki/

# In[2]:


# 알집파일을 다운로드합니다.
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# 유니코드 파일을 아스키(ascii)로 변환합니다.
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # 단어와 단어 뒤에 있는 문장 부호 사이에 공백을 만듭니다.
    # 예: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # (a-z, A-Z, ".", "?", "!", ",")을 제외한 모든 문자를 공백으로 바꿉니다.
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # 모델이 예측을 언제 시작하고 끝낼지 알게 하기 위해서 start와 end 토큰을 삽입합니다.
    w = '<start> ' + w + ' <end>'
    return w


en_sentence = u"May I borrow a pen?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
ko_sentence = u"내가 이책을 빌려도 될까요?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8')) # 한국어는 번역이 안되서 스페인을 그대로 사용함
print(preprocess_sentence(ko_sentence).encode('utf-8')) # 아스키코드로 변환하면 한국어는 못읽어 오는것이라고 판단하였습니다.
print(preprocess_sentence(ko_sentence).encode('cp949'))
print(preprocess_sentence(ko_sentence).encode('Euc-Kr'))
print(preprocess_sentence(ko_sentence).encode('Ansi'))


# 1. 억양을 제거합니다.
# 2. 문장을 정리합니다.
# 3. [영어, 스페인어] 형식의 단어 쌍을 반환합니다.
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)


en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])


# 최대길이를 텐서에 담는 함수
def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer



# 언어 데이터셋의 크기를 실험해봅니다.
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# 타겟 텐서의 최대 길이를 계산합니다.
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)



# 학습 데이터셋과 검증 데이터셋을 데이터셋의 80대 20으로 나눕니다.
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# 길이 확인하기
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


# 전환하는 함수
def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))


print ("입력 언어; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("타겟 언어; index to word mapping")
convert(targ_lang, target_tensor_train[0])


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# In[15]:


example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


# ## 인코더 모델과 디코더 모델 쓰기

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz                                                    #배치 사이즈
        self.enc_units = enc_units                                                  #encoder unit
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,return_state=True,     #output state와 hidden state를 이용(검색) 가능하게 한다.(time step별 hidden state를 출력)
                                       recurrent_initializer='glorot_uniform')      #glorot_uniform: 균등분포 방식, 파라미터 초기값 생성

    def call(self, x, hidden):
        x = self.embedding(x)                                                       # 임베딩 층 통과(차원 증가)
        output, state = self.gru(x, initial_state = hidden)                         # GRU의 hidden state을 활용함, 셀의 첫번째 호출에 전달될 초기 상태 텐서 목록
        
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))                            # 배치와 인코더 유닛을 0벡터로


# 인코더는 (시퀀스 길이, 단어집합크기) 2D텐서로 입력을 받아 (샘플의 수, 입력크기, 임베딩 차원) 인 3D 텐서를 반환한다.
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# 샘플 입력
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # 이를 통해 점수를 계산합니다.

        hidden_with_time_axis = tf.expand_dims(query, 1)                                            #query 차원 확장

        # score shape == (batch_size, max_length, 1)
        # 스코어에 self.V를 사용하기 때문에 마지막 축에 1이 들어갑니다.
        # self.V를 사용하기 전의 텐서의 형태는 (batch_size, max_length, units)입니다.


        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))                #alignment model 점수 계산

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)                                            #softmax에 사용, 가중치 계산


        # 합쳐서 나온 context_vector shape == (batch_size, hidden_size)
        context_vector = attention_weights * values                                                 #가중치*value = context value

        context_vector = tf.reduce_sum(context_vector, axis=1)                                      #가중치의 합(새로운 단어 예측)

        return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz                                                                    #배치 사이즈
        self.dec_units = dec_units                                                                  #decoder unit
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True, return_state=True,                    #output state와 hidden state를 이용(검색) 가능하게 한다.(time step 별 hidden state를 출력)-> Attention
                                       recurrent_initializer='glorot_uniform')                      #glorot_uniform: 균등 분포
        self.fc = tf.keras.layers.Dense(vocab_size)                                                 #fully connected layer

        # 바다나우어텐션 적용하기
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):                                                          # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)                                                                       # 임베딩을 후 shape == (batch_size, 1, embedding_dim)
        print(x.shape, "어쭈구리")

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)                              # concat 이후 shape (batch_size, 1, embedding_dim + hidden_size)
        print(x.shape, "켓쭈구리")

        output, state = self.gru(x)                                                                 #연결벡터를 GRU로 보내기
        print(output.shape)

        output = tf.reshape(output, (-1, output.shape[2]))                                          # output shape == (batch_size * 1, hidden_size)
        print(output.shape, "이놈봐라")

        x = self.fc(output)                                                                         # output shape == (batch_size, vocab)

        return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # 티쳐 포싱: 타겟을 다음 입력값으로 넘겨줍니다.
        for t in range(1, targ.shape[1]):
            # 인코더의 출력값을 디코더로 전달합니다.
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # 티쳐 포싱을 사용합니다.
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # 2 에포크마다 모델을 저장합니다(체크포인트)
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 예측된 ID는 모델에 다시 입력됩니다
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# 어텐션 가중치를 그리기 위한 함수입니다.
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


# checkpoint_dir에 있는 가장 최근의 체크포인트를 복원합니다.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



translate(u'hace mucho frio aqui.')


translate(u'esta es mi vida.')

