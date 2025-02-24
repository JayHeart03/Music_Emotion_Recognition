import pandas as pd
import numpy as np
import torch
from llm2vec import LLM2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, concatenate
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.optimizers import Adam

# **下载 NLTK 资源**
nltk.download('punkt')
nltk.download('stopwords')

# **数据集路径**
dataset_path = "MoodyLyrics4Q.csv"

# **1. 加载数据**
data = pd.read_csv(dataset_path)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据

# **2. 数据预处理**
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(cleaned)

data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# **3. 处理类别标签**
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['mood'])
labels = to_categorical(encoded_labels)

# **4. 使用 LLM2Vec 获取嵌入**
device = "cuda" if torch.cuda.is_available() else "cpu"

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map=device,
    torch_dtype=torch.bfloat16,
)

def get_llm2vec_embedding(text):
    return l2v.encode(text)

# **计算所有歌词的 LLM2Vec 嵌入**
data['llm2vec_embedding'] = data['processed_lyrics'].apply(get_llm2vec_embedding)
X_llm2vec = np.vstack(data['llm2vec_embedding'].values)  # 形状: (num_samples, embedding_dim)
embedding_dim = X_llm2vec.shape[1]  # 获取 LLM2Vec 维度

# **5. 处理文本数据以供 CNN 使用**
max_sequence_length = 250  # 设定最大长度
tokenized_lyrics = [text.split()[:max_sequence_length] for text in data['processed_lyrics']]  # 只保留前 max_sequence_length 词
word_index = {word: idx+1 for idx, word in enumerate(set(" ".join(data['processed_lyrics']).split()))}

X_cnn = np.zeros((len(tokenized_lyrics), max_sequence_length), dtype=int)
for i, sentence in enumerate(tokenized_lyrics):
    for j, word in enumerate(sentence):
        if word in word_index:
            X_cnn[i, j] = word_index[word]

# **6. 划分训练集**
x_train_cnn, x_val_cnn, y_train, y_val = train_test_split(X_cnn, labels, test_size=0.2, random_state=42, stratify=labels)
x_train_llm2vec, x_val_llm2vec = train_test_split(X_llm2vec, test_size=0.2, random_state=42, stratify=labels)

print(f"训练数据 CNN 形状: {x_train_cnn.shape}")
print(f"训练数据 LLM2Vec 形状: {x_train_llm2vec.shape}")

# **7. 定义 CNN + LLM2Vec 模型**
# **CNN 处理文本输入**
text_input = Input(shape=(max_sequence_length,), dtype='int32', name='Text_Input')
embedding_layer = Embedding(input_dim=len(word_index) + 1,
                            output_dim=300,
                            input_length=max_sequence_length,
                            trainable=True)(text_input)

conv1 = Conv1D(128, 5, activation='relu', name='Conv1')(embedding_layer)
pool1 = MaxPooling1D(5, name='Maxpool1')(conv1)
conv2 = Conv1D(64, 5, activation='relu', name='Conv2')(pool1)
pool2 = MaxPooling1D(5, name='Maxpool2')(conv2)
conv3 = Conv1D(32, 5, activation='relu', name='Conv3')(pool2)
pool3 = GlobalMaxPooling1D(name='GlobalMaxpool')(conv3)

# **LLM2Vec 输入**
llm2vec_input = Input(shape=(embedding_dim,), name='LLM2Vec_Input')
llm2vec_dense = Dense(128, activation='relu', name='LLM2Vec_Dense')(llm2vec_input)
llm2vec_dropout = Dropout(0.2, name='LLM2Vec_Dropout')(llm2vec_dense)

# **合并 CNN + LLM2Vec**
merged = concatenate([pool3, llm2vec_dropout], name='Merged')
dense1 = Dense(64, activation='relu', name='Dense1')(merged)
dropout1 = Dropout(0.2, name='Dropout1')(dense1)
output_layer = Dense(len(labels[0]), activation='softmax', name='Output')(dropout1)

# **定义模型**
model = Model(inputs=[text_input, llm2vec_input], outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# **查看模型结构**
model.summary()

# **8. 训练模型**
history = model.fit([x_train_cnn, x_train_llm2vec], y_train,
                    batch_size=16,
                    epochs=20,
                    validation_data=([x_val_cnn, x_val_llm2vec], y_val),
                    verbose=1)

# **9. 评估模型**
y_pred = model.predict([x_val_cnn, x_val_llm2vec])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))
print(f'F1 Score: {f1_score(y_true, y_pred_classes, average="weighted"):.2f}')

# **绘制损失和准确率曲线**
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# **绘制混淆矩阵**
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# model.save('model_cnn_llm2vec.h5')

# import pickle
# with open('tokenizer.pkl', 'wb') as f:
#     pickle.dump(word_index, f)

# with open('label_encoder.pkl', 'wb') as f:
#     pickle.dump(label_encoder, f)
