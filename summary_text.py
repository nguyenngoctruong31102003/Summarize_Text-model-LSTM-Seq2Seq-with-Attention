import os
import pickle
import re
import string
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from pyvi import ViTokenizer
from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, concatenate, TimeDistributed, dot, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# Class TextSummarizer làm các việc tiền xử lý, load dữ liệu
class TextSummarizer:
    def __init__(self, w2v_path, stopwords_path):
        self.w2v = KeyedVectors.load_word2vec_format(w2v_path)
        self.vocab = self.w2v.key_to_index
        self.stop_words = self.load_stopwords(stopwords_path)
    
    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())
        return stop_words

    # Đọc và chuyển đổi mã hóa của văn bản
    def read_and_convert(self, file_path):
        with open(file_path, 'r', encoding='utf-16') as file:
            content = file.read()
        return content.encode('utf-8').decode('utf-8')
    
    # Load Data
    def load_data(self, data_dir):
        data = []
        labels = []
        classes = 4  # lấy dữ liệu 3 folder train
        
        # Lấy danh sách các thư mục con trong 'data_dir'
        subdirs = sorted(os.listdir(data_dir))[:classes]
        
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(subdir_path, filename)
                        data.append(self.read_and_convert(file_path))
                        labels.append(i)
        return data, labels

    # Hàm tiền xử lý dữ liệu
    def preprocess_text(self, text):
        # Loại bỏ khoảng trắng đầu dòng và các ký tự xuống dòng
        text = text.strip().replace('\n', '. ')
        original_sentences = nltk.sent_tokenize(text)
        
        processed_sentences = []
        for sentence in original_sentences:
            words = word_tokenize(sentence)
            words = [word.lower() for word in words if word.lower() 
                     not in self.stop_words and word not in string.punctuation]
            processed_sentence = ' '.join(words)
            processed_sentences.append(processed_sentence)
        return original_sentences, processed_sentences
    
    # Hàm chọn lọc câu quan trọng
    def select_important_sentences(self, text, num_sentences=5):
        # Tiền xử lý dữ liệu
        original_sentences, processed_sentences = self.preprocess_text(text)
        
        # Tính điểm TF-IDF (Term Frequency-Inverse Document Frequency)
        # Đánh giá mức độ quan trọng của các câu dựa trên từ ngữ xuất hiện trong câu đó so với toàn bộ văn bản.
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        # Tính điểm quan trọng của từng câu bằng cách lấy tổng TF-IDF của các từ trong câu
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Chọn ra các câu có điểm cao nhất
        top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_sentences = [original_sentences[i] for i in sorted(top_sentence_indices)]
        
        return ' '.join(top_sentences)
    
    # Hàm tóm tắt văn bản
    def summarize_text(self, original_text, model_path, tokenizer_path, num_sentences):
        # Chọn lọc câu quan trọng với văn bản gốc và số lượng câu mong muốn
        important_text = self.select_important_sentences(original_text, num_sentences=num_sentences)
        
        # Load mô hình
        # tokenizer để chuyển đổi văn bản thành các số nguyên.
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Tiền xử lý dữ liệu
        sequences = tokenizer.texts_to_sequences([important_text])
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        
        # Tạo tóm tắt từ mô hình Seq2Seq với Attention
        predicted = model.predict([padded_sequences, padded_sequences])
        predicted_sequence = np.argmax(predicted[0], axis=-1)
        summary = ' '.join([tokenizer.index_word.get(i, '') for i in predicted_sequence if i != 0])
        
        # Hậu xử lý để cải thiện ngắt câu và logic
        summary = re.sub(r'\s+', ' ', summary)  # Xóa khoảng trắng dư thừa
        summary = re.sub(r'\.\s*\.', '.', summary)  # Xóa dấu chấm đôi
        summary = re.sub(r'\s*,\s*', ', ', summary)  # Chuẩn hóa dấu phẩy
        summary = re.sub(r'\s*\.\s*', '. ', summary)  # Chuẩn hóa dấu chấm
        summary = summary.strip()
        
        # Chuyển đổi câu thành chữ hoa đầu câu
        summary_sentences = nltk.sent_tokenize(summary)
        summary_sentences = [s.capitalize() for s in summary_sentences]
        summary = ' '.join(summary_sentences)
        
        # Đảm Bảo Ý Nghĩa Chính Của Văn Bản Gốc
        final_summary = []
        for original_sentence in nltk.sent_tokenize(important_text):
            if any(word in summary for word in original_sentence.split()):
                final_summary.append(original_sentence)
        summary = ' '.join(final_summary)
        
        return summary


# Class build_Model sẽ dùng cho việc xây dựng mô hình LSTM Seq2Seq với cơ chế Attention và train mô hình
class build_Model(TextSummarizer):
    def __init__(self, w2v_path, stopwords_path, data_dir):
        super().__init__(w2v_path, stopwords_path)  # Gọi hàm khởi tạo của TextSummarizer
        
        # Tải và chuyển đổi dữ liệu
        self.data, self.labels = self.load_data(data_dir)

        # Chuyển đổi danh sách tài liệu thành mảng numpy
        self.data_array = np.array(self.data)
        self.labels_array = np.array(self.labels)
        print('\n======================================')
        print(f'Tổng số tài liệu: {self.data_array.shape}')
        print(f'Tổng số nhãn: {self.labels_array.shape}')

        # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_array, self.labels_array, test_size=0.2, random_state=42)

        # In kích thước của các tập dữ liệu
        print(f'Kích thước tập huấn luyện: {self.X_train.shape}')
        print(f'Kích thước tập kiểm tra: {self.X_test.shape}')
        print('======================================\n')


        # Chuẩn bị dữ liệu cho mô hình Seq2Seq với Attention
        self.tokenizer = Tokenizer(num_words = 7000)
        self.tokenizer.fit_on_texts(self.X_train)
        self.X_train_sequences = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test_sequences = self.tokenizer.texts_to_sequences(self.X_test)

        self.max_length = 100
        self.X_train_padded = pad_sequences(self.X_train_sequences, 
                                            maxlen=self.max_length, padding='post', truncating='post')
        self.X_test_padded = pad_sequences(self.X_test_sequences,
                                            maxlen=self.max_length, padding='post', truncating='post')

    # Xây dựng mô hình LSTM Seq2Seq with Attention
    def build_model_LSTM_Seq2Seq(self):
        # Các tham số cho mô hình Seq2Seq với Attention
        vocab_size = len(self.tokenizer.word_index) + 1
        embedding_dim = 200
        latent_dim = 64

        # Encoder
        # Mã hóa chuỗi đầu vào thành các trạng thái ẩn
        encoder_inputs = Input(shape=(self.max_length,))

        # Embedding layer with L2 regularization
        enc_emb = Embedding(vocab_size, embedding_dim, trainable=True, embeddings_regularizer=l2(1e-4))(encoder_inputs)

        # Encoder lstm 1 layers with dropout and L2 regularization
        encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, 
                             dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(1e-4))
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        # Encoder lstm 2
        encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, 
                             dropout=0.3,recurrent_dropout=0.2, kernel_regularizer=l2(1e-4))
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        # Encoder lstm 3
        encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True, 
                           dropout=0.3,recurrent_dropout=0.2, kernel_regularizer=l2(1e-4))
        encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        # Giải mã chuỗi đầu ra từ các trạng thái ẩn của encoder.
        decoder_inputs = Input(shape=(None,))

        dec_emb_layer = Embedding(vocab_size, embedding_dim, trainable=True , embeddings_regularizer=l2(1e-4))
        dec_emb = dec_emb_layer(decoder_inputs)
        
        # decoder lstm layers with dropout and L2 regularization
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, 
                            dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(1e-4))
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

        # Attention Mechanism
        # Tính toán trọng số attention để tập trung vào các phần quan trọng của chuỗi đầu vào.
        attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
        attention = Activation('softmax')(attention)    # Chuyển đổi thành xác suất
        context = dot([attention, encoder_outputs], axes=[2, 1])
        # Concatenate context vector and decoder LSTM output
        decoder_combined_context = concatenate([context, decoder_outputs])
        
        # TimeDistributed layer with L2 regularization
        output = TimeDistributed(Dense(latent_dim, activation="relu", kernel_regularizer=l2(1e-4)))(decoder_combined_context)
        decoder_outputs = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(1e-4))(output)

        # Compile the model using Adam optimizer
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

    # Huấn Luyện Mô Hình
    def train_model(self, model_path, tokenizer_path):                     
        model = self.build_model_LSTM_Seq2Seq()
        
        es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
        
        history = model.fit(
            [self.X_train_padded, self.X_train_padded],
            np.expand_dims(self.X_train_padded, -1),
            epochs=20,
            batch_size=64,
            callbacks=[es, reduce_lr],
            validation_data=([self.X_test_padded, self.X_test_padded], 
                             np.expand_dims(self.X_test_padded, -1))
        )
        -+9
        # Lưu mô hình và tokenizer sau khi huấn luyện
        model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        #plotting graphs for accuracy 
        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        print('\n===========================================')
        print("Training completed and model is saved.")
        print('===========================================\n')
        
        return history