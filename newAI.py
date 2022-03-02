import numpy as np
import tflearn
from tensorflow.python.framework import ops
import random
import json
import underthesea as vi
import pickle

# XỬ LÝ SƠ LƯỢC DỮ LIỆU TRAIN
file = open('intents.json', encoding="utf8")
with file as json_data:
    intents = json.load(json_data) # Gán dữ liệu trong file json vào biến intents

# Lấy ra các stop-word trong stop-word.txt
text_file = open("stop-word.txt", "r", encoding="utf8")
lines = text_file.read().splitlines() 
text_file.close()
stop_words = lines

words = []
classes = []
documents = []
context = []
for intent in intents['intents']:
    for content in intent['content']:
        for pattern in content['patterns']:
            w = vi.word_tokenize(pattern) # Tách câu thành các từ có nghĩa VD: 'mấy', 'thích', 'đánh lộn',...
            words.extend(w) # Đưa các từ đã tách vào mảng word
            documents.append((w, intent['tag'], content['context'])) # Đưa các từ đã tách + tag của câu chứa từ đó vào documents
            if content['context'] not in context:
                context.append(content['context'])
    if intent['tag'] not in classes:
        classes.append(intent['tag']) # Thêm các tag mới

print(words)
words = [w.lower() for w in words if w not in stop_words] # Loại bỏ stop words, trách gây nhiễu và độ chính xác cao hơn
words = sorted(list(set(words))) # Sắp xếp và loại bỏ các từ bị trùng
classes = sorted(list(set(classes))) # Sắp xếp và loại bỏ các tag
context = sorted(list(set(context))) # Sắp xếp và loại bỏ các tag

# TẠO DỮ LIỆU TRAIN
training = []
output = []
output_empty = [0] * len(classes) # Tạo output_empty có độ dài bằng độ dài của classes
output_empty1 = [0] * len(context) # Tạo output_empty có độ dài bằng độ dài của classes

# Vector hóa dữ liệu train
for doc in documents:
    bag = []
    pattern_words = doc[0] # Lấy ra từng câu request trong file train
    pattern_words = [word.lower() for word in pattern_words] # Lower tất cả các từ

    # Vector hóa từ (One hot vector)
    for w in words:
         # Từ có xuất hiện trong từ điển được gán là 1 ngay tại vị trí của từ đó và ngược lại
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty) # Khởi tạo chiều dài của output_row (Độ dài của output_empty)
    output_row[classes.index(doc[1])] = 1 # Tag hiện tại được đánh dấu là 1, các tag còn lại là 0
    output_row1 = list(output_empty1) # Khởi tạo chiều dài của output_row (Độ dài của output_empty)
    output_row1[context.index(doc[2])] = 1 # Tag hiện tại được đánh dấu là 1, các tag còn lại là 0
    training.append([bag, output_row, output_row1]) # Thêm vào vector và vector tag của câu hiện tại
    
random.shuffle(training) # Xáo trộn dữ liệu train
training = np.array(training)

train_x = list(training[:,0]) # Các vector của câu
train_y = list(training[:,1]) # Các vector của tag
train_z = list(training[:,2]) # Các vector của tag

# BUILD NEURAL NETWORK
ops.reset_default_graph()
net1 = tflearn.input_data(shape=[None, len(train_x[0])]) # Khởi tạo lớp đầu vào [INFINITY, len(train_x[0])]
net2 = tflearn.fully_connected(net1, 8) # Lớp ẩn
net5 = tflearn.fully_connected(net2, len(train_y[0]), activation='softmax') # Lớp đầu ra
net6 = tflearn.regression(net5, optimizer='adam', loss='categorical_crossentropy') # Lớp ước tính hồi quy

# Xác định mô hình (Deep Neural Network)
model = tflearn.DNN(net6)
# Bắt đầu train
model.fit(train_x, train_y, n_epoch=1000, batch_size=64, show_metric=True)
# model.save('./data-train/model.tflearn')

# BUILD NEURAL NETWORK
ops.reset_default_graph()
nett1 = tflearn.input_data(shape=[None, len(train_x[0])]) # Khởi tạo lớp đầu vào [INFINITY, len(train_x[0])]
nett2 = tflearn.fully_connected(nett1, 8) # Lớp ẩn
nett5 = tflearn.fully_connected(nett2, len(train_z[0]), activation='softmax') # Lớp đầu ra
nett6 = tflearn.regression(nett5, optimizer='adam', loss='categorical_crossentropy') # Lớp ước tính hồi quy

# Xác định mô hình (Deep Neural Network)
modell = tflearn.DNN(nett6)
# Bắt đầu train
modell.fit(train_x, train_z, n_epoch=1000, batch_size=64, show_metric=True)
# model.save('./data-train/model.tflearn')
# pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "./data-train/training_data", "wb" ) )
# # restore our data structures
# data = pickle.load( open( "./data-train/training_data", "rb" ) )
# words = data['words']
# classes = data['classes']
# train_x = data['train_x']
# train_y = data['train_y']
# XỬ LÝ TIỀN DỮ LIỆU ĐẦU VÀO (REQUEST)
def clean_up_sentence(sentence):
    sentence_words = vi.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

# Vector hóa dữ liệu đầu vào (Tương tự phía trên)
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

# model.load('./data-train/model.tflearn')
# XỬ LÝ ĐẦU RA
ERROR_THRESHOLD = 0.5 # Dropout
def response(sentence):
    results = model.predict([bow(sentence, words)])[0] # Dự đoán dữ liệu đầu vào
    results = [[i,r] for i,r in enumerate(results)]
    # print(list(map(lambda x: x[1], results)))
    results.sort(key=lambda x: x[1], reverse=True)
    
    results1 = modell.predict([bow(sentence, words)])[0] # Dự đoán dữ liệu đầu vào
    results1 = [[i,r] for i,r in enumerate(results1)]
    # print(list(map(lambda x: x[1], results)))
    results1.sort(key=lambda x: x[1], reverse=True)
    print(results)
    print(results1)
    for i in intents['intents']:
        if i['tag'] == classes[results[0][0]]:
            for content in i['content']:
                if content['context'] == context[results1[0][0]]:
                    return random.choice(content['response'])
                # else:
                #     return random.choice(content['response'])

print(response('lộn'))