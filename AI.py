# -*- coding: utf-8 -*-
import speech_recognition #Thư viện nhận dạng giọng nói
from gtts import gTTS # Giúp trợ lý ảo nói (Thông qua file mp3)
import os # Thư viện hỗ trợ thao tác với file
from datetime import date, datetime # Hỗ trợ lấy ra ngày giờ
import playsound # Thư viện hỗ trợ chạy file mp3 đã lưu
import pyjokes # Thư viện giúp trợ lý ảo nói đùa
from googletrans import Translator # Thư viện hỗ trợ dịch văn bản
import re # Thư viện Regular Expression
import webbrowser # Thư viện hỗ trợ mở trình duyệt
from youtube_search import YoutubeSearch # Thư viện hỗ trợ tìm kiếm trên youtube
import time # Xử lý các tác vụ liên quan đến thời gian
import requests # Hỗ trợ gửi yêu cầu HTTP
from bs4 import BeautifulSoup # Hỗ trợ xử lý dữ liệu dạng html (Phân tích tài liệu html)
from underthesea import sent_tokenize
# Khởi tạo
ai_brain = " " # Chuỗi rỗng do ban đầu máy chưa được học gì
count = 0 # Count hỗ trợ người dùng không nói nhiều lần sẽ tự tắt
you = " " # Người nói
name_sir = "" # Tên của người nói
translator = Translator() # Get phương thức translator

import numpy as np
import tflearn
from tensorflow.python.framework import ops
import random
import json
import underthesea as vi

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
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = vi.word_tokenize(pattern) # Tách câu thành các từ có nghĩa VD: 'mấy', 'thích', 'đánh lộn',...
        words.extend(w) # Đưa các từ đã tách vào mảng word
        documents.append((w, intent['tag'])) # Đưa các từ đã tách + tag của câu chứa từ đó vào documents
        if intent['tag'] not in classes:
            classes.append(intent['tag']) # Thêm các tag mới

words = [w.lower() for w in words if w not in stop_words] # Loại bỏ stop words, trách gây nhiễu và độ chính xác cao hơn
words = sorted(list(set(words))) # Sắp xếp và loại bỏ các từ bị trùng
classes = sorted(list(set(classes))) # Sắp xếp và loại bỏ các tag

# TẠO DỮ LIỆU TRAIN
training = []
output = []
output_empty = [0] * len(classes) # Tạo output_empty có độ dài bằng độ dài của classes

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
    training.append([bag, output_row]) # Thêm vào vector và vector tag của câu hiện tại
    
random.shuffle(training) # Xáo trộn dữ liệu train
training = np.array(training)

train_x = list(training[:,0]) # Các vector của câu
train_y = list(training[:,1]) # Các vector của tag

# BUILD NEURAL NETWORK
ops.reset_default_graph()
net1 = tflearn.input_data(shape=[None, len(train_x[0])]) # Khởi tạo lớp đầu vào [INFINITY, len(train_x[0])]
net2 = tflearn.fully_connected(net1, 64,bias_init='zeros', bias=False) # Lớp ẩn
net3 = tflearn.fully_connected(net2, 64,bias_init='zeros', bias=False) # Lớp ẩn
net4 = tflearn.fully_connected(net3, 64,bias_init='zeros', bias=False) # Lớp ẩn
net5 = tflearn.fully_connected(net4, 64,bias_init='zeros', bias=False) # Lớp ẩn
net6 = tflearn.fully_connected(net5, len(train_y[0]), activation='softmax',bias_init='zeros', bias=False) # Lớp đầu ra
net7 = tflearn.regression(net6, optimizer='adam', loss='categorical_crossentropy') # Lớp ước tính hồi quy

# Xác định mô hình (Deep Neural Network)
model = tflearn.DNN(net7)
# Bắt đầu train
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

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

# XỬ LÝ ĐẦU RA
ERROR_THRESHOLD = 0.5 # Dropout
def response(sentence):
    results = model.predict([bow(sentence, words)])[0] # Dự đoán dữ liệu đầu vào
    results = [[i,r] for i,r in enumerate(results)]
    # print(list(map(lambda x: x[1], results)))
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]]

# Phương thức giúp trợ lí ảo nói
def ai_speak(text):
    print("Trợ lí: " + text + "\n") # In ra những gì trợ lý sẽ nói
    ai_mouth = gTTS(text, lang = "vi") # Gán cho biến ai_mouth để trợ lí có thể nói tiếng việt
    fileName = "AI.mp3" # Tên file mà trợ lí dùng để nói
    ai_mouth.save(fileName) # Lưu file (Cùng cấp với file python hiện tại)
    playsound.playsound(fileName, True) # Mở file AI.mp3 đã lưu ở trên
    os.remove(fileName) # Sau khi mở phải xóa file để tránh xung đột

# Phương thức giúp trợ lý có thể nghe
def ai_listen():
    ai_ear = speech_recognition.Recognizer() # Giúp trợ lý có thể nghe người dùng nói

    with speech_recognition.Microphone() as micro: # Gán speech_recognition.Microphone() cho biến micro
        ai_ear.adjust_for_ambient_noise(micro) # Giảm tiếng ồn
        print("Trợ lí: Tôi đang nghe ...")
        audio = ai_ear.record(micro, duration= 5) # Sau 5s sẽ tự ngắt micro
        print("Trợ lí: ...")
    try:
        you = ai_ear.recognize_google(audio, language = "vi-VN")
        # you = ai_ear.recognize_google(audio, language = "vi-VN").lower() # Nghe và nói theo tiếng việt, đưa về chữ thường để dễ xử lý
        print("Người sử dụng: " + you)
        return you
    except: # Bắt lỗi khi người dùng không nói gì hoặc người dùng tạo những âm thành không phải từ ngữ
        you = ""
        return you

# Phương thức get thứ trong tuần
def weekToday(x):
        switcher={
                0: 'Thứ hai',
                1:'Thứ ba',
                2:'Thứ tư',
                3:'Thứ năm',
                4:'Thứ sáu',
                5:'Thứ bảy',
                6:'Chủ nhật'
             }
        return switcher.get(x)

# Phương thức lấy ra những gì trợ lý ảo nghe được
def get_ai_listen():
    text = ai_listen()
    if text:
        return text.lower()
    elif text == "":
        return ""

# Phương thức get tên người dùng
def get_name_sir(name):
    ai_speak("Bạn tên là gì?")
    name = get_ai_listen()
    name = name.title()
    ai_speak("Xin chào {}".format(name))
    return name

# Gán tên người dùng đã khai báo từ trước cho biến vừa get được
name_sir = get_name_sir(name_sir)

# Phương thức hỗ trợ mở browser
def open_brower(text):
    regex = re.search('mở (.+)', text) 
    if regex:
        domain = regex.group(1) # Lấy ra phần tử phía sau từ "mở" cho đến cuối của "text"
        url = 'https://www.' + domain
        webbrowser.open(url)
        ai_speak("Brower của bạn đã được mở")
        return True
    else:
        return False
        
# Phương thức hỗ trợ tìm kiếm trên google
def google_search(text):
    # search_key = text.split("kiếm", (1))[1]
    search_key = re.search('kiếm (.+)', text).group(1)
    url = f"https://www.google.com/search?q={search_key}"
    webbrowser.open(url) # Mở browser

# Phương thức hỗ trợ tìm kiếm trên youtube
def youtube_search():
    ai_speak("Nói từ khóa bạn muốn tìm kiếm trên Youtube!")
    text = get_ai_listen()
    results = YoutubeSearch(text, max_results=10).to_dict() # Đưa ra 10 link kết quả của tìm kiếm từ khóa 'text'
    url = 'https://www.youtube.com/' + results[0]['url_suffix'] # Lấy url là phần link đầu tiên
    # Lấy ra title của video chuẩn bị mở
    r = requests.get(url)
    s = BeautifulSoup(r.text, "html.parser")
    ai_speak("Đang mở: " + s.find('title').text)
    time.sleep(0.5)
    webbrowser.open(url)

# Phương thức dùng cmd tìm một file cụ thể trong ổ đĩa C 
def search_C(file_name):
    try:
        out = os.popen('dir /a-d /b "c:\\' + file_name +'" /s').read() # Chạy cmd trả về kết quả dưới dạng string
        temp = out.splitlines()
        ai_speak("Ứng dụng của bạn đang được mở!")
        os.startfile(temp[0]) # Mở file, vì chỉ có 1 file duy nhất nên là [0]
    except:
        ai_speak("Xin lỗi, tôi không tìm thấy File!")



def get_file_name(disk, file_name):
    number = ['một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười'] #Dùng cho trường hợp tìm kiếm được nhiều hơn 1 file
    count = 0; # Đếm số file
    a = ""
    out = os.popen('dir /a-d /b "' + disk + ':\\' + file_name + '*" /s').read().strip() # Chạy lệnh cmd
    temp = out.splitlines()
    for x in range(len(temp)): # Chạy vòng lặp for từ 0 -> chiều dài mảng temp
        file_name = temp[x].split('\\') 
        file_name = file_name[len(file_name) - 1] # Lấy ra phần tử cuối cũng của mảng file_name
        count += 1
        print(str(count) + ". " + file_name) # In ra danh sách các file tìm được
    if count > 1:
        ai_speak("Bạn muốn mở file thứ mấy?")
        time.sleep(1)
        text = get_ai_listen()
        for x in range(len(number)): # Chạy vòng for kiểm tra người dùng có nói đúng thứ tự hiện có hay không
            if number[x] in text:
                a = number.index(number[x])
        if a == "":
            ai_speak('Không thể mở file bạn yêu cầu!')
        elif a != "":
            ai_speak("File của bạn đang được mở")
            os.startfile(temp[a])
    elif count == 1:
        ai_speak("File của bạn đang được mở")
        os.startfile(temp[0])
    if len(temp)  == 0:
        ai_speak('File bạn muốn tìm kiếm không có trong ổ đĩa này, hãy thử lại ở ổ đĩa khác!')
        number_disk()

def number_disk():
    out = os.popen('wmic logicaldisk get name').read().strip() # Chạy cmd lấy ra danh sách ổ đĩa
    a = "".join(out.split()) # Xóa tất cả các khoảng trắng trong chuỗi
    output = a.split("Name")[1] 
    result = output.split(":")
    count = 1 # Đếm số ổ đĩa
    count2 = 0; # Dùng để kiểm tra người dùng có yêu cầu mở đúng ổ đĩa có trong danh sách hay không
    for x in range(len(result) - 2):
        count += 1
    ai_speak("Máy hiện tại có " + str(count) + " ổ đĩa ")
    for x in range(len(result) - 1):
        ai_speak(str(result[x]))
    ai_speak("Bạn muốn tìm kiếm ở ổ đĩa nào?")
    text = get_ai_listen()
    for x in range(len(result) - 1):
        if str(result[x]) not in text.upper():
            count2 += 1
    if count2 == len(result) - 1 and text != "":
        ai_speak('Xin lỗi, ổ đĩa bạn yêu cầu không tồn tại')
        number_disk()
    elif count2 != len(result) - 1:
        ai_speak("Bạn muốn tìm kiếm file tên gì?")
        text2 = get_ai_listen()
        if text2 != "":
            if "c" in text:
                get_file_name('c', text2)
            elif 'd' in text:
                get_file_name('d', text2)
            elif 'e' in text:
                get_file_name('e', text2)


# Phương thức mở dứng dụng
def open_application():
    ai_speak("Bạn muốn mở ứng dụng gì?")
    text = get_ai_listen()
    if "soạn thảo văn bản" in text:
        search_C("WINWORD.exe")
    elif "excel" in text:
        search_C("excel.exe")
    elif "powerpoint" in text:
        search_C("POWERPNT.EXE")
    elif "trình duyệt" in text:
        search_C("msedge.exe")


# Phương thức hoạt động chính
while True :
    you = get_ai_listen()

    if you == "":
        count += 1
        if count <= 2:
            ai_brain = "Tôi không nghe rõ, bạn hãy thử lại!"
            ai_speak(ai_brain)
    elif "ngày" in you:
        today = date.today()
        ai_brain = today.strftime("%d/%m/%Y")
        ai_speak("Ngày hôm nay là: " + ai_brain + " nha {}".format(name_sir))
        count = 0
    elif "thứ mấy" in you and "hôm nay" in you:
        week = date.today().weekday()
        ai_speak(weekToday(week))
        count = 0
    elif "mấy giờ" in you:
        now = datetime.now()
        ai_brain = now.strftime("%H:%M")
        ai_speak(ai_brain)
        count = 0
    elif "đùa" in you:
        ai_brain = pyjokes.get_joke(language="en")
        translation = translator.translate(ai_brain, dest='vi', src='auto')
        ai_speak(translation.text)
        count = 0
    # elif "mở trình duyệt" in you or "mở trình duyệt và tìm kiếm" in you:
    #     open_brower(you)
    #     count = 0
    elif "mở google và tìm kiếm" in you:
        google_search(you)
        count = 0
    elif "mở ứng dụng" in you:
        open_application()
        count = 0;
    elif "youtube" in you:
        youtube_search()
        count = 0
    elif "tìm kiếm trong máy" in you or "tìm kiếm" == you:
        number_disk()
        count = 0;
    elif "tạm biệt" in you or "bye" in you or "tắt" in you or "cảm ơn" in you:
        ai_brain = "Tạm biệt"
        ai_speak(ai_brain)
        break
    else:
        ai_speak(response(you))
        count = 0

    if count > 2 and you == "":
        ai_brain = "Hình như bạn đếch cần tôi nữa rồi, tạm biệt!"
        ai_speak(ai_brain)
        break
