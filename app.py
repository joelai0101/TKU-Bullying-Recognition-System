import base64
from time import time
from unittest import result
from flask import Flask, render_template, Response, request, redirect, jsonify, url_for, stream_with_context, render_template_string
import cv2
import speech_recognition as sr
from datetime import datetime
import pyaudio
import wave
import numpy as np
import audio_processing
import text_processing
import video_processing
from multiprocessing import Process, Pool, Semaphore
import threading
import time
import requests

app = Flask(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = ""

pa = pyaudio.PyAudio()
r = sr.Recognizer()

camera = cv2.VideoCapture(0)  # use 0 for web camera

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    @stream_with_context
    def generate():

        video_streamer = video_recorder()
        video_streamer.start()
        
        v_id = 0
        v_semaphore = threading.Semaphore(2)

        frame_list = []
        frame_b64_list = []
        time.sleep(10)
        while True:
            try:
                while len(frame_list) != 60 or len(frame_b64_list) != 60:
                    frame_list = video_streamer.get_frame_list()
                    frame_b64_list = video_streamer.get_frame_b64_list()
                frame_list = video_streamer.get_frame_list()
                frame_b64_list = video_streamer.get_frame_b64_list()
                print(len(frame_list), len(frame_b64_list))
                video_predictor = video_prediction(frame_list, v_id, v_semaphore)
                print("create video_prediction thread {}".format(v_id))
                v_id += 1
                video_predictor.start()
                video_predictor.join()  # GPU: time.sleep(ms)
                pred_list = video_predictor.get_pred_list()
                frames_template = '''
                <div class="">
                    <h3 class="mt-5">影像辨識預測結果</h3>
                    <ul style="white-space: nowrap;">
                        {% for num in range(10)%}
                            <li style="list-style: none; display:inline;"><img src="data:image/jpg;base64, {{frame_list[6*num]}}" width="9%" alt=""></li>
                        {%endfor%}
                        <br>
                        {% for num in range(10)%}
                            <li style="list-style: none; display:inline;"><p style="display:inline;">{{pred_list[num]}}</p></li>
                        {%endfor%}
                    </ul>
                </div>
                '''
                yield render_template_string(frames_template, frame_list=frame_b64_list, pred_list=pred_list)
                frame_list = []
                frame_b64_list = []
            except Exception as e:
                print("Exception: "+str(e))
                frame_list = []
                frame_b64_list = []

    return app.response_class(generate())


@app.route('/audio_text')
def audio_text():
    @stream_with_context
    def generate():

        audio_streamer = audio_recorder()
        audio_streamer.start()

        a_id = 0
        t_id = 0

        a_semaphore = threading.Semaphore(2)
        t_semaphore = threading.Semaphore(2)

        while(True):
            time.sleep(10)
            frames = audio_streamer.get_frames()

            WAVE_OUTPUT_FILENAME = "./record/audio/" + \
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".wav"

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
                audio = r.record(source)

            try:

                speech2text = r.recognize_google(audio, language="zh-TW")
                print("Text: "+speech2text)

                emotion_predictor = audio_prediction(
                    WAVE_OUTPUT_FILENAME, a_id, a_semaphore)
                print("create audio_prediction thread {}".format(a_id))
                a_id += 1

                offensive_predictor = text_prediction(
                    speech2text, t_id, t_semaphore)
                print("create text_prediction thread {}".format(t_id))
                t_id += 1

                emotion_predictor.start()
                offensive_predictor.start()
                emotion_predictor.join()
                offensive_predictor.join()

                emotion_pred = emotion_predictor.get_predict()
                emotion_label = emotion_predictor.get_label()
                print(emotion_pred, emotion_label)

                offensive_pred = offensive_predictor.get_predict()
                offensive_label = offensive_predictor.get_label()
                print(offensive_pred, offensive_label)

                stream_template = '''
                <div class="col-lg-8  offset-lg-2">
                    <h3 class="mt-5">情緒預測結果</h3>
                    <p>{{emotion}}</p>
                    <h3 class="mt-5">語音轉文字結果（需連網）</h3>
                    <p>{{text}}</p>
                    <h3 class="mt-5">冒犯性文字預測結果</h3>
                    <p>{{offensive}}</p>
                </div>
                '''
                yield render_template_string(stream_template, text=speech2text, emotion=emotion_label, offensive=offensive_label)
            except Exception as e:
                print("Exception: "+str(e))
                

    return app.response_class(generate())

class video_recorder(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.frame_list = []
        self.frame_b64_list = []

    def run(self):

        while True:
            img_frame_list = []
            frame_b64_list = []
            i = 0
            while len(img_frame_list) != 60:  # 取六十幀
                # Capture frame-by-frame
                success, frame = camera.read()  # read the camera frame
                # print(camera.get(cv2.CAP_PROP_FPS)) # 30.0
                if not success:
                    break
                else:
                    i += 1
                    if (i % 5 == 0):  # 30 fps, 10 sec, total 300 frames
                        img_frame_list.append(frame)
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('ascii')
                        frame_b64_list.append(frame_b64)
            
            self.frame_list = img_frame_list
            self.frame_b64_list = frame_b64_list

    def get_frame_list(self):
        try:
            return self.frame_list
        except Exception:
            return None

    def get_frame_b64_list(self):
        try:
            return self.frame_b64_list
        except Exception:
            return None
        

class audio_recorder(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            stream = pa.open(format=FORMAT, channels=CHANNELS,
                             rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)

            stream.start_stream()
            print("開始錄音十秒")
            self.frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                self.frames.append(data)
            print("錄音結束!")
            stream.stop_stream()
   
    def get_frames(self):
        try:
            return self.frames
        except Exception:
            return None

    


class video_prediction(threading.Thread):
    def __init__(self, list, id, semaphore):
        threading.Thread.__init__(self)
        self.list = list
        self.id = id
        self.semaphore = semaphore

    def run(self):

        self.semaphore.acquire()
        print("Semaphore acquired by video_prediction thread {}".format(self.id))

        img_frame_list = self.list
        #文字list
        x_test = []
        for frame_element in img_frame_list:
            x_test.append(video_processing.node_print(frame_element))
        x_test = np.array(x_test)
        try1 = x_test.reshape(10, 6, 72)  # 10秒, 每秒 6個幀, 每張有72個節點
        pred = []
        pred_new = []

        for num in range(10):
            x_test_1 = try1[num].reshape(1, 6, 72)
            pred.append(video_processing.model.predict(x_test_1))

        for b in pred:
            pred_new.append(int(b*100))
        
        index = 0
        flag = 0
        flag2 = 0
        while(True):
            if pred_new[index] >= 50:
                flag = flag+1
            else:
                flag = 0
            if flag > flag2:
                flag2 = flag

            index = index+1
            if index >= 10:
                if flag2 >= 2:
                    send_msg("肢體霸凌發生")
                    save_video(img_frame_list)
                break


        self.result = pred_new

        time.sleep(1)

        print("Semaphore released by video_prediction thread {}".format(self.id))
        self.semaphore.release()


    def get_pred_list(self):
        try:
            return self.result
        except Exception:
                return None


class audio_prediction(threading.Thread):

    def __init__(self, path, id, semaphore):
        threading.Thread.__init__(self)
        self.path = path
        self.id = id
        self.semaphore = semaphore

    def run(self):

        self.semaphore.acquire()
        print("Semaphore acquired by audio_prediction thread {}".format(self.id))
        self.predict, self.label = audio_processing.predict(self.path)
        send_msg("情緒語音辨識："+str(self.label))

        print("Semaphore released by audio_prediction thread {}".format(self.id))
        self.semaphore.release()

    def get_predict(self):
        try:
            return self.predict
        except Exception:
            return None

    def get_label(self):
        try:
            return self.label
        except Exception:
            return None


class text_prediction(threading.Thread):

    def __init__(self, text, id, semaphore):
        threading.Thread.__init__(self)
        self.text = text
        self.id = id
        self.semaphore = semaphore

    def run(self):

        self.semaphore.acquire()
        print("Semaphore acquired by text_prediction thread {}".format(self.id))
        self.predict, self.label = text_processing.predict(self.text)
        send_msg("冒犯文字辨識："+str(self.label))

        print("Semaphore released by text_prediction thread {}".format(self.id))
        self.semaphore.release()

    def get_predict(self):
        try:
            return self.predict
        except Exception:
            return None

    def get_label(self):
        try:
            return self.label
        except Exception:
            return None


def lineNotifyMessage(token, msg):

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, params=payload)
    return r.status_code

def send_msg(msg):
    token = "LRy9Etnwv9gri3frPsgWBUex6UF00Dk2Yev5AGl02in"
    lineNotifyMessage(token, msg)


def save_video(img_list):
    localtime = time.localtime()
    result = time.strftime("%m_%d_%I_%M", localtime)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    VIDEO_OUTPUT_FILENAME = str(result+'霸凌片段')
    out = cv2.VideoWriter('./record/video/'+VIDEO_OUTPUT_FILENAME+'.mp4', fourcc, 6, (640, 480))
    for img in img_list:
        out.write(img)

if __name__ == '__main__':
    app.run(debug=True)
