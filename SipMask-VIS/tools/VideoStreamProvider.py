# file: videocaptureasync.py
import threading
import os
import cv2
import time

from datetime import datetime

class VideoStreamProvider:
    def __init__(self, src=None, width=None, height=None, play_back_speed=1):
        if src is None:
            print('Error: stream path for VideoStreamProvider not set.')
            exit()

        print("VideoStreamProvider: load images from " + str(src))

        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.play_back_speed = play_back_speed

        if width is None and height is None:
            print('Use native width & height')
        else:
            print('Use given width & height: '+str(width)+" "+str(height))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps is not None:
            print("Frames per second using self.cap.get(cv2.CAP_PROP_FPS) : {0}".format(self.fps))
        else:
            self.fps = 5
            print("Frames per second using default : {0}".format(self.fps))


        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.update_count = 0
        self.read_count = 0

        self.start()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def get(self, var1):
        self.cap.get(var1)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        #self.thread.start()
        #self.thread_screenshot = threading.Thread(target=self.takescreenshot, args=())
        #self.thread_screenshot.start()
        return self

    def update(self):
        frame_time = 1/self.fps / self.play_back_speed
        start_time = time.time()
        last_time = start_time
        print ("frame_time "+str(frame_time))
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.update_count += 1

                diff = time.time() - last_time
                time.sleep(max(0, frame_time - diff))
                # print(time.time() -last_time)
                #print(self.update_count / (time.time() - start_time))
                last_time = time.time()


    def read(self):
        frame = None
        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("frame was not grabbed "+str(self.read_count))
            self.read_count += 1
        # convert to RGB
        return self.grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

    def takescreenshot(self, type="minutely" ):
        while True:
            time.sleep(60)
            frame = self.read()
            if frame is None:
                print("cant save minutly frame")
                return


            if self.endshottaken:
                return

            pathForProcessed = 'doku/'+str(type)
            if not os.path.exists(pathForProcessed):
                os.makedirs(pathForProcessed)
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            cv2.imwrite(pathForProcessed + '/'+ dt_string + '.png', frame)
            print("writed endshot to "+pathForProcessed + '/'+ dt_string + '.png')
            self.endshottaken = True