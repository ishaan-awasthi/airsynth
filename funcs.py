import cv2
import threading
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import pyaudio
from pysine import sine

class Landmarker(): # TRACKER OBJECT | COURTESY @ OWEN TALMAGE
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()

    def createLandmarker(self):
        def update_result(result, _, __):
            self.result = result
            del _, __
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, 
            num_hands = 2,
            min_hand_detection_confidence = 0.3, 
            min_hand_presence_confidence = 0.3,
            min_tracking_confidence = 0.3,
            result_callback=update_result)
        #self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        self.landmarker.close()

class WavePlayerLoop(threading.Thread):
    def __init__(self, freq=440., volume=0.5, duration=1.):
        threading.Thread.__init__(self)
        self.p = pyaudio.PyAudio()
        self.volume = volume     # range [0.0, 1.0]
        self.fs = 44100          # sampling rate, Hz, must be integer
        self.duration = duration # in seconds, may be float
        self.f = freq            # sine frequency, Hz, may be float

    def run(self):
        self.samples = (np.sin(2*np.pi*np.arange(self.fs*self.duration)*self.f/self.fs)).astype(np.float32)
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True)
        self.stream.write(self.volume*self.samples)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def close(self):
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class MelodyStream(threading.Thread):
    def __init__(self, freq=440., duration=1.):
        threading.Thread.__init__(self)
        self.f = freq
        self.duration = duration # in seconds, may be float
        
    def run(self):
        sine(self.f, self.duration)

def add_strip(frame, start, end, bgr): # PRIVATE FOR RENDER_COLOR
    x = start[0]
    y = start[1]
    dx = end[0] - x
    dy = end[1] - y

    colored_strip = np.ones((dy, dx, 3), dtype='uint8') * 255
    colored_strip[:, :, 0] = bgr[0] 
    colored_strip[:, :, 1] = bgr[1]
    colored_strip[:, :, 2] = bgr[2] 

    alpha = 0.8
    added_image = cv2.addWeighted(frame[y:y+dy, x:x+dx, :], alpha, colored_strip, 1 - alpha, 0)
    frame[y:y+dy, x:x+dx] = added_image

def render_color(frame, h, w): # DISPLAY COLOR FILTER
    add_strip(frame, (0,int(h*0/8)), (w, int(h*1/8)), (0, 0, 255))
    add_strip(frame, (0,int(h*1/8)), (w, int(h*2/8)), (0, 127, 255))
    add_strip(frame, (0,int(h*2/8)), (w, int(h*3/8)), (0, 255, 255))
    add_strip(frame, (0,int(h*3/8)), (w, int(h*4/8)), (0, 255, 0))
    add_strip(frame, (0,int(h*4/8)), (w, int(h*5/8)), (155, 150, 0))
    add_strip(frame, (0,int(h*5/8)), (w, int(h*6/8)), (255, 0, 0))
    add_strip(frame, (0,int(h*6/8)), (w, int(h*7/8)), (211, 0, 148))
    add_strip(frame, (0,int(h*7/8)), (w, int(h*8/8)), (240, 16, 255))
    add_strip(frame, (955,0), (965, 1080), (255, 255, 255))


def render_landmarks(frame, detection_result): # LANDMARK RENDERING | COURTESY @ GOOGLE MEDIAPIPE
    try:
        if detection_result.hand_landmarks == []:
            return frame
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(frame)
            
            for hand in hand_landmarks_list:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand])
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS)
            return annotated_image
    except:
        return frame

def count_fingers_raised(frame, detection_result: mp.tasks.vision.HandLandmarkerResult): # FINGER COUNTING | COURTESY @ OWEN TALMAGE
    try:
        hand_landmarks_list = detection_result.hand_landmarks
        numRaised = 0
        for hand in hand_landmarks_list:
            hand_landmarks = hand
            for i in range(8,21,4):
                # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
                tip_y = hand_landmarks[i].y
                dip_y = hand_landmarks[i-1].y
                pip_y = hand_landmarks[i-2].y
                mcp_y = hand_landmarks[i-3].y
                if tip_y < min(dip_y,pip_y,mcp_y):
                    numRaised += 1
        annotated_image = np.copy(frame)
        cv2.putText(img = annotated_image, text = str(numRaised) + " Fingers Raised",
                        org = (100, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 3, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
        return annotated_image
    except:
        return frame

def fingertip_y(frame, detection_result, current_zone, h): # TRACK ZONE OF FINGERTIPS
    try:
        hand_landmarks = detection_result.hand_landmarks[0]
        y = 0
        for i in range(8,21,4):
            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
            tip_y = hand_landmarks[i].y
            dip_y = hand_landmarks[i-1].y
            pip_y = hand_landmarks[i-2].y
            mcp_y = hand_landmarks[i-3].y
            if tip_y < min(dip_y,pip_y,mcp_y):
                y += hand_landmarks[i].y
        annotated_image = np.copy(frame)
        height = annotated_image.shape[0]
        y = round(y*height/4)
        new_current_zone = ("red" if y<h*1/8 else
                            "orange" if y<h*2/8 else
                            "yellow" if y<h*3/8 else
                            "green" if y<h*4/8 else
                            "blue" if y<h*5/8 else
                            "indigo" if y<h*6/8 else
                            "purple" if y<h*7/8 else
                            "pink")
        """cv2.putText(img = annotated_image, text = new_current_zone,
                        org = (100, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 3, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)"""
        return annotated_image, new_current_zone
    except:
        new_current_zone = current_zone
        return frame, new_current_zone