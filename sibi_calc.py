import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import threading

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.7)

# KALKULATOR STATE
class SibiCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.num1 = 0
        self.operator = None
        self.num2 = 0
        self.result = 0
        self.state = "INPUT_NUM1"  # INPUT_NUM1, INPUT_OP, INPUT_NUM2, SHOW_RESULT
        self.stable_count = 0
        self.hold_frames = 0
        self.last_gesture = 0
        
    def process_gesture(self, gesture):
        if self.hold_frames > 0:
            self.hold_frames -= 1
            return self.result if self.state == "SHOW_RESULT" else 0
            
        # Stabilizer
        if gesture == self.last_gesture:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_gesture = gesture
            
        if self.stable_count < 8:
            return 0
            
        stable_gesture = gesture
        
        # State machine
        if self.state == "INPUT_NUM1":
            if 1 <= stable_gesture <= 100:
                self.num1 = stable_gesture
                self.state = "INPUT_OP"
                self.hold_frames = 30
                return stable_gesture
            elif stable_gesture == "CLEAR":
                self.reset()
                return 0
                
        elif self.state == "INPUT_OP":
            if stable_gesture == "+":
                self.operator = "+"
                self.state = "INPUT_NUM2"
                self.hold_frames = 30
                return stable_gesture
            elif stable_gesture == "-":
                self.operator = "-"
                self.state = "INPUT_NUM2"
                self.hold_frames = 30
                return stable_gesture
            elif stable_gesture == "X":
                self.operator = "*"
                self.state = "INPUT_NUM2"
                self.hold_frames = 30
                return stable_gesture
            elif stable_gesture == "/":
                self.operator = "/"
                self.state = "INPUT_NUM2"
                self.hold_frames = 30
                return stable_gesture
            elif stable_gesture == "CLEAR":
                self.reset()
                return 0
                
        elif self.state == "INPUT_NUM2":
            if 1 <= stable_gesture <= 100:
                self.num2 = stable_gesture
                self.state = "INPUT_EQUAL"
                self.hold_frames = 30
                return stable_gesture
            elif stable_gesture == "CLEAR":
                self.reset()
                return 0
                
        elif self.state == "INPUT_EQUAL":
            if stable_gesture == "=":
                self.calculate()
                self.state = "SHOW_RESULT"
                self.hold_frames = 60
                return self.result
            elif stable_gesture == "CLEAR":
                self.reset()
                return 0
                
        return 0
    
    def calculate(self):
        try:
            if self.operator == "+":
                self.result = self.num1 + self.num2
            elif self.operator == "-":
                self.result = self.num1 - self.num2
            elif self.operator == "*":
                self.result = self.num1 * self.num2
            elif self.operator == "/":
                if self.num2 != 0:
                    self.result = self.num1 / self.num2
                else:
                    self.result = "ERROR"
        except:
            self.result = "ERROR"

# Inisialisasi
calc = SibiCalculator()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# GESTURE DETECTION
def finger_status(lm):
    status = [0,0,0,0,0]  # T,I,M,R,P
    
    # Thumb
    palm_view = lm[5].x < lm[17].x
    if palm_view:
        status[0] = 1 if lm[4].x < lm[3].x else 0
    else:
        status[0] = 1 if lm[4].x > lm[3].x else 0
    
    # Fingers
    tips = [8,12,16,20]
    pips = [6,10,14,18]
    for i, (tip, pip) in enumerate(zip(tips, pips), 1):
        status[i] = 1 if lm[tip].y < lm[pip].y else 0
        
    return status

def detect_gesture(lm, status):
    T,I,M,R,P = status
    
    # ANGKA 1-9 (SIBI Standard)
    if [T,I,M,R,P] == [0,1,0,0,0]: return 1
    if [T,I,M,R,P] == [0,1,1,0,0]: return 2
    if [T,I,M,R,P] == [1,1,1,0,0]: return 3
    if [T,I,M,R,P] == [0,1,1,1,1]: return 4
    if [T,I,M,R,P] == [1,1,1,1,1]: return 5
    if [T,I,M,R,P] == [0,1,1,1,0]: return 6
    if [T,I,M,R,P] == [0,1,1,0,1]: return 7
    if [T,I,M,R,P] == [0,1,0,1,1]: return 8
    if [T,I,M,R,P] == [0,0,1,1,1]: return 9
    
    # OPERATOR SIBI
    # + : Dua jari telunjuk sejajar
    if I==1 and M==1 and T==0 and R==0 and P==0 and abs(lm[8].y - lm[12].y) < 0.05:
        return "+"
    
    # - : Satu garis lurus telunjuk
    if I==1 and all(x==0 for x in [T,M,R,P]) and lm[8].y > lm[6].y - 0.1:
        return "-"
    
    # × : Dua jari silang (index & middle)
    if I==1 and M==1 and T==0 and R==0 and P==0 and abs(lm[8].x - lm[12].x) < 0.08:
        return "X"
    
    # ÷ : Telunjuk & pinky sejajar
    if I==1 and P==1 and T==0 and M==0 and R==0 and abs(lm[8].y - lm[20].y) < 0.06:
        return "/"
    
    # = : Dua garis horizontal (index & middle horizontal)
    if I==1 and M==1 and abs(lm[8].x - lm[12].x) > 0.15 and lm[8].y > lm[6].y:
        return "="
    
    # CLEAR: Semua jari tertutup
    if all(x==0 for x in status):
        return "CLEAR"
    
    # 10,20,...,100 pakai modifier (sama seperti sebelumnya)
    if I==0 and M==0 and R==0 and P==0:  # Thumb up only
        return 10
    
    return 0

# Main Loop
cv2.namedWindow("SIBI Calculator", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("SIBI Calculator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🚀 SIBI Calculator Ready!")
print("ESC = Exit | C = Clear")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    gesture = 0
    info_text = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Right":
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                lm = hand_landmarks.landmark
                status = finger_status(lm)
                gesture = detect_gesture(lm, status)
                
                # Process calculator
                display_num = calc.process_gesture(gesture)
                
                # Status display
                if calc.state == "INPUT_NUM1":
                    info_text = f"Num1: {calc.num1}"
                elif calc.state == "INPUT_OP":
                    info_text = f"{calc.num1} {calc.operator}"
                elif calc.state == "INPUT_NUM2":
                    info_text = f"{calc.num1} {calc.operator} {calc.num2}"
                elif calc.state == "SHOW_RESULT":
                    info_text = f"={calc.result}"
    
    # DISPLAY BESAR
    if isinstance(display_num, (int, float)) or display_num in ["+", "-", "X", "/", "="]:
        cv2.putText(frame, str(display_num), (200, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), 12)
    
    # INFO TEXT
    cv2.putText(frame, info_text, (10, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # STATE INDICATOR
    state_color = (0,255,255) if calc.state != "SHOW_RESULT" else (0,255,0)
    cv2.putText(frame, f"State: {calc.state}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    
    cv2.imshow("SIBI Calculator", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):  # Clear
        calc.reset()

cap.release()
cv2.destroyAllWindows()
print("👋 SIBI Calculator Closed!")
