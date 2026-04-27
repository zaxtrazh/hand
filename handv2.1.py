import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# ====== WAVE & DISPLAY VARIABLES (sama) ======
prev_x = None
prev_dir = 0
dir_changes = 0
move_accum = 0
cooldown = 0

display_number = 0
hold_frames = 0
last_stable_number = 0
pending_tens = None
waiting_last_digit = False
stable_count = 0
prev_base = 0
STABLE_MIN = 8

# ====== CALCULATOR STATE ======
digit_list = []
calc_mode = False
current_operator = None
result = 0
calc_display = ""
math_step = 0

# ====== CLEAR STABILIZER - BARU ======
clear_stable_count = 0
CLEAR_STABLE_MIN = 12

# ====== ALL GESTURE FUNCTIONS (sama seperti sebelumnya) ======
def detect_wave(index_x):
    global prev_x, prev_dir, dir_changes, move_accum, cooldown
    if cooldown > 0:
        cooldown -= 1
        return False
    if prev_x is None:
        prev_x = index_x
        return False
    dx = index_x - prev_x
    prev_x = index_x
    move_accum += abs(dx)
    if abs(dx) < 0.025: return False
    direction = 1 if dx > 0 else -1
    if prev_dir == 0:
        prev_dir = direction
        return False
    if direction != prev_dir:
        dir_changes += 1
        prev_dir = direction
    if dir_changes >= 2 and move_accum > 0.04:
        dir_changes = 0
        move_accum = 0
        cooldown = 15
        return True

def is_thumb_up(lm): return lm[4].y - lm[2].y < -0.08 and abs(lm[4].x - lm[2].x) < 0.12
def is_c_hundred(lm):
    curled = (lm[8].y >= lm[6].y and lm[12].y >= lm[10].y and lm[16].y >= lm[14].y and lm[20].y >= lm[18].y and
              lm[8].y < lm[4].y and lm[12].y < lm[4].y and lm[16].y < lm[4].y and lm[20].y < lm[4].y)
    return curled and lm[4].y > lm[8].y and lm[4].x < lm[3].x and abs(lm[4].y - lm[8].y) > 0.07

def finger_status(lm):
    status = [0,0,0,0,0]
    palm_view = lm[5].x < lm[17].x
    if palm_view: status[0] = 1 if lm[4].x < lm[3].x else 0
    else: status[0] = 1 if lm[4].x > lm[3].x else 0
    tips, pips = [8,12,16,20], [6,10,14,18]
    for i,(tip,pip) in enumerate(zip(tips,pips), 1): status[i] = 1 if lm[tip].y < lm[pip].y else 0
    return status

def gesture_to_number(s, lm):
    T,I,M,R,P = s
    if is_c_hundred(lm): return 100
    if I==0 and M==0 and R==0 and P==0 and is_thumb_up(lm): return 10
    if s == [0,1,0,0,0]: return 1
    if s == [0,1,1,0,0]: return 2
    if s == [1,1,1,0,0]: return 3
    if s == [0,1,1,1,1]: return 4
    if s == [1,1,1,1,1]: return 5
    if s == [0,1,1,1,0]: return 6
    if s == [0,1,1,0,1]: return 7
    if s == [0,1,0,1,1]: return 8
    if s == [0,0,1,1,1]: return 9
    if s == [0,0,0,0,0]: return 0
    return 0

# ====== FIXED FUNCTIONS ======
def gesture_clear_stable(status):
    global clear_stable_count
    if status == [1,1,0,0,0]:
        clear_stable_count += 1
        return clear_stable_count >= CLEAR_STABLE_MIN
    clear_stable_count = 0
    return False

def detect_addition(right_lm, left_lm):
    ri, rm = right_lm[8], right_lm[12]
    li, lm_ = left_lm[8], left_lm[12]
    right_above = ri.y < li.y + 0.05 and rm.y < lm_.y + 0.05
    fingers_close = abs(ri.x - li.x) < 0.15 and abs(rm.x - lm_.x) < 0.15
    return right_above and fingers_close

def detect_multiplication(right_lm, left_lm):
    return abs(right_lm[8].x - left_lm[0].x) < 0.12 and abs(right_lm[8].y - left_lm[0].y) < 0.12 and left_lm[17].x > left_lm[5].x

def detect_division(right_lm, left_lm):
    return left_lm[0].z < left_lm[9].z and abs(right_lm[8].x - right_lm[5].x) < 0.08 and right_lm[8].y < left_lm[0].y + 0.1

def detect_equals(lm):
    return lm[4].y < lm[2].y and lm[20].y < lm[18].y and lm[8].y > lm[6].y and lm[12].y > lm[10].y and lm[16].y > lm[14].y

# ====== MAIN LOOP ======
cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    cv2.putText(frame, "Kamus BIMA | Calculator FIXED | DPPM 2026", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    number = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_count = len(results.multi_hand_landmarks)
        
        # ====== CLEAR - FIXED ======
        clear_detected = False
        if hand_count == 1:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    status = finger_status(hand_landmarks.landmark)
                    if gesture_clear_stable(status):
                        digit_list.clear()
                        current_operator = None
                        calc_display = "CLEARED!"
                        result = 0
                        math_step = 0
                        calc_mode = False
                        cv2.putText(frame, "CLEAR!", (250,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                        clear_detected = True
                        break
        
        if clear_detected: 
            cv2.imshow("Hand Tracking", frame)
            continue

        # ====== 1 TANGAN DIGIT - FIXED 10-99 ======
        if hand_count == 1:
            calc_mode = False
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                                         mp_draw.DrawingSpec(color=(0,255,255), thickness=2))
                    
                    lm = hand_landmarks.landmark
                    status = finger_status(lm)
                    base_number = gesture_to_number(status, lm)

                    if base_number == prev_base and base_number != 0:
                        stable_count += 1
                    else:
                        stable_count = 0
                    prev_base = base_number

                    stable_number = base_number if stable_count >= STABLE_MIN else 0
                    
                    # === DIGIT LOGIC SESUAI AWAL ===
                    if hold_frames > 0:
                        number = display_number
                        hold_frames -= 1
                    else:
                        display_number = 0
                        number = stable_number

                        # Semua logic 0-100 sama persis seperti kode awal Anda
                        if stable_number == 100 and last_stable_number == 1:
                            display_number = 100; number = 100; hold_frames = 25
                        elif waiting_last_digit and 0 <= stable_number <= 9:
                            display_number = last_stable_number + stable_number; number = display_number; hold_frames = 25; waiting_last_digit = False; pending_tens = 0; stable_number = 0; last_stable_number = 0
                        elif waiting_last_digit and last_stable_number == 10:
                            display_number = pending_tens * 10; number = display_number; hold_frames = 20; waiting_last_digit = True
                        elif stable_number == 10 and 1 <= last_stable_number <= 9:
                            pending_tens = last_stable_number; waiting_last_digit = True
                        elif 1 <= stable_number <= 9:
                            index_x = lm[8].x
                            if detect_wave(index_x):
                                display_number = stable_number + 10; number = display_number; hold_frames = 25

                    # === FIXED: Simpan SEMUA gesture 1-100 ===
                    if number != 0 and hold_frames == 0:
                        last_stable_number = number
                        display_number = last_stable_number
                        
                        # ✅ FIX: Semua angka tersimpan (termasuk 10-99)
                        if stable_number > 0 and len(digit_list) < 2:
                            digit_list.append(stable_number)
                            math_step += 1
                            cv2.putText(frame, f"SAVED {stable_number} ({len(digit_list)}/2)", (150,280), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # ====== 2 TANGAN OPERATOR (sama) ======
        elif hand_count == 2 and math_step == 1 and len(digit_list) == 1:
            right_hand = left_hand = None
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                                         mp_draw.DrawingSpec(color=(0,255,0), thickness=2))
                    right_hand = hand_landmarks.landmark
                else:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_draw.DrawingSpec(color=(255,0,0), thickness=2),
                                         mp_draw.DrawingSpec(color=(255,0,0), thickness=2))
                    left_hand = hand_landmarks.landmark
            
            if right_hand and left_hand and current_operator is None:
                if detect_addition(right_hand, left_hand): current_operator = "+"; math_step = 2
                elif detect_multiplication(right_hand, left_hand): current_operator = "×"; math_step = 2
                elif detect_division(right_hand, left_hand): current_operator = "÷"; math_step = 2
                
                if current_operator:
                    calc_display = f"{digit_list[0]} {current_operator} _ = ?"
                    cv2.putText(frame, current_operator, (300,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
                    cv2.putText(frame, "1 Tangan: Angka kedua", (150,320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # ====== EQUALS ======
        if hand_count == 1 and current_operator and len(digit_list) == 2:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right" and detect_equals(hand_landmarks.landmark):
                    op1, op2 = digit_list
                    result = op1 + op2 if current_operator == "+" else op1 * op2 if current_operator == "×" else op1 // op2 if current_operator == "÷" and op2 else 0
                    calc_display = f"{op1} {current_operator} {op2} = {result}"
                    cv2.putText(frame, "=", (300,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 5)
                    time.sleep(1.5)
                    digit_list.clear()
                    current_operator = None
                    math_step = 0

    else:
        prev_x = None; prev_dir = 0; dir_changes = 0; move_accum = 0

    # ====== DISPLAY ======
    cv2.putText(frame, f"Mode:{'MATH' if calc_mode else 'DIGIT'} Step:{math_step} List:{digit_list}", (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
    cv2.putText(frame, f"Op:{current_operator} {calc_display}", (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    cv2.putText(frame, f"N:{number} Clear:{clear_stable_count}/{CLEAR_STABLE_MIN}", (5,340), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    if calc_display and "=" in calc_display:
        cv2.putText(frame, calc_display, (80,320), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
    elif number: cv2.putText(frame, str(number), (250,350), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 10)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
