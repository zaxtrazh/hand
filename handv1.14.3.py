import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,  # ← DIUBAH untuk 2 tangan
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# ====== WAVE TRACKER (tetap sama) ======
prev_x = None
prev_dir = 0
dir_changes = 0
move_accum = 0
cooldown = 0

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

    if abs(dx) < 0.025:
        return False

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

# ====== DISPLAY HOLD (tetap sama) ======
display_number = 0
hold_frames = 0
last_stable_number = 0

# ====== MODE PENYUSUNAN ANGKA (tetap sama) ======
pending_tens = None
waiting_last_digit = False

# ====== STABILIZER (tetap sama) ======
stable_count = 0
prev_base = 0
STABLE_MIN = 8

# ====== CALCULATOR STATE - BARU! ======
digit_list = []  # Simpan angka dari mode digit
calc_mode = False  # False=digit mode, True=math mode
current_operator = None
operand1 = 0
operand2 = 0
result = 0
calc_display = ""

# ====== FUNGSI GESTURE BASIC (tetap sama) ======
def is_thumb_up(lm):
    mcp = lm[2]
    tip = lm[4]
    vx = tip.x - mcp.x
    vy = tip.y - mcp.y
    return vy < -0.08 and abs(vx) < 0.12

def is_c_hundred(lm):
    curled = (
        lm[8].y >= lm[6].y and
        lm[12].y >= lm[10].y and
        lm[16].y >= lm[14].y and
        lm[20].y >= lm[18].y and
        lm[8].y < lm[4].y and
        lm[12].y < lm[4].y and
        lm[16].y < lm[4].y and
        lm[20].y < lm[4].y
    )
    thumb_below = lm[4].y > lm[8].y
    thumb_left = lm[4].x < lm[3].x
    gap = abs(lm[4].y - lm[8].y) > 0.07
    return curled and thumb_below and thumb_left and gap

def finger_status(lm):
    status = [0,0,0,0,0]  # [T,I,M,R,P]
    palm_view = lm[5].x < lm[17].x

    if palm_view:
        if lm[4].x < lm[3].x:
            status[0] = 1
    else:
        if lm[4].x > lm[3].x:
            status[0] = 1

    tips = [8,12,16,20]
    pips = [6,10,14,18]

    for i,(tip,pip) in enumerate(zip(tips,pips), start=1):
        if lm[tip].y < lm[pip].y:
            status[i] = 1

    return status

def gesture_to_number(s, lm):
    T,I,M,R,P = s

    if is_c_hundred(lm):
        return 100

    if I==0 and M==0 and R==0 and P==0 and is_thumb_up(lm):
        return 10

    if [T,I,M,R,P] == [0,1,0,0,0]: return 1
    if [T,I,M,R,P] == [0,1,1,0,0]: return 2
    if [T,I,M,R,P] == [1,1,1,0,0]: return 3
    if [T,I,M,R,P] == [0,1,1,1,1]: return 4
    if [T,I,M,R,P] == [1,1,1,1,1]: return 5
    if [T,I,M,R,P] == [0,1,1,1,0]: return 6
    if [T,I,M,R,P] == [0,1,1,0,1]: return 7
    if [T,I,M,R,P] == [0,1,0,1,1]: return 8
    if [T,I,M,R,P] == [0,0,1,1,1]: return 9
    if [T,I,M,R,P] == [0,0,0,0,0]: return 0

    return 0

# ====== MATH OPERATOR GESTURES - BARU! ======
def detect_addition(right_lm, left_lm):
    # Jari telunjuk + tengah kanan di atas kiri
    right_index_tip = right_lm[8]
    right_middle_tip = right_lm[12]
    left_index_tip = left_lm[8]
    left_middle_tip = left_lm[12]
    
    # Right fingers above left fingers
    right_above = (right_index_tip.y < left_index_tip.y + 0.05 and 
                   right_middle_tip.y < left_middle_tip.y + 0.05)
    
    # Fingers close together (stacked)
    fingers_close = (abs(right_index_tip.x - left_index_tip.x) < 0.15 and
                     abs(right_middle_tip.x - left_middle_tip.x) < 0.15)
    
    return right_above and fingers_close

def detect_subtraction(right_lm, left_lm, prev_distance=0):
    # Mirip addition tapi jari kanan dijauhkan
    right_index_tip = right_lm[8]
    right_middle_tip = right_lm[12]
    left_index_tip = left_lm[8]
    left_middle_tip = left_lm[12]
    
    right_above = (right_index_tip.y < left_index_tip.y + 0.08 and 
                   right_middle_tip.y < left_middle_tip.y + 0.08)
    
    current_distance = abs(right_index_tip.x - left_index_tip.x) + abs(right_middle_tip.x - left_middle_tip.x)
    
    # Dijauhkan dari posisi stacked
    separated = current_distance > 0.25
    
    return right_above and separated

def detect_multiplication(right_lm, left_lm):
    # Ujung telunjuk kanan menempel telapak kiri
    right_index_tip = right_lm[8]
    left_palm = left_lm[0]  # wrist
    
    # Telapak kiri menghadap kanan (palm facing right)
    left_palm_right = left_lm[17].x > left_lm[5].x  # pinky > thumb
    
    touch_palm = (abs(right_index_tip.x - left_palm.x) < 0.12 and
                  abs(right_index_tip.y - left_palm.y) < 0.12)
    
    return touch_palm and left_palm_right

def detect_division(right_lm, left_lm):
    # Tangan kiri telapak atas, tangan kanan tegak lurus (memotong)
    left_palm_up = left_lm[0].z < left_lm[9].z  # wrist higher than middle pip
    
    right_vertical = abs(right_lm[8].x - right_lm[5].x) < 0.08  # index close to thumb
    right_across = right_lm[8].y < left_lm[0].y + 0.1  # crossing left hand
    
    return left_palm_up and right_vertical and right_across

def detect_equals(right_lm):
    # Jempol + kelingking kanan, 3 jari tengah nutup
    thumb_up = right_lm[4].y < right_lm[2].y
    pinky_up = right_lm[20].y < right_lm[18].y
    middle_fingers_down = (right_lm[8].y > right_lm[6].y and 
                          right_lm[12].y > right_lm[10].y and
                          right_lm[16].y > right_lm[14].y)
    return thumb_up and pinky_up and middle_fingers_down

# ====== MAIN LOOP ======
cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    cv2.putText(frame, f"Kamus BIMA | Calculator Mode | Didanai DPPM 2026", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    number = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_count = len(results.multi_hand_landmarks)
        
        # ====== 1 TANGAN KANAN = DIGIT MODE ======
        if hand_count == 1:
            calc_mode = False  # Exit math mode
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    lm = hand_landmarks.landmark
                    status = finger_status(lm)
                    base_number = gesture_to_number(status, lm)

                    # Stabilizer logic (sama seperti sebelumnya)
                    if base_number == prev_base and base_number != 0:
                        stable_count += 1
                    else:
                        stable_count = 0
                    prev_base = base_number

                    stable_number = base_number if stable_count >= STABLE_MIN else 0
                    
                    # Existing digit logic...
                    if hold_frames > 0:
                        number = display_number
                        hold_frames -= 1
                    else:
                        display_number = 0
                        number = stable_number

                        if stable_number == 100 and last_stable_number == 1:
                            display_number = 100
                            number = display_number
                            hold_frames = 25
                        elif waiting_last_digit and 0 <= stable_number <= 9:
                            display_number = last_stable_number + stable_number
                            number = display_number
                            hold_frames = 25
                            waiting_last_digit = False
                            pending_tens = 0
                            stable_number = 0
                            last_stable_number = 0
                        elif waiting_last_digit and last_stable_number == 10:
                            display_number = pending_tens * 10 
                            number = display_number
                            hold_frames = 20
                            waiting_last_digit = True
                        elif stable_number == 10 and 1 <= last_stable_number <= 9:
                            pending_tens = last_stable_number
                            waiting_last_digit = True
                        elif 1 <= stable_number <= 9:
                            index_x = lm[8].x
                            if detect_wave(index_x):
                                display_number = stable_number + 10
                                number = display_number
                                hold_frames = 25

                    if number != 0 and hold_frames == 0:
                        last_stable_number = number
                        display_number = last_stable_number
                        
        # ====== 2 TANGAN = MATH MODE ======
        elif hand_count == 2 and calc_mode == False:
            calc_mode = True
            digit_list.append(display_number)  # Simpan angka terakhir
            if len(digit_list) >= 2:
                operand1 = digit_list[-2]
                operand2 = digit_list[-1]
            cv2.putText(frame, "MATH MODE ACTIVATED!", (200,100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Process 2 hands untuk operator
        if hand_count == 2 and calc_mode:
            right_hand = None
            left_hand = None
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, (0,255,0), (0,255,0))
                    right_hand = hand_landmarks.landmark
                else:  # Left
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, (255,0,0), (255,0,0))
                    left_hand = hand_landmarks.landmark
            
            if right_hand is not None and left_hand is not None:
                # Deteksi operator
                if current_operator is None:  # Belum ada operator
                    if detect_addition(right_hand, left_hand):
                        current_operator = "+"
                        cv2.putText(frame, "+", (300,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
                    elif detect_multiplication(right_hand, left_hand):
                        current_operator = "×"
                        cv2.putText(frame, "×", (300,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
                    elif detect_division(right_hand, left_hand):
                        current_operator = "÷"
                        cv2.putText(frame, "÷", (300,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
                
                # Hitung hasil jika ada operator
                if current_operator and len(digit_list) >= 2:
                    op1, op2 = digit_list[-2], digit_list[-1]
                    if current_operator == "+":
                        result = op1 + op2
                    elif current_operator == "×":
                        result = op1 * op2
                    elif current_operator == "÷" and op2 != 0:
                        result = op1 // op2  # Integer division
                    else:
                        result = 0
                    
                    # Tampilkan expression
                    calc_display = f"{op1} {current_operator} {op2} = {result}"
                
                # Equals gesture untuk finalize
                if detect_equals(right_hand):
                    calc_display = f"RESULT: {result}"
                    cv2.putText(frame, "=", (300,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 5)
                    # Reset untuk kalkulasi berikutnya
                    current_operator = None
                    digit_list = digit_list[:len(digit_list)-2]

    else:
        # Tidak ada tangan → simpan ke list (digit mode)
        if not calc_mode and display_number != 0:
            digit_list.append(display_number)
            display_number = 0
        prev_x = None
        prev_dir = 0
        dir_changes = 0
        move_accum = 0

    # ====== DISPLAY INFO ======
    cv2.putText(frame, f"Mode: {'MATH' if calc_mode else 'DIGIT'}", (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Digits: {digit_list}", (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    cv2.putText(frame, f"Calc: {calc_display}", (5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Number: {number}", (5,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    # Tampilkan hasil besar
    if calc_display:
        cv2.putText(frame, calc_display, (50,320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    else:
        cv2.putText(frame, f"{number}", (250,350), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 10)

    cv2.imshow("Hand Tracking", frame)
   
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
