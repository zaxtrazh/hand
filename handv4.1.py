import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,  # ⭐ Support 2 tangan
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# ====== WAVE TRACKER (ASLI) ======
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

    if abs(dx) < 0.02:
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

# ====== DISPLAY HOLD (ASLI) ======
display_number = None
hold_frames = 0
last_stable_number = None

# ====== MODE PENYUSUNAN ANGKA (ASLI) ======
pending_tens = None
waiting_last_digit = False

# ====== STABILIZER (ASLI) ======
stable_count = 0
prev_base = 0
STABLE_MIN = 1

# ⭐ ====== BARU: MATH SYSTEM ======
calculation = []  # [num1, op, num2]
current_mode = "DIGIT"  # DIGIT, OPERATOR
last_display_number = None
operator_stable = ""
result = None

# ====== THUMB UP (ASLI) ======
def is_thumb_up(lm):
    jempol = lm[4].y < lm[2].y
    telunjuk = lm[6].y < lm[10].y
    tengah = lm[10].y < lm[14].y
    manis = lm[14].y < lm[18].y
    return jempol and telunjuk and tengah and manis

# ====== GESTURE C (ASLI) ======
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

# ====== FINGER STATUS (ASLI) ======
def finger_status(lm):
    status = [0,0,0,0,0]
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

# ====== GESTURE MAPPING (ASLI) ======
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

# ⭐ ====== BARU: OPERATOR DETECTION ======
def detect_operator(right_lm, left_lm=None):
    kanan_2_jari = right_lm[8].x < right_lm[6].x and right_lm[12].x < right_lm[10].x and right_lm[5].y < right_lm[16].y and right_lm[5].y < right_lm[16].y 
    kiri_2_jari = left_lm[8].x > left_lm[6].x and left_lm[12].x > left_lm[10].x and left_lm[8].x > left_lm[16].x and left_lm[8].x > left_lm[20].x and left_lm[8].x > left_lm[4].x    
    jarak = abs(right_lm[8].y - left_lm[8].y) < 0.06 and abs(right_lm[12].y - left_lm[12].y) < 0.06 and left_lm[4].y < right_lm[4].y
    if kanan_2_jari and kiri_2_jari and jarak:
        return "+"   
    
    # - PENGURANGAN: I+M kanan dijauhkan dari I+M kiri
    dua_jari_kanan = right_lm[8].y < right_lm[6].y and right_lm[12].y < right_lm[10].y and right_lm[10].y < right_lm[16].y and right_lm[10].y < right_lm[20].y and right_lm[14].y < right_lm[4].y
    dua_jari_kiri = left_lm[8].x > left_lm[6].x and left_lm[12].x > left_lm[10].x and left_lm[8].x > left_lm[16].x and left_lm[8].x > left_lm[20].x and left_lm[8].x > left_lm[4].x
    rentang = abs(right_lm[8].x - left_lm[8].x) > 0.18
    if dua_jari_kanan and dua_jari_kiri and rentang:
       return "-"
    
    # * PERKALIAN: Ujung I kanan → pangkal M kiri + telapak kiri samping kanan
    kiri_tegak = left_lm[8].y < left_lm[4].y and left_lm[12].y < left_lm[4].y and left_lm[16].y < left_lm[4].y and left_lm[20].y < left_lm[4].y   
    kanan_tunjuk = right_lm[8].x < right_lm[5].x and right_lm[8].x < right_lm[4].x
    tempel_tangan = abs(right_lm[8].x - left_lm[9].x) < 0.06
    if kiri_tegak and kanan_tunjuk and tempel_tangan:
        return "x"
    
    # / PEMBAGIAN: Telapak kiri atas + kanan tegak lurus lama
    kiri_tengadah = left_lm[8].x > left_lm[4].x and left_lm[12].x > left_lm[4].x and left_lm[16].x > left_lm[4].x and left_lm[20].x > left_lm[4].x
    kanan_potong = right_lm[8].y > right_lm[4].y and right_lm[12].y > right_lm[4].y and right_lm[16].y > right_lm[4].y and right_lm[20].y > right_lm[4].y
    if kanan_potong and kiri_tengadah:
        return ":"
    
    # = SAMA DENGAN: Jempol + kelingking kanan, 3 jari tutup
    right_thumb_up = right_lm[4].y < right_lm[2].y
    right_pinky_up = right_lm[20].y < right_lm[18].y
    right_i_closed = right_lm[8].y >= right_lm[6].y
    right_m_closed = right_lm[12].y >= right_lm[10].y
    right_r_closed = right_lm[16].y >= right_lm[14].y
    if right_thumb_up and right_pinky_up and right_i_closed and right_m_closed and right_r_closed:
        return "="
    
    # CLEAR: Telunjuk + jempol kiri
    right_i_up_clear = right_lm[8].y < right_lm[6].y and right_lm[12].y > right_lm[10].y and right_lm[16].y > right_lm[14].y and right_lm[20].y > right_lm[18].y
    right_thumb_up_clear = right_lm[4].y < right_lm[2].y
    if right_i_up_clear and right_thumb_up_clear:
        return "CLEAR"
    
    # EXIT: tangan membentuk gunung
    tempel_telunjuk = abs(right_lm[8].x - left_lm[8].x) < 0.06
    tempel_tengah = abs(right_lm[12].x - left_lm[12].x) < 0.06
    tempel_manis = abs(right_lm[16].x - left_lm[16].x) < 0.06
    gestur_exit_kiri = left_lm[8].y < left_lm[4].y and left_lm[12].y < left_lm[4].y and left_lm[16].y < left_lm[4].y and left_lm[20].y < left_lm[4].y
    gestur_exit_kanan = right_lm[8].y < right_lm[4].y and right_lm[12].y < right_lm[4].y and right_lm[16].y < right_lm[4].y and right_lm[20].y < right_lm[4].y
    if tempel_telunjuk and tempel_tengah and tempel_manis and gestur_exit_kiri and gestur_exit_kanan:
        return "EXIT"
    
    return None

cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    cv2.putText(frame, f"Kamus BIMA | Didanai oleh DPPM Kemdiktisaintek 2026", (5,20),
    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 2)

    number = 0
    hands_detected = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = len(results.multi_hand_landmarks)
        
        # ⭐ 1 TANGAN KANAN = MODE DIGIT (ASLI + SAVE LOGIC)
        if hands_detected == 1:
            current_mode = "DIGIT"
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    lm = hand_landmarks.landmark
                    status = finger_status(lm)
                    base_number = gesture_to_number(status, lm)

                    # ASLI STABILIZER & LOGIC
                    if base_number == prev_base and base_number != 0:
                        stable_count += 1
                    else:
                        stable_count = 0
                    prev_base = base_number

                    stable_number = base_number if stable_count >= STABLE_MIN else 0
                    
                    if hold_frames > 0:
                        number = display_number
                        hold_frames -= 1
                    else:
                        display_number = None
                        number = stable_number

                        # ASLI GESTURE LOGIC
                        if stable_number == 100 and last_stable_number == 1:
                            display_number = 100
                            number = display_number
                            last_display_number = number
                            hold_frames = 25
                        elif waiting_last_digit and 1 <= stable_number <= 9 and last_stable_number != None:
                            display_number = last_stable_number + stable_number
                            number = display_number
                            last_stable_number = number
                            last_display_number = last_stable_number
                            hold_frames = 25
                            waiting_last_digit = False
                        elif waiting_last_digit and last_stable_number == 10:
                            display_number = pending_tens * 10 
                            number = display_number
                            last_display_number = number
                            hold_frames = 5
                        elif stable_number == 10 and last_stable_number != None and 1 <= last_stable_number <= 9 :
                            pending_tens = last_stable_number
                            display_number = pending_tens * 10 
                            last_display_number = display_number
                            waiting_last_digit = True
                        elif 1 <= stable_number <= 9 and last_stable_number != 10:
                            index_x = lm[8].x
                            if detect_wave(index_x):
                                display_number = stable_number + 10
                                number = display_number
                                last_stable_number = number
                                last_display_number = last_stable_number
                                hold_frames = 25
                                
                    if number !=0 and hold_frames == 0:
                        last_stable_number = number
                        display_number = last_stable_number
                        last_display_number = display_number  # ⭐ Simpan stabil number
        
        # ⭐ 2 TANGAN = MODE OPERATOR
        elif hands_detected == 2:
            current_mode = "OPERATOR"
            right_lm = None
            left_lm = None
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    right_lm = hand_landmarks.landmark
                    cv2.putText(frame, "RIGHT", (int(hand_landmarks.landmark[0].x*640), int(hand_landmarks.landmark[0].y*360)-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    left_lm = hand_landmarks.landmark
                    cv2.putText(frame, "LEFT", (int(hand_landmarks.landmark[0].x*640), int(hand_landmarks.landmark[0].y*360)-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            if right_lm and left_lm:
                operator_detected = detect_operator(right_lm, left_lm)
                if operator_detected:
                    cv2.putText(frame, operator_detected, (300,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 5)
                    operator_stable = operator_detected

    else:
        # ⭐ TANGAN TURUN: SIMPAN KE CALCULATION
        if current_mode == "DIGIT" and last_display_number != None and (len(calculation) == 0 or len(calculation) == 2) :
            calculation.append(last_display_number)
            last_display_number = None
            number = None
            display_number = None
            stable_number = None
            last_stable_number = None
            print(f"DIGIT SAVED: {calculation}")  # Debug
            
        elif current_mode == "OPERATOR" and operator_stable != "" : #and operator_stable != None:
            if operator_stable == "CLEAR":
                calculation.clear()
                result = None
                print("CLEAR ALL")
            elif operator_stable == "EXIT":
                break
            elif len(calculation) == 1 and operator_stable != "=" :
                calculation.append(operator_stable)
                print(f"OP SAVED: {calculation}")  # Debug
            elif operator_stable == "=" and len(calculation) == 3:
                calculation.append(operator_stable)
                print(f"OP SAVED: {calculation}")  # Debug
            
            operator_stable = ""
        
        # Reset wave
        prev_x = None
        prev_dir = 0
        dir_changes = 0
        move_accum = 0

    # ⭐ HITUNG RESULT jika "=" adalah ELEMEN TERAKHIR (FIXED!)
    if len(calculation) >= 4 and calculation[-1] == "=":
        num1 = calculation[0]  # Index 0
        op = calculation[1]    # Index 1  
        num2 = calculation[2]  # Index 2
        
        calculation.pop() # Hapus "=" dari list
        
        if op == "+": result = num1 + num2
        elif op == "-": result = num1 - num2
        elif op == "x": result = num1 * num2
        elif op == ":": result = num1 / num2 if num2 != 0 else 0
        
        number2 = result
        print(f"CALC: {num1} {op} {num2} = {result}")  # Debug

    # DISPLAY INFO
    cv2.putText(frame, f"N:{number}", (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"LS:{last_stable_number}", (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"DN:{last_display_number}", (5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if current_mode == "DIGIT":
        cv2.putText(frame, f"{last_display_number}", (300,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (100,0,255), 5)
    cv2.putText(frame, f"C: {calculation}", (5,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"M: {current_mode}", (5,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"O: {operator_stable}", (5,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if result == None:
        cv2.putText(frame, f"{''.join(map(str,calculation))}", (5,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)    
    else:
        cv2.putText(frame, f"{''.join(map(str,calculation))}=", (5,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
        cv2.putText(frame, f"{result}", (5,350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)  
    
    cv2.imshow("Hand Tracking", frame)
   
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
