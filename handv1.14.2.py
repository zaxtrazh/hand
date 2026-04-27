import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# ====== WAVE TRACKER ======
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

    return False

# ====== DISPLAY HOLD ======
display_number = 0
hold_frames = 0
last_stable_number = 0

# ====== MODE PENYUSUNAN ANGKA ======
pending_tens = None
waiting_last_digit = False

# ====== STABILIZER ======
stable_count = 0
prev_base = 0
STABLE_MIN = 8

# ====== THUMB UP (10) ======
def is_thumb_up(lm):
    mcp = lm[2]
    tip = lm[4]
    vx = tip.x - mcp.x
    vy = tip.y - mcp.y
    return vy < -0.08 and abs(vx) < 0.12

# ====== GESTURE C (100) ======
def is_c_hundred(lm):
    # 4 jari melengkung (tip di bawah pip)
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

    # Jempol di bawah telunjuk
    thumb_below = lm[4].y > lm[8].y

    # C menghadap kiri -> jempol di kanan pergelangan
    thumb_left = lm[4].x < lm[3].x

    # Ada rongga C (jarak horizontal jelas)
    gap = abs(lm[4].y - lm[8].y) > 0.07

    return curled and thumb_below and thumb_left and gap
    #return curled and thumb_below and gap

# ====== FINGER STATUS ======
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

# ====== GESTURE MAPPING ======
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

cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    cv2.putText(frame, f"Kamus BIMA | Didanai oleh DPPM Kemdiktisaintek 2026", (10,20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    number = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            if handedness.classification[0].label == "Right":
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                status = finger_status(lm)
                base_number = gesture_to_number(status, lm)

                # ===== STABILIZER =====
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
                    display_number = 0
                    number = stable_number

                    # ===== 1 → C = 100 =====
                    if stable_number == 100 and last_stable_number == 1:
                        display_number = 100
                        number = display_number
                        hold_frames = 25
                        
                    # ===== digit → 👍 → digit =====
                    elif waiting_last_digit and 0 <= stable_number <= 9:
                        display_number = last_stable_number + stable_number
                        number = display_number
                        hold_frames = 25
                        waiting_last_digit = False
                        pending_tens = 0
                        stable_number = 0
                        last_stable_number = 0
                        
                    # ===== digit → 👍 =====
                    elif waiting_last_digit and last_stable_number == 10:
                        display_number = pending_tens * 10 
                        number = display_number
                        hold_frames = 20
                        waiting_last_digit = True
                        
                    elif stable_number == 10 and 1 <= last_stable_number <= 9:
                        pending_tens = last_stable_number
                        waiting_last_digit = True

                    # ===== wave = 11–19 =====
                    elif 1 <= stable_number <= 9:
                        index_x = lm[8].x
                        if detect_wave(index_x):
                            display_number = stable_number + 10
                            number = display_number
                            hold_frames = 25

                if number != 0 and hold_frames == 0:
                    last_stable_number = number
                    display_number = last_stable_number
                    
    else:
        prev_x = None
        prev_dir = 0
        dir_changes = 0
        move_accum = 0
    cv2.putText(frame, f"number:{number}", (5,40),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Last_stable_number:{last_stable_number}", (5,60),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"pending_tens:{pending_tens}", (5,80),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"display_number:{display_number}", (5,100),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
    cv2.putText(frame, f"{number}", (250,350),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 10)

    cv2.imshow("Hand Tracking", frame)
   
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
