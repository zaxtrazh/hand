# Deteksi 1-19

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
wave_cooldown = 0

def detect_wave(index_x):
    global prev_x, prev_dir, dir_changes, wave_cooldown

    if wave_cooldown > 0:
        wave_cooldown -= 1
        return False

    if prev_x is None:
        prev_x = index_x
        return False

    dx = index_x - prev_x
    prev_x = index_x

    if abs(dx) < 0.005:
        return False

    direction = 1 if dx > 0 else -1

    if prev_dir == 0:
        prev_dir = direction
        return False

    if direction != prev_dir:
        dir_changes += 1
        prev_dir = direction

    if dir_changes >= 2:
        dir_changes = 0
        wave_cooldown = 20
        return True

    return False

# ====== THUMB UP DETECTOR (10) ======
def is_thumb_up(lm):
    mcp = lm[2]
    tip = lm[4]

    vx = tip.x - mcp.x
    vy = tip.y - mcp.y

    return vy < -0.08 and abs(vx) < 0.12

# ====== FINGER STATUS ======
def finger_status(lm):
    status = [0,0,0,0,0]  # [T,I,M,R,P]

    palm_view = lm[5].x < lm[17].x

    # Thumb pakai X dibalik otomatis
    if palm_view:
        if lm[4].x < lm[3].x:
            status[0] = 1
    else:
        if lm[4].x > lm[3].x:
            status[0] = 1

    # 4 jari lain pakai Y
    tips = [8,12,16,20]
    pips = [6,10,14,18]

    for i,(tip,pip) in enumerate(zip(tips,pips), start=1):
        if lm[tip].y < lm[pip].y:
            status[i] = 1

    return status

# ====== GESTURE MAPPING ======
def gesture_to_number(s, lm):
    T,I,M,R,P = s

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

    return 0

# ====== WINDOW ======
cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ====== MAIN LOOP ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    number = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            if handedness.classification[0].label == "Right":
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                lm = hand_landmarks.landmark
                status = finger_status(lm)

                base_number = gesture_to_number(status, lm)
                number = base_number

                # ==== WAVE ⇒ 11–19 ====
                if 1 <= base_number <= 9:
                    index_x = lm[8].x
                    if detect_wave(index_x):
                        number = base_number + 10

    cv2.putText(frame, f"{number}", (250,200),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 10)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
