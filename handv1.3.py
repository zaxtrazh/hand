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

def finger_status(lm):
    status = [0,0,0,0,0]  # [T,I,M,R,P]

    # === DETEKSI TELAPAK / PUNGGUNG ===
    palm_view = lm[5].x < lm[17].x

    # ===== THUMB (sumbu X dibalik jika telapak) =====
    if palm_view:
        if lm[4].x < lm[3].x:
            status[0] = 1
    else:
        if lm[4].x > lm[3].x:
            status[0] = 1

    # ===== 4 jari lain (tetap pakai Y, ini sudah benar) =====
    tips = [8,12,16,20]
    pips = [6,10,14,18]

    for i,(tip,pip) in enumerate(zip(tips,pips), start=1):
        if lm[tip].y < lm[pip].y:
            status[i] = 1

    return status

def gesture_to_number(s):
    T,I,M,R,P = s

    if [T,I,M,R,P] == [0,1,0,0,0]: return 1
    if [T,I,M,R,P] == [0,1,1,0,0]: return 2
    if [T,I,M,R,P] == [1,1,1,0,0]: return 3
    if [T,I,M,R,P] == [0,1,1,1,1]: return 4
    if [T,I,M,R,P] == [1,1,1,1,1]: return 5
    if [T,I,M,R,P] == [0,1,1,1,0]: return 6
    if [T,I,M,R,P] == [0,1,1,0,1]: return 7
    if [T,I,M,R,P] == [0,1,0,1,1]: return 8
    if [T,I,M,R,P] == [0,0,1,1,1]: return 9
    if [T,I,M,R,P] == [1,0,0,0,0]: return 10

    return 0

cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
                number = gesture_to_number(status)

    cv2.putText(frame, f"{number}", (250,200),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 10)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
