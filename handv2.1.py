import cv2
import mediapipe as mp

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

# ====== ALL GLOBAL VARIABLES ======
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

digit_list = []
calc_mode = False
current_operator = None
result = 0
calc_display = ""
math_step = 0

clear_stable_count = 0
CLEAR_STABLE_MIN = 12
clear_hold_frames = 0

# ====== GESTURE FUNCTIONS ======
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
    status = [0,0,0,0,0]
    palm_view = lm[5].x < lm[17].x
    if palm_view:
        if lm[4].x < lm[3].x: status[0] = 1
    else:
        if lm[4].x > lm[3].x: status[0] = 1
    tips = [8,12,16,20]
    pips = [6,10,14,18]
    for i,(tip,pip) in enumerate(zip(tips,pips), start=1):
        if lm[tip].y < lm[pip].y: status[i] = 1
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
    return (abs(right_lm[8].x - left_lm[0].x) < 0.12 and 
            abs(right_lm[8].y - left_lm[0].y) < 0.12 and 
            left_lm[17].x > left_lm[5].x)

def detect_division(right_lm, left_lm):
    return (left_lm[0].z < left_lm[9].z and 
            abs(right_lm[8].x - right_lm[5].x) < 0.08 and 
            right_lm[8].y < left_lm[0].y + 0.1)

def detect_equals(lm):
    return (lm[4].y < lm[2].y and lm[20].y < lm[18].y and 
            lm[8].y > lm[6].y and lm[12].y > lm[10].y and lm[16].y > lm[14].y)

# ====== MAIN LOOP ======
cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindow
