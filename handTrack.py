import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
from collections import deque

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# for visualization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    #print("Failed to capture video")
    exit(1)

# Configure PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Get screen size
screen_width, screen_height = pyautogui.size()

# Define the portion of the camera view to map to the full screen (70% here)
inner_area_percent = 0.7

def calculate_margins(frame_width, frame_height, inner_area_percent):
    margin_width = frame_width * (1 - inner_area_percent) / 2
    margin_height = frame_height * (1 - inner_area_percent) / 2
    return margin_width, margin_height

def convert_to_screen_coordinates(x, y, frame_width, frame_height, margin_width, margin_height):
    screen_x = np.interp(x, (margin_width, frame_width - margin_width), (0, screen_width))
    screen_y = np.interp(y, (margin_height, frame_height - margin_height), (0, screen_height))
    return screen_x, screen_y

def get_landmark_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return np.hypot(x2 - x1, y2 - y1)

# Movement Thread
class CursorMovementThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_x, self.current_y = pyautogui.position()
        self.target_x, self.target_y = self.current_x, self.current_y
        self.running = True
        self.active = False
        self.jitter_threshold = 0.003

    def run(self):
        while self.running:
            if self.active:
                distance = np.hypot(self.target_x - self.current_x, self.target_y - self.current_y)
                screen_diagonal = np.hypot(screen_width, screen_height)
                if distance / screen_diagonal > self.jitter_threshold:
                    step = max(0.0001, distance / 12)
                    if distance != 0:
                        step_x = (self.target_x - self.current_x) / distance * step
                        step_y = (self.target_y - self.current_y) / distance * step
                        self.current_x += step_x
                        self.current_y += step_y
                        pyautogui.moveTo(self.current_x, self.current_y, _pause=False)
                time.sleep(0)
            else:
                time.sleep(0.1)

    def update_target(self, x, y):
        self.target_x, self.target_y = x, y

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def stop(self):
        self.running = False

movement_thread = CursorMovementThread()
movement_thread.start()

# Control variables
mouse_pressed = False

# Existing thresholds
base_touch_threshold = 0.3
base_curl_threshold = 1.5
base_swipe_threshold = 0.20       # normalized X delta in [0,1]
base_neutral_margin = 0.01

# Swipe detector parameters
SWIPE_WINDOW_SEC = 0.35           # lookback window to measure displacement (secs)
SWIPE_VEL_THRESH = 0.8            # normalized width per second
SWIPE_COOLDOWN = 1.0              # seconds between swipes
last_swipe_time = 0.0
swipe_buffer = deque()            # (t, x) pairs while open palm is held

# for pinch hysteresis
PINCH_DOWN = 0.58 
PINCH_UP   = 0.74
PINCH_DEBOUNCE = 0.07
last_pinch_event = 0.0

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            movement_thread.activate()

            for hand_landmarks in results.multi_hand_landmarks:

                # draw the hand landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
                margin_width, margin_height = calculate_margins(frame.shape[1], frame.shape[0], inner_area_percent)

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                # Hand size for normalization
                hand_size = get_landmark_distance(index_mcp, pinky_mcp)
                if hand_size < 1e-6:
                    continue

                # Distances (normalized by hand_size)
                d_wrist_middle_tip = get_landmark_distance(wrist, middle_tip) / hand_size
                d_wrist_pinky_tip  = get_landmark_distance(wrist, pinky_tip)  / hand_size
                d_wrist_ring_tip   = get_landmark_distance(wrist, ring_tip)   / hand_size
                d_wrist_index_tip  = get_landmark_distance(wrist, index_tip)  / hand_size

                # cursor movement when index extended and others curled
                is_curl = all(d < base_curl_threshold for d in [
                    d_wrist_ring_tip, d_wrist_pinky_tip, d_wrist_middle_tip
                ])
                if is_curl and (d_wrist_index_tip > base_curl_threshold):
                    index_tip_x = int(index_tip.x * frame.shape[1])
                    index_tip_y = int(index_tip.y * frame.shape[0])
                    target_x, target_y = convert_to_screen_coordinates(
                        index_tip_x, index_tip_y, frame.shape[1], frame.shape[0], margin_width, margin_height
                    )
                    movement_thread.update_target(target_x, target_y)

                # open palm detection
                open_palm = all(d > base_curl_threshold for d in [
                    d_wrist_index_tip, d_wrist_middle_tip, d_wrist_ring_tip, d_wrist_pinky_tip
                ])

                # left-click on pinch
                # needs to be open palm to avoid click on curl
                pinch = get_landmark_distance(middle_tip, thumb_tip) / hand_size
                now = time.time()
                if (not mouse_pressed) and open_palm and (pinch < PINCH_DOWN) and (now - last_pinch_event > PINCH_DEBOUNCE):
                    print(f"Mouse Down: {frame_count} : open palm - {open_palm}, pinch - {pinch}:{PINCH_DOWN}/{PINCH_UP}")
                    pyautogui.mouseDown()
                    mouse_pressed = True
                    last_pinch_event = now
                elif mouse_pressed and (pinch > PINCH_UP) and (now - last_pinch_event > PINCH_DEBOUNCE):
                    print(f"Mouse UP: {frame_count} : open palm - {open_palm}, pinch - {pinch}:{PINCH_DOWN}/{PINCH_UP}")
                    pyautogui.mouseUp()
                    mouse_pressed = False
                    last_pinch_event = now

                t_now = time.time()
                if open_palm:
                    hand_x = float(middle_mcp.x)
                    swipe_buffer.append((t_now, hand_x))

                    # Drop old samples
                    while swipe_buffer and (t_now - swipe_buffer[0][0] > SWIPE_WINDOW_SEC):
                        swipe_buffer.popleft()

                    # check displacement
                    if len(swipe_buffer) >= 2:
                        t0, x0 = swipe_buffer[0]
                        t1, x1 = swipe_buffer[-1]
                        dt = max(1e-6, t1 - t0)
                        dx = x1 - x0
                        speed = dx / dt

                        can_fire = (t_now - last_swipe_time) > SWIPE_COOLDOWN
                        if can_fire and abs(dx) >= base_swipe_threshold and abs(speed) >= SWIPE_VEL_THRESH:
                            if dx > 0:
                                pyautogui.press('right')
                                print(f"Swipe Right: {frame_count} : {dx}/{base_swipe_threshold}")
                            else:
                                pyautogui.press('left')
                                print(f"Swipe Left: {frame_count} : {dx}/{base_swipe_threshold}")
                            last_swipe_time = t_now
                            swipe_buffer.clear()  # reset window to avoid repeats
                else:
                    # Not open palm → reset swipe window gradually
                    if swipe_buffer:
                        # keep tiny buffer so small gaps don't kill the gesture
                        if (t_now - swipe_buffer[-1][0]) > 0.15:
                            swipe_buffer.clear()

        else:
            if mouse_pressed:
                pyautogui.mouseUp()
                mouse_pressed = False
            movement_thread.deactivate()
            swipe_buffer.clear()

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    movement_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
