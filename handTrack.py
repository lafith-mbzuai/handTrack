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

# Set up the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to capture video")
    exit(1)

# Configure PyAutoGUI
'''
PyAutoGUI has a safety feature where you can stop a script by quickly moving the mouse to a corner of the screen.
This line disables that feature, which is necessary for the program to have full control of the cursor.
'''
pyautogui.FAILSAFE = False

'''
By default, PyAutoGUI adds a tiny pause after every command.
Setting it to zero removes this delay, making the mouse control as responsive as possible.
'''
pyautogui.PAUSE = 0

# Get screen size
screen_width, screen_height = pyautogui.size()

# Define the portion of the camera view to map to the full screen (70% here)
# to avoid jerky movement if hand moves to the edge of camera.
inner_area_percent = 0.7

# Swipe detection parameters
SWIPE_HISTORY_SIZE = 10  # Number of frames to track for swipe
SWIPE_MIN_DISTANCE = 0.25  # Minimum horizontal movement as fraction of frame width
SWIPE_MAX_FRAMES = 15  # Maximum frames for a swipe to complete
SWIPE_VERTICAL_TOLERANCE = 0.15  # Maximum vertical movement as fraction of frame height
SWIPE_COOLDOWN_FRAMES = 20  # Frames to wait before detecting next swipe
OPEN_PALM_THRESHOLD = 1.8  # Threshold for detecting open palm (normalized)

# Calculate the margins around the inner area
def calculate_margins(frame_width, frame_height, inner_area_percent):
    margin_width = frame_width * (1 - inner_area_percent) / 2
    margin_height = frame_height * (1 - inner_area_percent) / 2
    return margin_width, margin_height

# Convert video coordinates to screen coordinates
def convert_to_screen_coordinates(x, y, frame_width, frame_height, margin_width, margin_height):
    # resolution of video captures is different from the screen's resolution.
    screen_x = np.interp(x, (margin_width, frame_width - margin_width), (0, screen_width))
    screen_y = np.interp(y, (margin_height, frame_height - margin_height), (0, screen_height))
    return screen_x, screen_y

# function to get distance between two landmarks
def get_landmark_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# swipe detection
class SwipeDetector:
    def __init__(self, frame_width):
        self.position_history = deque(maxlen=SWIPE_HISTORY_SIZE)
        self.frame_width = frame_width
        self.cooldown_counter = 0
        self.swipe_start_frame = 0
        self.is_swiping = False
        
    def update_frame_width(self, width):
        self.frame_width = width
        
    def add_position(self, x, y, frame_count):
        self.position_history.append((x, y, frame_count))
        
    def clear_history(self):
        self.position_history.clear()
        self.is_swiping = False
        
    def detect_swipe(self, current_frame):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None
            
        if len(self.position_history) < 3:
            return None
            
        # Get positions from history
        positions = list(self.position_history)
        
        # Check if movement is recent enough
        start_frame = positions[0][2]
        if current_frame - start_frame > SWIPE_MAX_FRAMES:
            # Movement too slow, reset
            self.clear_history()
            return None
            
        # Calculate horizontal and vertical displacement
        start_x, start_y = positions[0][0], positions[0][1]
        end_x, end_y = positions[-1][0], positions[-1][1]
        
        horizontal_displacement = (end_x - start_x) / self.frame_width
        vertical_displacement = abs(end_y - start_y) / self.frame_width
        
        # Check if vertical movement is within tolerance
        if vertical_displacement > SWIPE_VERTICAL_TOLERANCE:
            return None
            
        # Check if horizontal movement is sufficient
        if abs(horizontal_displacement) > SWIPE_MIN_DISTANCE:
            # Determine direction
            direction = 'right' if horizontal_displacement > 0 else 'left'
            
            # Set cooldown and clear history
            self.cooldown_counter = SWIPE_COOLDOWN_FRAMES
            self.clear_history()
            
            return direction
            
        return None

# check if hand is in open palm position
def is_open_palm(hand_landmarks, hand_size):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Check if all fingers are extended (far from wrist)
    distances = [
        get_landmark_distance(wrist, index_tip) / hand_size,
        get_landmark_distance(wrist, middle_tip) / hand_size,
        get_landmark_distance(wrist, ring_tip) / hand_size,
        get_landmark_distance(wrist, pinky_tip) / hand_size
    ]
    
    # All fingers should be extended
    status = all(d > OPEN_PALM_THRESHOLD for d in distances)
    return status

# Movement Thread for smoother cursor movement
# this is a seperate thread, running at whatever rate CPU allows(100-1000 iter per sec).
# moves the cursor to the target location set by the main camera loop, (~30 iter per sec).
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
                    # moves cursor faster to target if its far away from the target, slows down when it gets closer.
                    step = max(0.0001, distance / 12)  # Smoother movement
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

# Initialize the movement thread
movement_thread = CursorMovementThread()
movement_thread.start()

# Initialize control variables
mouse_pressed = False
base_touch_threshold = 0.5
base_curl_threshold = 1.5

# Initialize swipe detector
swipe_detector = SwipeDetector(640)  # Default width, will be updated

# Frame counter for swipe detection
frame_count = 0
current_mode = 'neutral'  # Can be 'neutral', 'pointing', 'open_palm'

try:
    previous_y = None
    last_action = 'neutral'

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Flip the frame horizontally for a natural selfie-view, and convert the BGR image to RGB
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Update swipe detector with current frame width
        swipe_detector.update_frame_width(frame.shape[1])

        # Process the frame and find hands
        results = hands.process(frame)

        # Convert the frame color back so it can be displayed
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Check for the presence of hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Calculate margins based on the current frame size
                margin_width, margin_height = calculate_margins(frame.shape[1], frame.shape[0], inner_area_percent)

                # Get all necessary landmarks
                ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                # calculate hand size for normalizing
                hand_size = get_landmark_distance(index_mcp, pinky_mcp) # independant of tip

                # Check if hand is in open palm position for swipe detection
                if is_open_palm(hand_landmarks, hand_size):
                    current_mode = 'open_palm'
                    movement_thread.deactivate()
                    
                    # Track palm center for swipe
                    palm_center_x = (wrist.x + middle_mcp.x) / 2
                    palm_center_y = (wrist.y + middle_mcp.y) / 2
                    
                    # Convert to pixel coordinates
                    palm_x = int(palm_center_x * frame.shape[1])
                    palm_y = int(palm_center_y * frame.shape[0])
                    
                    # Add position to swipe detector
                    swipe_detector.add_position(palm_x, palm_y, frame_count)
                    
                    # Check for swipe
                    swipe_direction = swipe_detector.detect_swipe(frame_count)
                    if swipe_direction:
                        if swipe_direction == 'left':
                            pyautogui.press('left')
                            print("Swipe Left - Left key pressed!")
                        else:
                            pyautogui.press('right')
                            print("Swipe Right - Right key pressed!")
                else:
                    # Not in open palm, clear swipe history
                    if current_mode == 'open_palm':
                        swipe_detector.clear_history()
                    
                    ## 1. Cursor movement (pointing gesture)
                    # curl detection
                    # normalized distance calculation
                    d_wrist_middle_tip = get_landmark_distance(wrist, middle_tip) / hand_size
                    d_wrist_pinky_tip = get_landmark_distance(wrist, pinky_tip) / hand_size
                    d_wrist_ring_tip = get_landmark_distance(wrist, ring_tip) / hand_size
                    d_wrist_index_tip = get_landmark_distance(wrist, index_tip) / hand_size
                    
                    if all(d < base_curl_threshold for d in [
                        d_wrist_ring_tip,
                        d_wrist_pinky_tip,
                        d_wrist_middle_tip
                    ]) and (d_wrist_index_tip > base_curl_threshold):
                        current_mode = 'pointing'
                        movement_thread.activate()
                        
                        index_tip_x = int(index_tip.x * frame.shape[1])
                        index_tip_y = int(index_tip.y * frame.shape[0])
                        # Convert video coordinates to screen coordinates
                        target_x, target_y = convert_to_screen_coordinates(index_tip_x, index_tip_y, frame.shape[1], frame.shape[0], margin_width, margin_height)
                        # Update target position in movement thread
                        movement_thread.update_target(target_x, target_y)
                        
                        ## 2. Left Click (while pointing)
                        # Check if index finger and thumb are touching (for clicking)
                        index_thumb_distance = get_landmark_distance(index_tip, thumb_tip) / hand_size
                        if index_thumb_distance < base_touch_threshold:
                            print("Click!")
                            if not mouse_pressed:
                                pyautogui.mouseDown()
                                mouse_pressed = True
                        else:
                            if mouse_pressed:
                                pyautogui.mouseUp()
                                mouse_pressed = False
                    else:
                        current_mode = 'neutral'
                        movement_thread.deactivate()
                        if mouse_pressed:
                            pyautogui.mouseUp()
                            mouse_pressed = False
        else:
            # no hands detected
            current_mode = 'neutral'
            if mouse_pressed:
                pyautogui.mouseUp()
                mouse_pressed = False
            movement_thread.deactivate()
            swipe_detector.clear_history()

        # # Display current mode on screen (optional - for debugging)
        # cv2.putText(frame, f"Mode: {current_mode}", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # # Show the frame (optional - for debugging)
        # cv2.imshow('Hand Tracking', frame)

        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

finally:
    movement_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
