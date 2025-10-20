import time
import pyautogui
import enum
import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from transitions import Machine
import threading


# configure PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
# Get screen size
screen_width, screen_height = pyautogui.size()

# define the portion of the camera view to map to the full screen (70% here)
inner_area_percent = 0.7
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing utilities
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_hands = solutions.hands

# Global variable to store latest results for visualization
latest_result = None
SWIPE_COOLDOWN = 1.0
CLICK_COOLDOWN = 0.3

# SCROLL_THRESHOLD = 0.005 
# SCROLL_SENSITIVITY = 0.05

last_swipe_time = 0.0
last_click_time = 0.0
swipe_entry_x = None
just_swiped = False
# scroll_previous_y = None

# Initialize state machin
class States(enum.Enum):
    IDLE = 0
    POINT = 1
    SWIPE = 2
    # SCROLL = 3

transitions = [
    ['point_up_detected', States.IDLE, States.POINT],
    ['open_palm_detected', States.IDLE, States.SWIPE],
    # ['fist_detected', States.IDLE, States.SCROLL],
    ['nothing', States.POINT, States.IDLE],
    # ['nothing', States.SCROLL, States.IDLE],
    ['nothing', States.SWIPE, States.IDLE]]

machine = Machine(states=States, transitions=transitions, initial=States.IDLE)

def calculate_margins(frame_width, frame_height, inner_area_percent):
    margin_width = frame_width * (1 - inner_area_percent) / 2
    margin_height = frame_height * (1 - inner_area_percent) / 2
    return margin_width, margin_height

def convert_to_screen_coordinates(x, y, frame_width, frame_height, margin_width, margin_height):
    screen_x = np.interp(x, (margin_width, frame_width - margin_width), (0, screen_width))
    screen_y = np.interp(y, (margin_height, frame_height - margin_height), (0, screen_height))
    return screen_x, screen_y

def detect_pinch(landmarks, hand_size):
    """Detect pinch gesture between thumb and index finger"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    distance = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    pinch_threshold = hand_size * 0.20  # 15% of hand size
    pinched = bool(distance < pinch_threshold)
    # print(distance, pinch_threshold , pinched)
    return pinched 

def calculate_hand_size(landmarks):
    """Calculate hand size as distance between wrist and middle finger tip"""
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    return np.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)

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

# scroll thread
# class ScrollThread(threading.Thread):
#     def __init__(self):
#         super().__init__()
#         self.daemon = True
#         self.scroll_queue = []
#         self.scroll_lock = threading.Lock()
#         self.running = True
#         self.inertia = 0.95  # Slower reduction for rolling stop effect
#         self.scroll_step = 0.01  # Smaller step for smoother scroll
#         self.inertia_threshold = 0.01  # Minimum inertia scroll amount

#     def run(self):
#         while self.running:
#             if self.scroll_queue:
#                 with self.scroll_lock:
#                     scroll_amount = self.scroll_queue.pop(0)
#                 pyautogui.scroll(scroll_amount)
#                 # Apply inertia effect if the queue is empty
#                 if len(self.scroll_queue) == 0 and abs(scroll_amount) > self.inertia_threshold:
#                     scroll_amount *= self.inertia
#                     if abs(scroll_amount) > self.scroll_step:
#                         with self.scroll_lock:
#                             self.scroll_queue.append(scroll_amount)
#             time.sleep(0.005)  # Increased frequency for smoother processing

#     def add_scroll(self, scroll_amount):
#         with self.scroll_lock:
#             self.scroll_queue.append(scroll_amount)

#     def stop(self):
#         self.running = False


# scroll_thread = ScrollThread()
movement_thread = CursorMovementThread()
movement_thread.start()
# scroll_thread.start()

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on the image"""
    annotated_image = np.copy(rgb_image)
    
    # Check if we have hand landmarks
    if detection_result.hand_landmarks:
        # Iterate through each detected hand
        for hand_landmarks in detection_result.hand_landmarks:
            # Convert to MediaPipe landmark proto format for drawing
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                ) for landmark in hand_landmarks
            ])
            
            # Draw the hand landmarks on the image
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return annotated_image
def draw_gesture_info_on_frame(frame, result, cap):
    """Draw gesture labels and FPS on the annotated frame"""
    annotated_frame = frame.copy()
    
    # Add gesture labels on the frame
    if result.gestures:
        y_position = 30
        for i, gesture in enumerate(result.gestures):
            if gesture:
                top_gesture = gesture[0]
                text = f'Hand {i}: {top_gesture.category_name} ({top_gesture.score:.2f})'
                
                # Draw background rectangle for better text visibility
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(annotated_frame, 
                            (10, y_position - text_height - 5),
                            (15 + text_width, y_position + 5),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(annotated_frame, text,
                          (10, y_position),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          (0, 255, 0), 2, cv2.LINE_AA)
                y_position += 35
    
    # Display FPS
    fps_text = f'FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}'
    cv2.putText(annotated_frame, fps_text,
               (annotated_frame.shape[1] - 100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
               (255, 255, 255), 2, cv2.LINE_AA)
    
    return annotated_frame

# Create a gesture recognizer instance with the live stream mode:
def process_detection(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback function to process gesture recognition results"""
    global latest_result, last_swipe_time, last_click_time, swipe_entry_x, just_swiped#, scroll_previous_y
    latest_result = result


    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        hand_size = calculate_hand_size(landmarks)
        is_pinched = detect_pinch(landmarks, hand_size)

        if machine.state == States.SWIPE and is_pinched:
            t_now = time.time()
            if (t_now - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = t_now
                print(f"{machine.state} - Left click executed")
            return # dont do gesture processing if pinch
        

    if result.gestures:
        # Get the top gesture for each detected hand
        for i, gesture in enumerate(result.gestures):
            if gesture:
                top_gesture = gesture[0]
                curr_g = top_gesture.category_name

                if curr_g == "Pointing_Up" and machine.state != States.POINT:
                    if machine.state != States.IDLE:
                        machine.nothing() 
                    machine.point_up_detected()
                    swipe_entry_x = None
                elif curr_g == "Open_Palm" and machine.state != States.SWIPE:
                    if machine.state != States.IDLE:
                        machine.nothing()
                    machine.open_palm_detected()

                # elif curr_g == "Closed_Fist" and machine.state != States.SCROLL:
                #     if machine.state != States.IDLE:
                #         machine.nothing()
                #     machine.fist_detected()
                #     swipe_entry_x = None
                #     scroll_previous_y = None
                #     print(f"{machine.state} - Scroll mode activated")

                elif curr_g == "None" and machine.state in [States.SWIPE]:
                    machine.nothing()
                    swipe_entry_x = None


    if machine.state == States.POINT and result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        index_tip = landmarks[8]

        index_tip_x = int(index_tip.x * output_image.width)
        index_tip_y = int(index_tip.y * output_image.height)

        margin_width, margin_height = calculate_margins(output_image.width, output_image.height, inner_area_percent)
        target_x, target_y = convert_to_screen_coordinates(
            index_tip_x, index_tip_y, output_image.height, output_image.width, margin_width, margin_height
        )
        movement_thread.update_target(target_x, target_y)

    if machine.state == States.SWIPE and result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        hand_size = calculate_hand_size(landmarks)

        # lateral move check
        middle_mcp = landmarks[9]
        current_x = middle_mcp.x

        if swipe_entry_x is None:
            swipe_entry_x = current_x
            just_swiped = False
        else:
            dx = current_x - swipe_entry_x
            t_now = time.time()
            swipe_threshold = hand_size * 0.3

            # reset the "just swiped" flag if hand returns near center
            if just_swiped and abs(dx) < swipe_threshold * 0.3:
                just_swiped = False

            if (t_now - last_swipe_time) > SWIPE_COOLDOWN and not just_swiped:
                if dx > swipe_threshold:
                    pyautogui.press('right')
                    last_swipe_time = t_now
                    just_swiped = True
                    print(f"{machine.state} - Swipe detected: Right arrow clicked.")
                elif dx < -swipe_threshold:
                    pyautogui.press('left')
                    last_swipe_time = t_now
                    just_swiped = True
                    print(f"{machine.state} - Swipe detected: left arrow clicked.")

    # if machine.state == States.SCROLL and result.hand_landmarks:
    #     landmarks = result.hand_landmarks[0]
    #     wrist = landmarks[0]
    #     current_y = wrist.y

    #     if scroll_previous_y is not None:
    #         delta_y = current_y - scroll_previous_y
    #         if abs(delta_y) > SCROLL_THRESHOLD:
    #             # Negative delta_y because hand moving up should scroll up
    #             scroll_amount = -delta_y * screen_height * SCROLL_SENSITIVITY
    #             scroll_thread.add_scroll(scroll_amount)

    #     scroll_previous_y = current_y


# Initialize options for gesture recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),  # Update path to your model
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,  # Maximum number of hands to detect
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=process_detection)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to video file path
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create gesture recognizer
with GestureRecognizer.create_from_options(options) as recognizer:
    print("Gesture recognizer initialized. Press 'q' to quit.")

    movement_thread.activate()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Convert BGR image to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Get current timestamp in milliseconds
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # Perform gesture recognition on the frame
        # In LIVE_STREAM mode, this is async and results go to callback
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Draw landmarks on the frame if we have results
        if latest_result:
            annotated_frame = draw_landmarks_on_image(rgb_frame, latest_result)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            annotated_frame = draw_gesture_info_on_frame(annotated_frame, latest_result, cap)
            cv2.imshow('Gesture Recognition with Hand Landmarks', annotated_frame)
        else:
            cv2.imshow('Gesture Recognition with Hand Landmarks', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    movement_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture recognition stopped.")
