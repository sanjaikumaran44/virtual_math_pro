import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
st.image('MathGestures.png')

# Color options with their RGB values
color_options = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "PINK": (255, 0, 255),
    "ORANGE": (0, 165, 255)
}

col1, col2 = st.columns([3, 2])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

    # Add instructions
    st.markdown("""
    **Gesture Controls:**
    - Index finger up: Draw
    - Thumb up: Erase
    - All fingers up (thumb + index + middle + ring + pinky): Erase All
    - All fingers up except pinky (thumb + index + middle + ring): Solve equation
    - Click on color buttons to change drawing color
    """)

genai.configure(api_key="AIzaSyAVMjytGdvg02j_CyfyvMa7NZG2wt6EZzI")
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

# Set fixed large canvas size
width, height = 2000, 2000
cap.set(3, width)
cap.set(5, height)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


# Create color buttons
def create_color_buttons(img):
    button_width = 150
    button_height = 60
    spacing = 10
    y_position = 20
    buttons = []

    # First, create CLEAR button with black outline
    x_position = 40
    cv2.rectangle(img, (x_position, y_position),
                  (x_position + button_width, y_position + button_height),
                  (0, 0, 0), 2)
    cv2.putText(img, "CLEAR", (x_position + 40, y_position + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    buttons.append(("CLEAR", x_position, y_position, x_position + button_width, y_position + button_height))

    # Then create color buttons
    for i, (color_name, color_value) in enumerate(color_options.items()):
        x_position = 40 + (button_width + spacing) * (i + 1)

        # Draw rectangle with color fill
        cv2.rectangle(img, (x_position, y_position),
                      (x_position + button_width, y_position + button_height),
                      color_value, -1)

        # Add border in color
        cv2.rectangle(img, (x_position, y_position),
                      (x_position + button_width, y_position + button_height),
                      (0, 0, 0), 2)

        # Add text
        cv2.putText(img, color_name, (x_position + 30, y_position + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        buttons.append((color_name, x_position, y_position, x_position + button_width, y_position + button_height))

    return buttons


def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=True)  # Set draw=True to display hand landmarks

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None


def draw(info, prev_pos, canvas, draw_color):
    fingers, lmList = info
    current_pos = None
    mode = ""
    clear_canvas = False

    # Drawing mode - index finger up
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, draw_color, 10)
        mode = "draw"

    # Eraser mode - thumb up
    elif fingers == [1, 0, 0, 0, 0]:
        current_pos = lmList[4][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.circle(canvas, current_pos, 30, (0, 0, 0), -1)
        mode = "erase"

    # Clear all - all fingers up
    elif fingers == [0, 0, 1, 1, 1]:
        clear_canvas = True
        mode = "clear all"

    return current_pos, canvas, mode, clear_canvas


def check_button_click(lmList, buttons):
    # Check if index finger is clicking a button
    x, y = lmList[8][0:2]  # Index finger tip position

    for button_name, x1, y1, x2, y2 in buttons:
        if x1 < x < x2 and y1 < y < y2:
            return button_name

    return None


def sendToAI(model, canvas, fingers):
    # All fingers up except pinky to solve
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return None


prev_pos = None
canvas = None
image_combined = None
output_text = ""
current_color = color_options["BLUE"]  # Default color

# Continuously get frames from the webcam
while run:
    # Capture each frame from the webcam
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from camera. Please check your camera connection.")
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    # Create color selection buttons
    buttons = create_color_buttons(img)

    info = getHandInfo(img)
    mode = ""
    clear_canvas = False

    if info:
        fingers, lmList = info

        # Check for button clicks when index finger is up
        if fingers == [0, 1, 0, 0, 0]:
            button_clicked = check_button_click(lmList, buttons)
            if button_clicked:
                if button_clicked == "CLEAR":
                    canvas = np.zeros_like(img)
                else:
                    current_color = color_options[button_clicked]
                # Skip drawing for this frame to avoid drawing on buttons
                prev_pos = None
                continue

        prev_pos, canvas, mode, clear_canvas = draw(info, prev_pos, canvas, current_color)

        # Clear canvas if indicated
        if clear_canvas:
            canvas = np.zeros_like(img)

        # Check if solving is requested - All fingers up except pinky
        if fingers == [1, 1, 1, 1, 0]:
            mode = "solving"
            # Display solving status
            cv2.putText(img, "Solving...", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            result = sendToAI(model, canvas, fingers)
            if result:
                output_text = result
    else:
        prev_pos = None  # Reset position when hand not detected

    # Display current mode on screen if active
    if mode == "draw":
        cv2.putText(img, "Mode: DRAWING", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif mode == "erase":
        cv2.putText(img, "Mode: ERASING", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif mode == "clear all":
        cv2.putText(img, "Mode: CLEAR ALL", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif mode == "solving":
        cv2.putText(img, "Mode: SOLVING", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show current color indicator
    cv2.putText(img, "Current Color:", (width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(img, (width - 150, 20), (width - 50, 60), current_color, -1)
    cv2.rectangle(img, (width - 150, 20), (width - 50, 60), (0, 0, 0), 2)

    # Combine original image with canvas
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.6, 0)

    # Display the hand position and gesture info
    if info:
        fingers, lmList = info
        gesture_text = f"Fingers up: {fingers}"
        cv2.putText(image_combined, gesture_text, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Add a short delay to reduce CPU usage
    cv2.waitKey(1)