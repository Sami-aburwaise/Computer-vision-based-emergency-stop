import math
import time
import cv2 as cv
import mediapipe as mp

# Mediapipe initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=3, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Mouse coordinates for drawing rectangle
xp, yp = [], []
mouseX, mouseY = 0, 0
clicks, drawn = 0, False

# Thresholds
alarm_thresh, stop_thresh = 150, 100
currentState, pastState = '0', '0'

# Webcam initialization
cam = cv.VideoCapture(0)  # Use webcam as video source

# Boolean flag to control drawing of landmarks and connections
draw_landmarks = True  # Set to False to disable drawing landmarks and connections

# Function to register mouse click for drawing rectangle points
def click(event, x, y, flags, param):
    global clicks, mouseX, mouseY, drawn
    if event == cv.EVENT_LBUTTONDOWN and not drawn:
        xp.append(x)
        yp.append(y)
        clicks += 1

cv.namedWindow("camera")
cv.setMouseCallback("camera", click)

# Calculate closest point on segment
def closest_point_on_segment(p1, p2, cx, cy):
    x1, y1, x2, y2 = *p1, *p2
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return p1, math.dist(p1, (cx, cy))
    t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
    closest = (x1 + t * dx, y1 + t * dy)
    return closest, math.dist(closest, (cx, cy))

# Initialize FPS timing before the loop
t1 = time.time()

while True:
    frameAvailable, frame = cam.read()
    if not frameAvailable:
        print("Error: Could not read frame from webcam.")
        break

    # Calculate FPS
    t2 = time.time()
    fps = int(1 / (t2 - t1))
    t1 = t2  # Update t1 to the current time for the next iteration

    # Display FPS on frame
    cv.putText(frame, f'FPS: {fps}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Process frame for hand landmarks
    imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # Draw the lines for drawn rectangle
    if len(xp) > 1:
        for i in range(len(xp) - 1):
            cv.line(frame, (xp[i], yp[i]), (xp[i + 1], yp[i + 1]), (0, 0, 255), 2)

    if len(xp) == 4 and not drawn:
        xp.append(xp[0])
        yp.append(yp[0])
        drawn = True
    elif xp and not drawn:
        cv.line(frame, (xp[-1], yp[-1]), (mouseX, mouseY), (0, 0, 255), 2)

    hands_distances = []
    if drawn and results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            closest_distance, closest_landmark_pos, closest_projection = float('inf'), None, None
            h, w, _ = frame.shape

            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Calculate distance from the current landmark to each segment
                for i in range(len(xp) - 1):
                    segment_start = (xp[i], yp[i])
                    segment_end = (xp[i + 1], yp[i + 1])
                    projection, distance = closest_point_on_segment(segment_start, segment_end, cx, cy)

                    # Update closest distance and projection point if closer
                    if distance < closest_distance:
                        closest_distance, closest_landmark_pos, closest_projection = distance, (cx, cy), projection

            # Set arrow color based on closest distance
            arrow_color = (0, 255, 0) if closest_distance > alarm_thresh else \
                          (0, 255, 255) if closest_distance > stop_thresh else (0, 0, 255)

            # Draw arrow from closest landmark to closest point on the rectangle
            if closest_landmark_pos and closest_projection:
                cv.arrowedLine(frame, closest_landmark_pos, tuple(map(int, closest_projection)), arrow_color, 2)
                cv.putText(frame, f'Distance: {int(closest_distance)}', closest_landmark_pos, cv.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)

            hands_distances.append(closest_distance)

            # Drawing landmarks and connections
            if draw_landmarks:
                # Draw landmarks as small circles without the numbers
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a circle at each landmark

                # Draw connections between landmarks (lines between adjacent landmarks)
                connections = mpHands.HAND_CONNECTIONS  # This contains all the connections between landmarks
                for start, end in connections:
                    start_x, start_y = int(handLms.landmark[start].x * w), int(handLms.landmark[start].y * h)
                    end_x, end_y = int(handLms.landmark[end].x * w), int(handLms.landmark[end].y * h)
                    cv.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Update current state based on minimum distance across all hands
    if hands_distances:
        min_distance = min(hands_distances)
        currentState = '2' if min_distance <= stop_thresh else '1' if min_distance <= alarm_thresh else '0'
    else:
        currentState = '0'

    # Send state update if changed
    if currentState != pastState:
        print(f'send: {currentState}')
        pastState = currentState

    cv.imshow('camera', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
