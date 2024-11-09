import math
import time
import cv2 as cv
import mediapipe as mp

# Mediapipe initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=3, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Mouse things
xp = []
yp = []
mouseX = 0
mouseY = 0
clicks = 0

# UI
drawn = False
alarm_thresh = 150
stop_thresh = 100

currentState = '0'
pastState = '0'

# FPS timing
t1 = time.time()  # Start timing for FPS

# Webcam initialization
cam = cv.VideoCapture(0)  # Use webcam as video source

def click(event, x, y, flags, param):
    global clicks, mouseX, mouseY, drawn
    mouseX = x
    mouseY = y
    if event == cv.EVENT_LBUTTONDOWN:
        if not drawn:
            xp.append(x)
            yp.append(y)
            clicks += 1

cv.namedWindow("camera")
cv.setMouseCallback("camera", click)

# Function to calculate the closest point on a line segment (p1, p2) from a point (cx, cy)
def closest_point_on_segment(p1, p2, cx, cy):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:  # p1 and p2 are the same point
        return p1, math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)

    # Calculate the projection of point (cx, cy) onto the line segment
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:  # Closest to p1
        closest = p1
    elif t > 1:  # Closest to p2
        closest = p2
    else:  # Projection falls on the segment
        closest = (x1 + t * dx, y1 + t * dy)

    # Calculate the distance to the closest point
    distance = math.sqrt((cx - closest[0]) ** 2 + (cy - closest[1]) ** 2)
    return closest, distance

while True:
    # Read frame from webcam
    frameAvailable, frame = cam.read()
    if not frameAvailable:
        print("Error: Could not read frame from webcam.")
        break

    # Calculate FPS
    t2 = time.time()
    fps = 1 / (t2 - t1)
    t1 = t2  # Update for next frame

    # Display FPS on frame
    cv.putText(frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Detection
    imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # Draw lines for the drawn rectangle
    if len(xp) > 1:
        for i in range(len(xp) - 1):
            p1 = (xp[i], yp[i])
            p2 = (xp[i + 1], yp[i + 1])
            cv.line(frame, p1, p2, (0, 0, 255), 2)

    # Complete rectangle
    if len(xp) == 4:
        xp.append(xp[0])
        yp.append(yp[0])
        drawn = True
    elif xp and not drawn:
        # Draw line to mouse position if not completed
        cv.line(frame, (xp[-1], yp[-1]), (mouseX, mouseY), (0, 0, 255), 2)

    if drawn:
        hands_distances = []
        # Checking whether a hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:  # Working with each hand
                closest_distance = float('inf')
                closest_landmark_pos = None
                closest_projection = None

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    mpDraw.draw_landmarks(frame, handLms)

                    # Calculate the minimum distance from this landmark to the shape's edges
                    for i in range(len(xp) - 1):
                        segment_start = (xp[i], yp[i])
                        segment_end = (xp[i + 1], yp[i + 1])
                        projection, distance = closest_point_on_segment(segment_start, segment_end, cx, cy)

                        # Track the closest landmark and projection point on the shape
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_landmark_pos = (cx, cy)
                            closest_projection = projection

                # Determine color based on distance threshold for this specific hand
                if closest_distance <= stop_thresh:
                    arrow_color = (0, 0, 255)  # red
                elif closest_distance <= alarm_thresh:
                    arrow_color = (0, 255, 255)  # yellow
                else:
                    arrow_color = (0, 255, 0)  # green

                # Draw the line from the closest landmark to the closest projection point on the shape
                if closest_landmark_pos and closest_projection:
                    cv.arrowedLine(frame, closest_landmark_pos, (int(closest_projection[0]), int(closest_projection[1])), arrow_color, 2)
                    Distance = closest_distance

                    # Print distance
                    cv.putText(frame, 'Distance: ' + str(int(Distance)), closest_landmark_pos, cv.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 3)

                hands_distances.append(closest_distance)

        # Set state based on minimum distance among all hands, if hands were detected
        if hands_distances:
            if min(hands_distances) <= stop_thresh:
                currentState = '2'
            elif min(hands_distances) <= alarm_thresh:
                currentState = '1'
            else:
                currentState = '0'
        else:
            currentState = '0'  # No hands detected, set state to '0'


    # Send on change
    if currentState != pastState:
        print('send: ' + currentState)
        pastState = currentState

    cv.imshow('camera', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
