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
color = (0, 255, 0)
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
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    mpDraw.draw_landmarks(frame, handLms)
                    if id == 9:
                        lm9x, lm9y = cx, cy
                    elif id == 0:
                        lm0x, lm0y = cx, cy

                pDis = []
                # Calculate distances from hand landmark (lm9) to all points
                for i in range(len(xp) - 1):
                    # Calculate distance between the current hand landmark and the drawn point
                    dist = math.sqrt((xp[i] - lm9x)**2 + (yp[i] - lm9y)**2)
                    pDis.append(dist)
                    cv.circle(frame, (xp[i], yp[i]), 1, (0, 0, 0), 4)
                    cv.putText(frame, str(i), (xp[i], yp[i]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                # Find the closest point
                closest_point_idx = pDis.index(min(pDis))

                # Draw the line from the hand landmark to the closest point
                cv.arrowedLine(frame, (lm9x, lm9y), (xp[closest_point_idx], yp[closest_point_idx]), color, 2)
                Distance = min(pDis)

                # Print distance
                cv.putText(frame, 'Distance: ' + str(int(Distance)), (lm9x, lm9y), cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                hands_distances.append(Distance)

                if Distance <= stop_thresh:
                    color = (0, 0, 255)  # red
                elif Distance <= alarm_thresh:
                    color = (0, 255, 255)  # yellow
                else:
                    color = (0, 255, 0)  # green

            if min(hands_distances) <= stop_thresh:
                currentState = '2'
            elif min(hands_distances) <= alarm_thresh:
                currentState = '1'
            else:
                currentState = '0'
        else:
            currentState = '0'

    # Send on change
    if currentState != pastState:
        print('send: ' + currentState)
        pastState = currentState

    cv.imshow('camera', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
