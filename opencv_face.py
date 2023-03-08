import numpy as np
import dlib
import cv2
import time
import os

"""
Jaw Points = 0–16
Right Brow Points = 17–21
Left Brow Points = 22–26
Nose Points = 27–35
Right Eye Points = 36–41
Left Eye Points = 42–47
Mouth Points = 48–60
Lips Points = 61–67
"""
facial_features = {
    'jaw': (0, 16),
    'right_brow': (17, 21),
    'left_brow': (17, 21),
    'nose': (27, 35),
    'right_eye': (36, 41),
    'left_eye': (42, 47),
    'mouth': (48, 60),
    'lips': (61, 67),
}

# TODO(nick): make flat lookup array of this
def index_to_feature(index):
	for key, value in facial_features.items():
		if index >= value[0] and index <= value[1]:
			return key
	return None

facial_feature_to_color = {
    'jaw': (255, 0, 0),
    'right_brow': (0, 255, 0),
    'left_brow': (0, 0, 255),
    'nose': (255, 255, 0),
    'right_eye': (255, 0, 255),
    'left_eye': (0, 255, 255),
    'mouth': (0, 0, 0),
    'lips': (255, 255, 255),
}

start_time = time.time()

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
img = cv2.imread("ims/dan.png")

# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point

    # Create landmark object
    landmarks = predictor(image=gray, box=face)

    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        # Draw a circle
        feature = index_to_feature(n)
        color = facial_feature_to_color[feature] if feature else (0, 0, 0)

        print(feature, landmarks.part(n))

        cv2.circle(img=img, center=(x, y), radius=3, color=color, thickness=-1)

if not os.path_exists('bin'):
    os.makedirs('bin')

cv2.imwrite("bin/dan.png", img)
# show the image
#cv2.imshow(winname="Face", mat=img)

# Delay between every fram
#cv2.waitKey(delay=0)

# Close all windows
#cv2.destroyAllWindows()

end_time = time.time()

print("Took", (end_time - start_time) * 1000, "ms")

