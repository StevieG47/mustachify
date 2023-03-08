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

def distance(p0, p1):
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

# startup stuff
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image_name = "sa.png"

img = cv2.imread("ims/" + image_name)

# actual timing starts here
start_time = time.time()

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    p0 = (face.left(), face.top())
    p1 = (face.right(), face.bottom())

    #cv2.rectangle(img, p0, p1, (0, 255, 0), 1)

    landmarks = predictor(image=gray, box=face)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        feature = index_to_feature(n)
        color = facial_feature_to_color[feature] if feature else (0, 0, 0)
        cv2.circle(img=img, center=(x, y), radius=3, color=color, thickness=-1)

    # TODO(nick): maybe consider a few different feature points

    nose_p_32 = (landmarks.part(32).x, landmarks.part(32).y)
    nose_p_34 = (landmarks.part(34).x, landmarks.part(34).y)

    mouth_p_48 = (landmarks.part(48).x, landmarks.part(48).y)
    mouth_p_49 = (landmarks.part(49).x, landmarks.part(49).y)
    mouth_p_53 = (landmarks.part(53).x, landmarks.part(53).y)
    mouth_p_54 = (landmarks.part(54).x, landmarks.part(54).y)

    mouth_left = mouth_p_49
    mouth_right = mouth_p_53
    nose = (np.mean((nose_p_32[0], nose_p_34[0])), np.mean((nose_p_32[1], nose_p_34[1])))


    # X coord between left/right mouth points
    center_x = np.mean((mouth_left[0],mouth_right[0]))

    # 2d distance between mouth points, use as mask rectangle width
    #mouth_width = np.sqrt((mouth_left[0]-mouth_right[0])**2 + (mouth_left[1]-mouth_right[1])**2)
    mouth_width = distance(mouth_p_48, mouth_p_54)

    # Get distance from nose to middle of mouth, use as mask rectangle height
    max_mouth_y = np.max((mouth_left[1],mouth_right[1]))
    min_mouth_y = np.min((mouth_left[1],mouth_right[1]))
    if nose[1] < min_mouth_y:
        nose_top_mouth_y_dist = abs(nose[1]-min_mouth_y) # right side up face
    else:
        nose_top_mouth_y_dist = abs(nose[1]-max_mouth_y) # upside down face
    nose_to_mouth_dist = nose_top_mouth_y_dist + 0.5*(max_mouth_y-min_mouth_y) # ydist from nose to mouth

    # Y coord, set as halfway from nose to mouth
    if nose[1] < min_mouth_y:
        center_y = nose[1] + nose_to_mouth_dist/2
    else:
        center_y = nose[1] - nose_to_mouth_dist/2

    # Set rectangle properties for rectangle moustache mask
    # points are x, y
    rect_center = (center_x, center_y)
    rect_len_width = (mouth_width, nose_to_mouth_dist)
    rect_angle = np.arctan2(max_mouth_y-min_mouth_y, abs(mouth_left[0]-mouth_right[0]))*180/np.pi # rotate cw off x axis
    if mouth_right[1] < mouth_left[1]:
        rect_angle *= -1 # get rotation right

    # Create rectangle moustache mask
    box = cv2.boxPoints((rect_center, rect_len_width, rect_angle))
    box = np.int0(box)

    cv2.drawContours(img, [box], 0, (0,0,0), -1)



if not os.path.exists('bin'):
    os.makedirs('bin')

cv2.imwrite("bin/" + image_name, img)

#cv2.imshow(winname="Face", mat=img)
#cv2.waitKey(delay=0)
#cv2.destroyAllWindows()

end_time = time.time()

print("Took", (end_time - start_time) * 1000, "ms")
print(image_name)
