import numpy as np
import dlib
import cv2
import time
import os

def distance(p0, p1):
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

# inputs: path to shape_predictor_68_face_landmarks.dat
def get_face_predictor_detector(landmark_path):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(landmark_path)
	return detector, predictor

# inputs: grayscale image
def get_image_mask(image, detector, predictor):

	start_time = time.time()

	# Init output mask
	mask_image = np.zeros((image.shape))

	debug_image = image.copy()
	faces = detector(image)
	print('Found ' +str(len(faces))+ ' face(s)')
	for face in faces:

	    landmarks = predictor(image=image, box=face)
	    for n in range(0, 68):
	        x = landmarks.part(n).x
	        y = landmarks.part(n).y

	    nose_p_32 = (landmarks.part(32).x, landmarks.part(32).y)
	    nose_p_34 = (landmarks.part(34).x, landmarks.part(34).y)
	    nose_p_31 = (landmarks.part(31).x, landmarks.part(31).y)
	    nose_p_35 = (landmarks.part(35).x, landmarks.part(35).y)

	    mouth_p_48 = (landmarks.part(48).x, landmarks.part(48).y)
	    mouth_p_49 = (landmarks.part(49).x, landmarks.part(49).y)
	    mouth_p_53 = (landmarks.part(53).x, landmarks.part(53).y)
	    mouth_p_54 = (landmarks.part(54).x, landmarks.part(54).y)
	    mouth_p_51 = (landmarks.part(51).x, landmarks.part(51).y)
	    mouth_p_62 = (landmarks.part(62).x, landmarks.part(62).y)

	    mouth_left  = mouth_p_49
	    mouth_right = mouth_p_53
	    nose = (np.mean((nose_p_31[0], nose_p_35[0])), np.mean((nose_p_31[1], nose_p_35[1])))

	    # X coord between left/right mouth points
	    center_x = np.mean((mouth_left[0],mouth_right[0]))

	    # 2d distance between mouth points, use as mask rectangle width
	    mouth_width = distance(mouth_p_48, mouth_p_54)

	    # Get distance from nose to middle of mouth, use as mask rectangle height
	    nose_to_mouth_dist = distance(nose, mouth_p_62)

	    # Y coord, set as halfway from nose to mouth
	    # TODO: account for angle
	    if nose[1] < mouth_p_51[1]:
	        center_y = nose[1] + nose_to_mouth_dist/2
	    else:
	        center_y = nose[1] - nose_to_mouth_dist/2

	    # Set rectangle properties for rectangle moustache mask
	    # points are x, y
	    rect_center = (center_x, center_y)
	    rect_len_width = (mouth_width*1.25, nose_to_mouth_dist*1.0)
	    rect_angle = np.arctan2(abs(mouth_p_48[1]-mouth_p_54[1]), abs(mouth_p_48[0]-mouth_p_54[0]))*180/np.pi # rotate cw off x axis
	    if mouth_p_54[1] < mouth_p_48[1]:
	        rect_angle *= -1 # get rotation right

	    # Create rectangle moustache mask
	    box = cv2.boxPoints((rect_center, rect_len_width, rect_angle))
	    box = np.int0(box)

	    # Draw bounding box to mask image
	    mask_image = cv2.drawContours(mask_image,[box],0,(255,255,255),-1)
	    debug_image = cv2.drawContours(debug_image,[box],0,(0,0,0),-1)

	print("Time to get image mask: ", np.round((time.time() - start_time) * 1000,2), "ms")
	return mask_image, debug_image