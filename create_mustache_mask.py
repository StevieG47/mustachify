from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Read image
image = cv2.imread('./ims/dan.png')
imageOG = image.copy()
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Run detector
detector = MTCNN()
faces = detector.detect_faces(image)
print("Found " + str(len(faces)) + " faces")

# Init output mask
image_out = np.zeros((image.shape))

# Loop through all detected faces
# and find moustache-mask-area based on detected
# facial keypoints
for face in faces:

    # Get coords of mouth and nose
    mouth_left = face['keypoints']['mouth_left']
    mouth_right = face['keypoints']['mouth_right']
    nose = face['keypoints']['nose']

    # X coord between left/right mouth points
    center_x = np.mean((mouth_left[0],mouth_right[0]))

    # 2d distance between mouth points, use as mask rectangle width
    mouth_width = np.sqrt((mouth_left[0]-mouth_right[0])**2 + (mouth_left[1]-mouth_right[1])**2)

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
    rect_len_width = (mouth_width+ mouth_width*0.5, nose_to_mouth_dist)
    rect_angle = np.arctan2(max_mouth_y-min_mouth_y, abs(mouth_left[0]-mouth_right[0]))*180/np.pi # rotate cw off x axis
    if mouth_right[1] < mouth_left[1]:
        rect_angle *= -1 # get rotation right

    # Create rectangle moustache mask
    rot_rectangle = (rect_center, rect_len_width, rect_angle)
    box = cv2.boxPoints(rot_rectangle) 
    box = np.int0(box)

    # Draw mustache bounding box
    image = cv2.drawContours(image,[box],0,(0,0,0),-1)
    image_out = cv2.drawContours(image_out,[box],0,(255,255,255),-1)

    # Draw facial keypoints
    radius = 3
    cv2.circle(image,face['keypoints']['mouth_left'],radius,[0,0,255],-1)
    cv2.circle(image,face['keypoints']['mouth_right'],radius,[0,0,255],-1)
    cv2.circle(image,face['keypoints']['nose'],radius,[255,0,255],-1)

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
size = 512
image     = cv2.resize(image,(512,512))
image_out = cv2.resize(image_out,(512,512))
image_out_show = image_out.astype('uint8')
cv2.imshow('Image',np.hstack((image,image_out_show)))
cv2.waitKey(0)
cv2.destroyAllWindows()

imageOG = cv2.resize(imageOG,(512,512))
cv2.imwrite('./ims/input_image.png',imageOG)
cv2.imwrite('./ims/mask_image.png',image_out)