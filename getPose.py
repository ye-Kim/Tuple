import cv2 
import os
import numpy as np 
import mediapipe as mp 
import matplotlib.pyplot as plt 

# setting route of images
image_route = "./images/"
IMAGE_FILENAMES = os.listdir('./images/')
IMAGE_FILENAMES.sort()

sample_img = IMAGE_FILENAMES[2]

# initializing mediapipe pose class
mp_pose = mp.solutions.pose 

# setting the pose function 
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, enable_segmentation=True, model_complexity=2)

# initializing mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils 

# read sample image
sample_image = cv2.imread(image_route + sample_img)

# show sample image
def show_image(image, title=None):
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(image[:,:,::-1]) # convert BGR to RGB
    plt.show()

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape 
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

        if display:
            plt.figure(figsize=(18, 18))
            plt.subplot(121);plt.imshow(image[:,:,::-1])
            plt.title("Original image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1])
            plt.title("Output image");plt.axis('off')
            plt.show()
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        else: 
            return output_image, landmarks 
        
def detect_and_segment_pose(image, pose, title=None, display=True):
    BG_COLOR = (192,192,192) # grey
    mp_dlrawing_styles = mp.solutions.drawing_styles

    output_image = image.copy()
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1)>0.1
    bp_image = np.zeros(output_image.shape, dtype=np.uint8)
    bp_image[:] = BG_COLOR
    output_image = np.where(condition, output_image, bp_image)

    #Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_dlrawing_styles.get_default_pose_landmarks_style())
    
    if title != None:
        cv2.imwrite('/annotated_image' + str(title) + '.png', output_image)

    #Plot pose world landmarks.
    # mp_drawing.plot_landmarks(results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)

    if display:
        #Show annotated image
        plt.figure(figsize=(18,18))
        plt.subplot(121);plt.imshow(image[:,:,::-1])
        plt.title("Original image");plt.axis('off')
        plt.subplot(122);plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("After segmentation");plt.axis('off')
        plt.show()

    else: 
        return output_image
    