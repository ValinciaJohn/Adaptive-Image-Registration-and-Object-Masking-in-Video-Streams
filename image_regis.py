import cv2
import numpy as np

# Load the template image and manually marked coordinates of the fields
template_image = cv2.imread('template2.jpeg')
marked_fields_coords = np.load('marked_fields_coords.npy', allow_pickle=True)

# Initialize video capture
video_capture = cv2.VideoCapture('temp_video.mp4')

# Feature detector
orb = cv2.ORB_create()
kp_template, des_template = orb.detectAndCompute(template_image, None)

# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect features in the frame
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    
    # Match features
    matches = bf.match(des_template, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    pts_template = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography if enough matches are found
    if len(pts_template) >= 4 and len(pts_frame) >= 4:
        homography, _ = cv2.findHomography(pts_template, pts_frame, cv2.RANSAC, 5.0)
        
        # Mask each field
        for marked_field_coords in marked_fields_coords:
            # Transform the marked field coordinates
            marked_field_transformed = cv2.perspectiveTransform(marked_field_coords.reshape(-1, 1, 2), homography)
            
            # Debug: Print transformed coordinates
            print("Transformed coordinates:", marked_field_transformed)
            
            # Ensure coordinates are within frame bounds
            valid_coords = True
            for coord in marked_field_transformed:
                if coord[0][0] < 0 or coord[0][1] < 0 or coord[0][0] > frame.shape[1] or coord[0][1] > frame.shape[0]:
                    valid_coords = False
                    break
            
            if valid_coords:
                # Apply masking with a solid color (e.g., black)
                mask_color = (0, 0, 0)  # Change to any color if needed
                cv2.fillPoly(frame, [np.int32(marked_field_transformed)], mask_color)
            else:
                print("Invalid coordinates, skipping field.")
    else:
        print("Not enough matches found to compute homography.")
    
    # Display the result
    cv2.imshow('Masked Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
