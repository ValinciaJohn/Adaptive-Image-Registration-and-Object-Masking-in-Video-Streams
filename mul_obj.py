import cv2
import numpy as np
import glob

# List to store the coordinates of the marked points for each field
all_marked_fields_coords = []

# Globals to store current field points and fields
current_field = []
fields = []

# Globals for display image and scale factor
display_image = None
scale_factor = 1.0

def mark_field(event, x, y, flags, param):
    global current_field, fields, display_image, scale_factor
    if event == cv2.EVENT_LBUTTONDOWN:
        scaled_x, scaled_y = int(x / scale_factor), int(y / scale_factor)
        current_field.append((scaled_x, scaled_y))
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Template", display_image)
        if len(current_field) == 4:
            fields.append(np.array(current_field, dtype='float32'))
            current_field = []

def load_templates_and_mark_fields(template_files):
    global display_image, scale_factor, fields
    template_images = []

    for template_file in template_files:
        image = cv2.imread(template_file)

        # Resize the image to fit within the window
        window_size = (800, 600)  # Adjust the window size as needed
        height, width = image.shape[:2]
        scale_factor = min(window_size[1] / height, window_size[0] / width)
        display_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

        cv2.imshow("Template", display_image)
        cv2.setMouseCallback("Template", mark_field)
        cv2.waitKey(0)

        # Ensure we have at least one field marked
        if fields:
            print(f"Marked fields coordinates for {template_file}:", fields)
            all_marked_fields_coords.append(fields.copy())
            template_images.append(image)
            fields.clear()
        else:
            print(f"Please mark at least one field for {template_file}.")
            continue

    return template_images

# Load all template images from the templates directory
template_files = glob.glob('template/*.jpeg')  # Adjust the path and file extension as needed
template_images = load_templates_and_mark_fields(template_files)

# Save the coordinates to a file for later use
np.save('marked_fields_coords.npy', all_marked_fields_coords)

# Initialize video capture from live camera
video_capture = cv2.VideoCapture(0)

# Feature detector using SIFT
sift = cv2.SIFT_create()
kp_template_list = []
des_template_list = []

# Compute keypoints and descriptors for each template
for template_image in template_images:
    kp_template, des_template = sift.detectAndCompute(template_image, None)
    kp_template_list.append(kp_template)
    des_template_list.append(des_template)

# FLANN-based Matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features in the frame using SIFT
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
    
    for i, (kp_template, des_template) in enumerate(zip(kp_template_list, des_template_list)):
        # Match features using FLANN-based matcher
        matches = flann.knnMatch(des_template, des_frame, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Extract location of good matches
        pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography if enough matches are found
        if len(pts_template) >= 4 and len(pts_frame) >= 4:
            homography, mask = cv2.findHomography(pts_template, pts_frame, cv2.RANSAC, 5.0)
            
            if homography is not None:
                # Mask each field for the current template
                for marked_field_coords in all_marked_fields_coords[i]:
                    # Ensure marked_field_coords has the correct shape
                    marked_field_coords = np.array(marked_field_coords, dtype='float32')
                    marked_field_coords = marked_field_coords.reshape(-1, 1, 2)
                    
                    if marked_field_coords.shape[1] == 1 and marked_field_coords.shape[2] == 2:
                        # Transform the marked field coordinates
                        marked_field_transformed = cv2.perspectiveTransform(marked_field_coords, homography)
                        
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
                        print("Invalid shape of marked_field_coords, skipping field.")
            else:
                print(f"Homography could not be computed for template {i+1}")
        else:
            print(f"Not enough matches found to compute homography for template {i+1}")
    
    # Display the result
    cv2.imshow('Masked Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
