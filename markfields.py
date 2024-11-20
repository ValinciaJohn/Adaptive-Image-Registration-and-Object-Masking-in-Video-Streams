import cv2
import numpy as np

# List to store the coordinates of the marked points for each field
fields = []
current_field = []

def mark_field(event, x, y, flags, param):
    global current_field, fields
    if event == cv2.EVENT_LBUTTONDOWN:
        current_field.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Template", image)
        if len(current_field) == 4:
            fields.append(np.array(current_field, dtype='float32'))
            current_field = []
            cv2.imshow("Template", image)

# Load the template image
image = cv2.imread('template2.jpeg')
cv2.imshow("Template", image)
cv2.setMouseCallback("Template", mark_field)
cv2.waitKey(0)

# Ensure we have at least one field marked
if fields:
    print("Marked fields coordinates:", fields)
else:
    print("Please mark at least one field.")

# Save the coordinates to a file for later use
np.save('marked_fields_coords.npy', fields)
