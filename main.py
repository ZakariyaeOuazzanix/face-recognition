import threading  # Imports the threading module to allow concurrent execution of tasks (in this case, checking the face in a separate thread).
import cv2  # Imports the OpenCV library for computer vision tasks.
from deepface import DeepFace  # Imports the DeepFace library, which is used for face recognition.

# Initialize the webcam video capture.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the video frame width and height to 640x480.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize a counter to track frames.
counter = 0

# Initialize a boolean variable to store the face match status.
face_match = False

# Create an empty list to store reference images for face matching.
reference_images = []

# Load 6 reference images from a specific directory and append them to the list.
for i in range(1,7):
    reference_images.append(cv2.imread('I:/learing_projects/face-recognition-python/reference_images/reference_img-'+str(i)+'.jpg'))

# Define a function to check if the current frame matches any of the reference images.
def check_face(frame):
    global face_match  # Access the global variable face_match.
    try:
        # Loop through each reference image and verify if it matches the frame.
        for i in range(6):
            if DeepFace.verify(frame, reference_images[i])['verified']:
                face_match = True  # If a match is found, set face_match to True and exit the loop.
                return
        
        face_match = False  # If no match is found, set face_match to False.

    except ValueError:
        face_match = False  # Handle potential errors by setting face_match to False.

# Main loop to continuously capture frames from the webcam.
while True:
    # Read a frame from the video capture.
    return_value, frame = cap.read()

    # If a frame was successfully captured:
    if return_value:
        # Every 30 frames, start a new thread to check for a face match.
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(), ) ).start()  # Start a thread for face matching.
            except ValueError:
                pass  # Ignore any errors that occur during threading.

        counter += 1  # Increment the frame counter.

        # If a face match was found, display "MATCH!" on the video frame.
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # If no match was found, display "NO MATCH!" on the video frame.
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Display the current video frame in a window titled "video".
        cv2.imshow("video", frame)

    # Check if the 'q' key was pressed to exit the loop.
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break

# Destroy all OpenCV windows after the loop ends.
cv2.destroyAllWindows()
