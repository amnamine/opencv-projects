import cv2

# Load pre-trained age estimation model
age_net = cv2.dnn.readNetFromTensorflow(cv2.data.haarcascades + 'opencv_age_estimation.pb')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    # Perform face detection using Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess face ROI for age estimation
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), swapRB=True)

        # Set the blob as input to the age estimation model
        age_net.setInput(blob)

        # Perform forward pass and get age estimation
        age_preds = age_net.forward()
        age_index = age_preds[0].argmax()
        age = age_index + 1

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write age text
        text = f'Age: {age} years'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow("Age Estimation", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
