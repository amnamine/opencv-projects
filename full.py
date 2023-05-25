import cv2

# Load the pre-trained MobileNet model
net = cv2.dnn.readNetFromTensorflow(cv2.data.haarcascades + 'mobilenet_v2_1.0_224_frozen.pb', cv2.data.haarcascades + 'mobilenet_v2_1.0_224_labels.txt')

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frames from the video capture
    ret, frame = cap.read()

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(224, 224), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass through the network to obtain the predictions
    predictions = net.forward()

    # Get the class label with the highest confidence
    class_id = np.argmax(predictions)
    confidence = predictions[0, class_id]

    # Get the label text for the class
    with open(cv2.data.haarcascades + 'mobilenet_v2_1.0_224_labels.txt') as f:
        labels = f.readlines()
    label = labels[class_id].strip()

    # Display the label and confidence on the frame
    cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Object Recognition', frame)

    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
