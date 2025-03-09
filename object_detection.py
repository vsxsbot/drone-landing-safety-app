import cv2
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov5s.pt')  # Make sure 'yolov5s.pt' is in the same directory

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Draw an imaginary circle in the center of the frame
    circle_center = (frame_width // 2, frame_height // 2)
    circle_radius = 100
    cv2.circle(frame, circle_center, circle_radius, (255, 0, 0), 2)  # Blue circle

    # Perform object detection
    results = model(frame, stream=True)

    collision_detected = False

    # Iterate over detected objects
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Get the class name
            class_name = model.names[int(class_id)]  # Get the object name from the model

            # Display the class name and confidence score on the bounding box
            label = f"{class_name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Calculate the center of the bounding box
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Check for collision with the circle
            distance = cv2.norm(object_center, circle_center, cv2.NORM_L2)
            if distance < circle_radius:
                collision_detected = True

    # Display collision status
    status_text = "Safe: True" if not collision_detected else "Safe: False"
    color = (0, 255, 0) if not collision_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()