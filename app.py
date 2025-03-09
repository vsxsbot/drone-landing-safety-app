from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv5 model
model = YOLO('yolov5s.pt')  # Make sure 'yolov5s.pt' is in the same directory

# Open webcam
cap = cv2.VideoCapture(0)

def generate_frames():
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
                class_name = model.names[int(class_id)]

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

        # Encode frame for web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)