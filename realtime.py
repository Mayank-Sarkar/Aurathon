# realtime_detection.py
import cv2
from ultralytics import YOLO

def main():
    # Load your trained model
    model = YOLO("detect/yolov8n_leaf_detection/weights/best.pt")

    # Open webcam (0 = default camera, change index if needed)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True)

        # Plot results on the frame
        for r in results:
            annotated_frame = r.plot()

        # Show the frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
