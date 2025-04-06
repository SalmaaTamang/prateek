import cv2
import os
import time
from fastapi import FastAPI
from fastapi.responses import FileResponse
from Pipeline import ClassifierPipelineAlexnet  

app = FastAPI()

video_dir = 'captured_videos'
img_dir = 'img'

if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

roi_size = 224

@app.get("/capture_video/")
async def capture_video_and_extract_frames():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Cannot open camera"}
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    roi_start = (
        frame_width // 2 - roi_size // 2,
        frame_height // 2 - roi_size // 2
    )
    roi_end = (
        frame_width // 2 + roi_size // 2,
        frame_height // 2 + roi_size // 2
    )

    video_filename = os.path.join(video_dir, 'captured_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Can't receive frame."}

        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)
        cv2.putText(frame, str(i), (frame_width // 2 - 30, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8, cv2.LINE_AA)
        cv2.imshow('Recording', frame)
        cv2.waitKey(1000)

    start_time = time.time()
    while int(time.time() - start_time) < 5:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Can't receive frame."}
        
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    extracted = 0
    frame_no = 0
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = frame_count // 5

    extracted_files = []
    last_frame_path = None

    while extracted < 5 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % interval == 0:
            x1, y1 = roi_start
            x2, y2 = roi_end
            roi_frame = frame[y1:y2, x1:x2]

            img_filename = os.path.join(img_dir, f"frame_{extracted+1}.jpg")
            cv2.imwrite(img_filename, roi_frame)
            extracted_files.append(f"frame_{extracted+1}.jpg")
            last_frame_path = img_filename  # Keep track of the last frame
            extracted += 1
        frame_no += 1

    cap.release()

    if last_frame_path:
        # Initialize the prediction pipeline
        pipeline = ClassifierPipelineAlexnet()
        pipeline.initialize_model()
        pipeline.load_model()

        # Load the last frame image
        last_frame = cv2.imread(last_frame_path)

        # Perform prediction on the last frame
        predicted_label = pipeline.predict(last_frame, last_frame_path)

        # Return the result
        return {
            "extracted_frames": extracted_files,
            "predicted_label": predicted_label
        }

    return {"extracted_frames": extracted_files, "predicted_label": None}
