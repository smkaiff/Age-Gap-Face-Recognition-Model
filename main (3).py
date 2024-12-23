from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import face_recognition
import numpy as np
import threading
import time
import webbrowser

app = Flask(__name__)

# Load the encoding file
with open("EncodeFile.p", "rb") as file:
    encodeListKnownWithID = pickle.load(file)
encodeListKnown, personID = encodeListKnownWithID

# Initialize webcam
camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

# Variables to manage matching and stopping
face_match_start_time = None
face_matched = False
match_name = None

def generate_frames():
    global face_match_start_time, face_matched, match_name
    while True:
        success, frame = camera.read()
        if not success or face_matched:
            break
        else:
            # Resize and process frame for face recognition
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = personID[matchIndex]
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    # Draw rectangle and display name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{name} (Verified)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    # Check if face is matched for 2 consecutive seconds
                    if face_match_start_time is None:
                        face_match_start_time = time.time()
                    elif time.time() - face_match_start_time > 2:
                        face_matched = True
                        match_name = name
                        break
                else:
                    face_match_start_time = None

            # Encode the frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/check_status")
def check_status():
    if face_matched:
        return jsonify({"status": "success", "name": match_name})
    return jsonify({"status": "waiting"})

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
