import cv2
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.platform import gfile
import csv
import os
from datetime import datetime

# ---------------------------
# Attendance setup
# ---------------------------

ATTENDANCE_FILE = "attendance.csv"

# create file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

marked_names = set()

def mark_attendance(name):

    if name == "Unknown":
        return

    if name in marked_names:
        return

    now = datetime.now()
    time_string = now.strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, time_string])

    marked_names.add(name)

    print(f"Attendance marked for {name}")


# ---------------------------
# Load classifier
# ---------------------------

with open("classifier.pkl", "rb") as f:
    model, class_names = pickle.load(f)

# Load FaceNet model
model_path = "../models/20180402-114759/20180402-114759.pb"

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------
# Start TensorFlow session
# ---------------------------

with tf.Graph().as_default():
    with tf.Session() as sess:

        with gfile.FastGFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Camera failed to open")
            exit()
        else:
            print("Camera opened successfully")

        print("Press Q to exit")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(60,60)
            )

            for (x, y, w, h) in faces:

                face = frame[y:y+h, x:x+w]

                face = cv2.resize(face, (160,160))

                face = face.astype(np.float32)

                face = np.expand_dims(face, axis=0)

                emb = sess.run(
                    embeddings,
                    feed_dict={
                        images_placeholder: face,
                        phase_train_placeholder: False
                    }
                )

                predictions = model.predict_proba(emb)

                best_index = np.argmax(predictions)

                best_prob = predictions[0][best_index]

                # ---------------------------
                # Unknown detection
                # ---------------------------

                if best_prob < 0.7:
                    name = "Unknown"
                else:
                    name = class_names[best_index]
                    mark_attendance(name)

                text = f"{name} ({best_prob:.2f})"

                # draw rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                # draw name
                cv2.putText(
                    frame,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            cv2.imshow("Live Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
