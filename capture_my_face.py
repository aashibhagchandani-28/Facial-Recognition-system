import cv2
import os

person_name = input("Enter your name: ")

dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

os.makedirs(person_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
max_images = 30

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160,160))

        file_path = os.path.join(person_path, f"{count}.jpg")

        success = cv2.imwrite(file_path, face_img)

        if success:
            print("Saved:", file_path)
            count += 1

    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
