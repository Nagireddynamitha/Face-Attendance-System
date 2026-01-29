import cv2
import face_recognition
import pickle
import os
from datetime import datetime
import pandas as pd

# Step 1: Load known encodings
encodings_path = os.path.join(os.getcwd(), "..", "dataset", "encodings.pkl")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Step 2: Prepare attendance CSV
attendance_file = os.path.join(os.getcwd(), "..", "attendance", "attendance.csv")
os.makedirs(os.path.dirname(attendance_file), exist_ok=True)

# Load existing attendance or create new
if os.path.exists(attendance_file):
    df_attendance = pd.read_csv(attendance_file)
else:
    df_attendance = pd.DataFrame(columns=["Name", "Date", "Punch_in", "Punch_out"])

# Step 3: Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("📸 Press 'Q' to quit")

# Step 4: Recognize faces in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Find best match
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]

            # Attendance logic
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            # Check if this person already has a punch-in today
            mask = (df_attendance["Name"] == name) & (df_attendance["Date"] == date_str)
            if not df_attendance[mask].empty:
                # Already punched in → update punch-out
                df_attendance.loc[mask, "Punch_out"] = time_str
            else:
                # First punch → add new row
                df_attendance = pd.concat(
                    [df_attendance, pd.DataFrame([{
                        "Name": name,
                        "Date": date_str,
                        "Punch_in": time_str,
                        "Punch_out": ""
                    }])],
                    ignore_index=True
                )
            # Save to CSV
            df_attendance.to_csv(attendance_file, index=False)

        # Draw rectangle and name
        top, right, bottom, left = [v * 4 for v in face_location]  # scale back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Attendance - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Attendance session ended.")
