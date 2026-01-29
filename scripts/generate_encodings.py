import face_recognition
import os
import pickle
import cv2

# Step 1: Paths
dataset_path = os.path.join(os.getcwd(), "..", "dataset", "registered_faces")
encodings_file = os.path.join(os.getcwd(), "..", "dataset", "encodings.pkl")

# Step 2: Lists to store embeddings and names
known_encodings = []
known_names = []

# Step 3: Loop through each user folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Loop through each image
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"Processing {image_path}...")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get face encodings (assume one face per image)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 0:
            print(f"No face found in {image_path}")
            continue

        # Save first encoding and name
        known_encodings.append(encodings[0])
        known_names.append(person_name)

# Step 4: Save encodings to file
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print(f"Encodings saved to {encodings_file}")
print(f"Total faces encoded: {len(known_encodings)}")
