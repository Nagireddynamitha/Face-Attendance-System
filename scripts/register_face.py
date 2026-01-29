import cv2
import os

# Step 0: Show working directory (for debugging)
print("Current working directory:", os.getcwd())

# Step 1: Enter your name
name = input("Enter your name: ").strip()

# Step 2: Create folder to save images
save_path = os.path.join(os.getcwd(), "..", "dataset", "registered_faces", name)
os.makedirs(save_path, exist_ok=True)
print("Saving images to:", save_path)

# Step 3: Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'C' to capture image")
print("Press 'Q' to quit")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    # Show video frame
    cv2.imshow("Register Face", frame)

    # Detect key press
    key = cv2.waitKey(10)  # delay 10ms

    # Capture image if 'C' or 'c' is pressed
    if key == ord('c') or key == 67:
        img_path = os.path.join(save_path, f"img_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Image saved: {img_path}")
        count += 1

    # Quit if 'Q' or 'q' is pressed
    elif key == ord('q') or key == 81:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Registration complete. Total images saved: {count}")
