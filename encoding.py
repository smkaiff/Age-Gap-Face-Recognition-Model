import cv2
import face_recognition
import pickle
import os

# Path to the folder containing images
folderPath = "Images"
modePathList = os.listdir(folderPath)
imgList = []
personID = []

# Read images and their corresponding IDs
for path in modePathList:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is not None:  # Check if the image was loaded correctly
        imgList.append(img)
        personID.append(os.path.splitext(path)[0])  # Extract ID from filename
        print(f"Loaded image: {path} with ID: {personID[-1]}")
    else:
        print(f"Warning: Could not load image: {path}")

print(f"Total images loaded: {len(imgList)}")

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Ensure at least one face encoding is found
            encodeList.append(encodings[0])  # Take the first face found
        else:
            print("Warning: No face found in image.")

    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithID = [encodeListKnown, personID]
print("Encoding Completed Successfully...")

# Save the encodings to a file
with open("EncodeFile.p", "wb") as file:
    pickle.dump(encodeListKnownWithID, file)

print("File saved successfully.")
