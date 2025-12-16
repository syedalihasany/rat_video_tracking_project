import cv2

video_path = "ES30_1_13_24.mp4"
out_path    = "ES30_1_13_24_cropped.mp4"

# Manually crop (adjust coordinates after running find_crop_coordinates.py)
x1, y1 = 733, 349   # top-left corner of crop
x2, y2 = 1304, 803 # bottom-right corner of crop
# ---------------------------------------------------------

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# compute cropped frame size
W = x2 - x1
H = y2 - y1

writer = cv2.VideoWriter(
    out_path, 
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    crop = frame[y1:y2, x1:x2]   # slice
    writer.write(crop)

cap.release()
writer.release()
print("DONE - Cropped video saved.")