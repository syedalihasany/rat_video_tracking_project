import cv2

# Set video path and frame index
video_path   = r"ES30_1_13_24.mp4"
frame_index  = 500                       # frame to sample (0 = first)

# Grab a frame from the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

# Jump to desired frame (optional)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {frame_index} from video")

# Mouse callback to record clicks
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked → x={x}, y={y}")

# Show frame and wait for clicks
cv2.imshow("Select crop corners (press any key to quit)", frame)
cv2.setMouseCallback("Select crop corners (press any key to quit)", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()