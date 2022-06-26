import cv2
import os
import time
import numpy as np
directory_path = os.getcwd()


def draw_lines_all(img, lines, color=[0, 0, 255], thickness=7):
  try:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  except:
    print('empty')

def color_filter(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  img_thresh_low = cv2.inRange(hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
  img_thresh_high = cv2.inRange(hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))
  mask = cv2.bitwise_or(img_thresh_low, img_thresh_high)

  return mask

def canny_filter(img):

  low_thres = 80
  high_thres = 160
  edges = cv2.Canny(img, low_thres, high_thres)

  return edges

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

cap = cv2.VideoCapture(directory_path + '/cone_test/rm_vid1.mp4')

while (cap.isOpened()):
 
  # Capture frame-by-frame
  ret, frame = cap.read()

  kernel_size = 2
  blur = cv2.blur(frame, (kernel_size, kernel_size), 2)

  cv2.imshow('gaus. blur', blur)

  hsv_filtered = color_filter(blur)

  blur = cv2.bitwise_and(blur, blur, mask=hsv_filtered)

  cv2.imshow('orange?', blur)

  edges = canny_filter(blur)

  cv2.imshow('edges?', edges)

  contours, _ = cv2.findContours(np.array(edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  img_contours = np.zeros_like(edges)
  cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
  cv2.imshow('img_contours', img_contours)

  approx_contours = []

  for c in contours:
    approx = cv2.approxPolyDP(c, 5, closed = True)
    approx_contours.append(approx)
  img_approx_contours = np.zeros_like(edges)
  cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)
  cv2.imshow('img_approx_contours', img_approx_contours)

  all_convex_hulls = []
  for ac in approx_contours:
    all_convex_hulls.append(cv2.convexHull(ac))
  img_all_convex_hulls = np.zeros_like(edges)
  cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)
  cv2.imshow('img_all_convex_hulls', img_all_convex_hulls)

  convex_hulls_3to10 = []
  for ch in all_convex_hulls:
      if 3 <= len(ch) <= 10:
          convex_hulls_3to10.append(cv2.convexHull(ch))
  img_convex_hulls_3to10 = np.zeros_like(edges)
  cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)
  cv2.imshow('img_convex_hulls_3to10', img_convex_hulls_3to10)

  # time when we finish processing for this frame
  new_frame_time = time.time()

  # Calculating the fps

  # fps will be number of frame processed in given time frame
  # since their will be most of time error of 0.001 second
  # we will be subtracting it to get more accurate result
  fps = 1/(new_frame_time-prev_frame_time)
  prev_frame_time = new_frame_time

  # converting the fps into integer
  fps = int(fps)

  # converting the fps to string so that we can display it on frame
  # by using putText function
  fps = str(fps)

  # putting the FPS count on the frame
  cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
 

  cv2.imshow('frame', frame)

  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
 
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()