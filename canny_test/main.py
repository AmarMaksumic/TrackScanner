import cv2
import os
import numpy as np
directory_path = os.getcwd()

def draw_lines_all(img, lines, color=[255, 0, 0], thickness=7):
  try:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
  except:
    print('errored2')

def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    y_bottom = img.shape[0]
    y_upper = 7*(frame.shape[0]/10)

    slope = 0
    b = 0

    
    try:


      for line in lines:
          for x1,y1,x2,y2 in line:
              # test and filter values to slope
              if ((y2-y1)/(x2-x1)) > 0.5 and ((y2-y1)/(x2-x1)) < 0.8:
                  
                  slope = ((y2-y1)/(x2-x1))
                  b = y1 - slope*x1
                  
                  x_bottom_pos.append((y_bottom - b)/slope)
                  x_upper_pos.append((y_upper - b)/slope)
                                        
              elif ((y2-y1)/(x2-x1)) < -0.5 and ((y2-y1)/(x2-x1)) > -0.8:
              
                  slope = ((y2-y1)/(x2-x1))
                  b = y1 - slope*x1
                  
                  x_bottom_neg.append((y_bottom - b)/slope)
                  x_upper_neg.append((y_upper - b)/slope)

      # a new 2d array with means 
      lines_mean = np.array([[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))], 
                              [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))]])

      # Draw the lines
      for i in range(len(lines_mean)):
          cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)
    except:
      print('errored')

# frame = cv2.imread(directory_path + '/canny_test/road2.jpg')

cap = cv2.VideoCapture(directory_path + '/canny_test/stolacneum.mp4')
# if frame is None:
#     print('Could not open or find the image')
#     exit(0)

while (cap.isOpened()):
 
  # Capture frame-by-frame
  ret, frame = cap.read()

  pt_a_x = 3.5*(frame.shape[1]/7)
  pt_a_y = 8*(frame.shape[0]/10)
  pt_b_x = 2.5*(frame.shape[1]/7)
  pt_b_y = 10*(frame.shape[0]/10)
  pt_c_x = 5.5*(frame.shape[1]/7)
  pt_c_y = 10*(frame.shape[0]/10)
  pt_d_x = 4.5*(frame.shape[1]/7)
  pt_d_y = 8*(frame.shape[0]/10)

  grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # cv2.imshow('grayscale', grayscale)

  kernel_size = 3
  blur = cv2.blur(grayscale, (kernel_size, kernel_size), 0)

  # cv2.imshow('gaus. blur', blur)

  # Here, I have used L2 norm. You can use L1 also.
  width_AD = np.sqrt(((pt_a_x - pt_d_x) ** 2) + ((pt_a_y - pt_d_y) ** 2))
  width_BC = np.sqrt(((pt_b_x - pt_c_x) ** 2) + ((pt_b_y - pt_c_y) ** 2))
  maxWidth = max(int(width_AD), int(width_BC))


  height_AB = np.sqrt(((pt_a_x - pt_b_x) ** 2) + ((pt_a_y - pt_b_y) ** 2))
  height_CD = np.sqrt(((pt_c_x - pt_d_x) ** 2) + ((pt_c_y - pt_d_y) ** 2))
  maxHeight = max(int(height_AB), int(height_CD))

  input_pts = np.float32([[pt_a_x, pt_a_y], [pt_b_x, pt_b_y], [pt_c_x, pt_c_y], [pt_d_x, pt_d_y]])
  output_pts = np.float32([[0, 0],
                          [0, maxHeight - 1],
                          [maxWidth - 1, maxHeight - 1],
                          [maxWidth - 1, 0]])

  # Compute the perspective transform M
  M = cv2.getPerspectiveTransform(input_pts,output_pts)
  out = cv2.warpPerspective(blur, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

  cv2.imshow('modified perspective', out)

  low_thres = 75
  high_thres = 200
  edges = cv2.Canny(out, low_thres, high_thres)

  cv2.imshow('canny edges', edges)

  # vertices = np.array([[(pt_b_x, pt_b_y), 
  #                       (pt_a_x, pt_a_y), 
  #                       (pt_d_x, pt_d_y), 
  #                       (pt_c_x, pt_c_y)]], 
  #                       dtype=np.int32)

  # mask = np.zeros_like(edges)
  # cv2.fillPoly(mask, vertices, (255, 255, 255))
  # masked_edges = cv2.bitwise_and(edges, mask)

  # cv2.imshow('masked edges', edges)

  rho = 3
  theta = np.pi / 180
  threshold = 100
  min_line_len = 150
  max_line_gap = 100
  lines = cv2.HoughLinesP(out, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

  draw_lines(out, lines)
  cv2.imshow('image', out)

  draw_lines_all(out, lines)
  cv2.imshow('image2', out)

  # cv2.waitKey(0)

  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
 
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()