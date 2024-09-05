import cv2
import numpy as np

def makecanny(img):
    if img is None:
        raise ValueError("Input image is None")
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernal = 3
    blur = cv2.GaussianBlur(grayimg, (kernal, kernal), 0)
    canny = cv2.Canny(blur, 300, 300)
    return canny

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)
    rectangle = np.array([[
        (200, height),      # Bottom-left corner
        (600, int(height * 0.69)),  # Top-left corner (mid-height)
        (700, int(height * 0.69)),  # Top-right corner (mid-height)
        (1100, height)      # Bottom-right corner
    ]], np.int32)
    cv2.fillPoly(mask, rectangle, 255)
    maskedimg = cv2.bitwise_and(img, mask)
    return maskedimg

def houghlines(img):
    if img is None:
        raise ValueError("Input image is None")
    return cv2.HoughLinesP(img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)


def make_points(img, line):
    slope, intercept = line
    y1 = img.shape[0]  
    y2 = int(y1 * 0.77)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def fill_lane(frame, left_line, right_line):
    lane_img = np.zeros_like(frame)
    if left_line is not None and right_line is not None:
        left_x1, left_y1, left_x2, left_y2 = left_line[0]
        right_x1, right_y1, right_x2, right_y2 = right_line[0]
        pts = np.array([[left_x1, left_y1], [left_x2, left_y2], [right_x2, right_y2], [right_x1, right_y1]], np.int32)
        cv2.fillPoly(lane_img, [pts], (0, 150, 0)) 

        cv2.line(lane_img, (left_x1, left_y1), (left_x2, left_y2), (0, 0, 255), 10)
        cv2.line(lane_img, (right_x1, right_y1), (right_x2, right_y2), (0, 0, 255), 10)
        
        combined_img = cv2.addWeighted(frame, 0.7, lane_img, 0.5, 1)
        return combined_img
    else:
        return frame

def display_lines(img, lines):
    if lines is None:
        return np.zeros_like(img)
    line_img = np.zeros_like(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_img

def addWeighted(frame, line_img):
    return cv2.addWeighted(frame, 0.8, line_img, 1, 1)

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) == 0 or len(right_fit) == 0:
        return None
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(img, left_fit_average)
    right_line = make_points(img, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines




cap = cv2.VideoCapture("test1.mp4")

if not cap.isOpened():
    raise ValueError("Error opening video file")

while True:
    succ, frame = cap.read()
    if not succ:
        print("Failed to read frame or video ended.")
        break

    try:
        cannyimg = makecanny(frame)
        croppedimg = region_of_interest(cannyimg)
        lines = houghlines(croppedimg)
        avglines = average_slope_intercept(frame, lines)
        line_img = display_lines(frame, avglines)
        combo_img = addWeighted(frame, line_img)
        try:
            left_line, right_line = avglines
            lane_filled = fill_lane(frame, left_line, right_line)
            resized_img = cv2.resize(lane_filled,(700,500))
            cv2.imshow("output", resized_img)
        # cv2.imshow("output", croppedimg)
        except:
            resized_img = cv2.resize(combo_img,(700,500))
            cv2.imshow("output", resized_img)
    except Exception as e:
        print(f"Error processing frame: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
