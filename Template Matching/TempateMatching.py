import cv2
import numpy as np

print("[INFO] loading images...")
image = cv2.imread('main_diamond.png')
template = cv2.imread('template_diamond.png')
cv2.imshow("Image", image)
cv2.imshow("Template", template)
print("[INFO] Image Size : ",image.shape)
print("[INFO] Template Size : ",template.shape)

# convert both the image and template to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# perform template matching
print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

##################### Using minmaxLoc will only fetch the Max coordinate and hence will detect only one similar point
# (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

## determine the starting and ending (x, y)-coordinates of the
## bounding box
# (startX, startY) = maxLoc
# endX = startX + template.shape[0]
# endY = startY + template.shape[1]
#
# # draw the bounding box on the image
# cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)

################################## Multiple Object Detection

# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image, so we
# can draw on it
(yCoords, xCoords) = np.where(result >= 0.90)
clone = image.copy()
print("[INFO] {} matched locations ",xCoords, yCoords)
# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
	# draw the bounding box on the image
	cv2.rectangle(clone, (x, y), (x + template.shape[0], y + template.shape[1]),
		(255, 0, 0), 1)

cv2.rectangle(clone, (62, 247), (62 + template.shape[0], 247 + template.shape[1]),
		(0, 255, 0), 10)
cv2.imshow("Before NMS", clone)
cv2.waitKey(0)
