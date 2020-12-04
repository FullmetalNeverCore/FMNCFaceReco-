import cv2 
import sys

imgpth = sys.argv[1]
casc = sys.argv[2]

fccs = cv2.CascadeClassifier(casc)
img = cv2.imread(imgpth)
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = fccs.detectMultiScale(
    gry,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)
print("FOUND ONE".format(len(face)))

for (x,y,z,c) in face:
    cv2.rectangle(img, (x, y), (x+z, y+c), (255, 0, 0), 2)

cv2.imshow("Faces", img)
cv2.waitKey(0)