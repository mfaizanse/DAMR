import cv2

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard_create(6,7,0.3,0.20, dictionary)
img = board.draw((2828, 3700))
cv2.imwrite("Board.png", img)
cv2.waitKey(0)