import cv2
# from PIL import Image
#
# img1 = Image.open('total_value.jpg')
# img2 = Image.open('total_value_Dueling DQN.jpg')
# background = Image.new('RGB', (img1.size[0], img1.size[1]), (0, 0, 0))
# merged = img1.convert('L') + img2.convert('L')
# merged = merged.point(lambda i: i.point(lambda j: j if j >= 0 else (255, 0, 0)), optimize='free')
# img = merged.convert('RGB')
# img.show()
src1 = cv2.imread('total_value (2).jpg')
src2 = cv2.imread('total_value_Dueling DQN (2).jpg')

c = cv2.addWeighted(src1, 0.3, src2, 0.7, 0)
cv2.imshow('addWeighted', c)
cv2.waitKey(0)
