import cv2

# read image as grey scale

def rotateFunction(img):

    # img = cv2.imread(imagePath)
    # get image height, width
    (h, w) = img.shape[:2
             ]
    # calculate the center of the image
    center = (w / 2, h / 2)

    angle90 = 90
    angle180 = 180

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))

    # 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(img, M, (w, h))


    cv2.imshow('Image rotated by 90 degrees', rotated90)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

    cv2.imshow('Image rotated by 180 degrees', rotated180)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

    return rotated90, rotated180


if __name__ == '__main__':
    image = "C:\\Users\\KSHITIJ\\PycharmProjects\\rotateImage\\virat.jpg"
    rotateFunction(image)