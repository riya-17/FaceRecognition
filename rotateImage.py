import cv2
import numpy as np


# rotation of the image
def rotate_image(image, angle):
    # Size of the image input
    size = (image.shape[1], image.shape[0])
    center_of_image = tuple(np.array(size) / 2)
    rotation = np.vstack(
        [cv2.getRotationMatrix2D(center_of_image, angle, 1.0), [0, 0, 1]]
    )

    rotation2 = np.matrix(rotation[0:2, 0:2])
    width = size[0] * 0.5
    height = size[1] * 0.5
    rotated_coordinates = [
        (np.array([-width, height]) * rotation2).A[0],
        (np.array([width, height]) * rotation2).A[0],
        (np.array([-width, -height]) * rotation2).A[0],
        (np.array([width, -height]) * rotation2).A[0]
    ]

    # Size of the image output
    x_coordinates = [pt[0] for pt in rotated_coordinates]
    x_positif = [x for x in x_coordinates if x > 0]
    x_negatif = [x for x in x_coordinates if x < 0]
    y_coordinates = [pt[1] for pt in rotated_coordinates]
    y_positif = [y for y in y_coordinates if y > 0]
    y_negatif = [y for y in y_coordinates if y < 0]
    right = max(x_positif)
    left = min(x_negatif)
    top = max(y_positif)
    bottom = min(y_negatif)
    new_width = int(abs(right - left))
    new_height = int(abs(top - bottom))
    translation_matrix = np.matrix([
        [1, 0, int(new_width * 0.5 - width)],
        [0, 1, int(new_height * 0.5 - height)],
        [0, 0, 1]
    ])

    compute_matrix = (np.matrix(translation_matrix) * np.matrix(rotation))[0:2, :]
    result = cv2.warpAffine(
        image,
        compute_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR
    )

    return result
