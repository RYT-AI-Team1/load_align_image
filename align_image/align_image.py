# import cv2
# import numpy as np


# def reorder_points(pts):
#     """
#     @param pts: list of 4 points
#     """
#     rect = np.zeros((4, 2), dtype = "float32")
#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis = 1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     # now, compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis = 1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     # return the ordered coordinates
#     return rect

# def perspective_img(img, keypoints):
#     """
#     @param img: input image
#     @param keypoints: list of 4 points
#     """

#     # if isinstance(keypoints, list):
#     #     keypoints = np.array(keypoints)

#     # rect = reorder_points(keypoints)
#     (tl, tr, br, bl) = keypoints

#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates

#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))

#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates

#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
    
#     # now that we have the dimensions of the new image, construct
#     # the set of destination points to obtain a "birds eye view",
#     # (i.e. top-down view) of the image, again specifying points
#     # in the top-left, top-right, bottom-right, and bottom-left order

#     dst_points = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1] ], dtype = "float32")

#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(keypoints, dst_points)
#     warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

#     # return the warped image
#     return warped


# if __name__ == "__main__":
#     img = cv2.imread('test_img.jpg')
#     keypoints = [[14.9, 7.7], [585.3, 9.8], [589.9, 367.5], [8.9, 369.4]]
#     warped = perspective_img(img, keypoints)
    
import cv2
import numpy as np

def perspective_img(img, keypoints):
    """
    @param img: input image
    @param keypoints: list of 4 points
    """

    # if isinstance(keypoints, list):
    #     keypoints = np.array(keypoints)
    # rect = reorder_points(keypoints)
    (tl, tr, br, bl) = keypoints

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order

    dst_points = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1] ], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(keypoints, dst_points)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


if __name__ == "__main__":
    img = cv2.imread('/home/hai/workspace/Load_align/CMT_good.jpg')
    keypoints = [[14.9, 7.7], [585.3, 9.8], [589.9, 367.5], [8.9, 369.4]]
    warped = perspective_img(img, keypoints)