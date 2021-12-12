import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
import NeuralNetwork as NN
from sklearn.decomposition import KernelPCA
import random


class ImageTexture(object):
    def getSamples8x8(self, img):
        while True:
            x = np.random.randint(low=0, high=1024)
            y = np.random.randint(low=0, high=1024)
            if img[x, y] == 255:
                break
        return x, y

    def plotGraphic(self, x, y, marker='o', xlabel='xlabel', ylabel='ylabel'):
        plt.figure()
        plt.plot(x, y, marker=marker)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt._show()

    def convertToJPG(self):
        for i in range(1, 323):
            if i < 10:
                img = cv2.imread(os.getcwd() + '\\MIAS\\mdb00' + str(i) + '.pgm', 0)
                cv2.imwrite('E:\\JPG\\mdb00' + str(i) + '.jpg', img)
            elif i >= 10 and i < 100:
                img = cv2.imread(os.getcwd() + '\\MIAS\\mdb0' + str(i) + '.pgm', 0)
                cv2.imwrite('E:\\JPG\\mdb0' + str(i) + '.jpg', img)
            else:
                img = cv2.imread(os.getcwd() + '\\MIAS\\mdb' + str(i) + '.pgm', 0)
                cv2.imwrite('E:\\JPG\\mdb' + str(i) + '.jpg', img)

    def saveImg(self, path, filename, img):
        cv2.imwrite(path + filename + '.jpg', img)

    def quantatizeImage(self, img, gray_level=256):
        n = 0
        while gray_level != 1:
            gray_level = np.floor_divide(gray_level, 2)
            n += 1
        img /= 2 ** (8 - n)
        # Gmax = 255
        # Gmin = 0
        # K1 = (gray_level-1) / (Gmax - Gmin)
        # K2 = 1 - K1 * Gmin
        # img = img * K1 + K2
        return img

    def showImage(self, title, src):
        cv2.namedWindow(title, 0)
        cv2.resizeWindow(title, 48, 48)
        cv2.imshow(title, src)
        cv2.waitKey(0)

    def getTextureFeatures(self, img, n=256):
        img = self.quantatizeImage(img, gray_level=n)
        row, col = img.shape
        features = []
        directions = 4

        for d in range(1, 9, 2):
            for i in xrange(0, directions):
                glcm = np.zeros((n, n))
                # features
                AUTOCORRELATION = 0
                CONTRAST = 0
                CORRELATION_I = 0
                CORRELATION_II = 0
                CLUSTER_PROMINENCE = 0
                CLUSTER_SHADE = 0
                DISSIMILARITY = 0
                ENERGY = 0
                ENTROPY = 0
                HOMOGENEITY_I = 0
                HOMOGENEITY_II = 0
                MAX_PROBABILITY = 0
                SUM_AVERAGE = 0
                SUM_ENTROPY = 0
                SUM_VARIANCE = 0
                DIF_VARIANCE = 0
                DIF_ENTROPY = 0
                INF_CORRELATION_I = 0
                INF_CORRELATION_II = 0

                #descriptors of texture
                PX = np.zeros(n)
                PY = np.zeros(n)
                PX_plus_PY = np.zeros(2 * n - 2)
                PX_minus_PY = np.zeros(n)
                HX = 0
                HY = 0
                HXY = 0
                HXY1 = 0
                HXY2 = 0
                MEAN_X = 0
                SIGMA_X = 0

                # GLCM with 0 degree
                if i == 0:
                    for x in xrange(0, row):
                        for y in xrange(0, col - d):
                            p1 = img[x, y]
                            p2 = img[x, y + d]
                            glcm[p1, p2] += 1
                            glcm[p2, p1] += 1
                # GLCM with 45 degree
                elif i == 1:
                    for x in xrange(d, row):
                        for y in xrange(0, col - d):
                            p1 = img[x, y]
                            p2 = img[x - d, y + d]
                            glcm[p1, p2] += 1
                            glcm[p2, p1] += 1
                # GLCM with 90 degree
                elif i == 2:
                    for x in xrange(d, row):
                        for y in xrange(0, col):
                            p1 = img[x, y]
                            p2 = img[x - d, y]
                            glcm[p1, p2] += 1
                            glcm[p2, p1] += 1
                # GLCM with 135 degree
                else:
                    for x in xrange(d, row):
                        for y in xrange(d, col):
                            p1 = img[x, y]
                            p2 = img[x - d, y - d]
                            glcm[p1, p2] += 1
                            glcm[p2, p1] += 1
                glcm /= np.sum(glcm)
                MAX_PROBABILITY = np.amax(glcm)
                for x in xrange(0, n):
                    for y in xrange(0, n):
                        AUTOCORRELATION += x * y *glcm[x, y]
                        ENERGY += glcm[x, y] ** 2
                        if glcm[x, y] > 0:
                            ENTROPY += -glcm[x, y] * np.log10(glcm[x, y])
                        CONTRAST += ((x - y) ** 2) * glcm[x, y]
                        DISSIMILARITY += glcm[x, y] * abs(x - y)
                        HOMOGENEITY_I += glcm[x, y] / (1 + abs(x - y))
                        HOMOGENEITY_II += glcm[x, y] / (1 + abs(x - y) ** 2)
                        MEAN_X += x * glcm[x, y]
                        PX[x] += glcm[x, y]
                        PY[x] += glcm[y, x]

                for k in xrange(2, 2*n):
                    for x in xrange(0, n):
                        for y in xrange(0, n):
                            if x + y == k:
                                PX_plus_PY[k-2] += glcm[x, y]
                for k in xrange(0, n):
                    for x in xrange(0, n):
                        for y in xrange(0, n):
                            if abs(x - y) == k:
                                PX_minus_PY[k] += glcm[x, y]
                for x in xrange(0, n):
                    if PX[x] > 0:
                        HX += -PX[x] * np.log10(PX[x])
                    if PY[x] > 0:
                        HY += -PY[x] * np.log10(PY[x])
                for x in xrange(0, n):
                    for y in xrange(0, n):
                        if glcm[x, y] > 0:
                            HXY += -glcm[x, y] * np.log10(glcm[x, y])
                        if glcm[x, y] * PX[x] * PY[x] > 0:
                            HXY1 += -glcm[x, y] * np.log10(PX[x] * PY[x])
                            HXY2 += -PX[x] * PY[x] * np.log10(PX[x] * PY[x])

                for x in xrange(2, 2 * n):
                    SUM_AVERAGE += x * PX_plus_PY[x-2]
                    if PX_plus_PY[x-2] > 0:
                        SUM_ENTROPY += - PX_plus_PY[x-2] * np.log10(PX_plus_PY[x - 2])

                for x in xrange(0, n):
                    SUM_VARIANCE += ((x - SUM_ENTROPY) ** 2) * PX_minus_PY[x - 2]

                for x in xrange(0, n):
                    DIF_VARIANCE += (x ** 2) * PX_minus_PY[x]
                    if PX_minus_PY[x] > 0:
                        DIF_ENTROPY += -PX_minus_PY[x] * np.log10(PX_minus_PY[x])

                INF_CORRELATION_I = (HXY - HXY1) / max(HX, HY)
                # if HXY2 - HXY > 0:
                #     print HXY2 - HXY
                #     INF_CORRELATION_II = np.sqrt(1 - np.exp(-2 * (HXY2 - HXY)))

                for x in xrange(0, n):
                    for y in xrange(0, n):
                        CLUSTER_PROMINENCE += ((x + y - MEAN_X ** 2) ** 4) * glcm[x, y]
                        CLUSTER_SHADE += ((x + y - MEAN_X ** 2) ** 3) * glcm[x, y]
                        SIGMA_X += ((x - MEAN_X) ** 2) * glcm[x, y]
                SIGMA_X = np.sqrt(SIGMA_X)
                for x in xrange(0, n):
                    for y in xrange(0, n):
                        if SIGMA_X * SIGMA_X != 0:
                            CORRELATION_I += (1 / (SIGMA_X * SIGMA_X)) * (x - MEAN_X) * (y - MEAN_X) * glcm[x, y]
                            CORRELATION_II += (1 / (SIGMA_X * SIGMA_X)) * (x * y * glcm[x, y] - MEAN_X ** 2)

                features.append(AUTOCORRELATION)
                features.append(CONTRAST)
                features.append(CORRELATION_I)
                features.append(CORRELATION_II)
                features.append(CLUSTER_PROMINENCE)
                features.append(CLUSTER_SHADE)
                features.append(DISSIMILARITY)
                features.append(ENERGY)
                features.append(ENTROPY)
                features.append(HOMOGENEITY_I)
                features.append(HOMOGENEITY_II)
                features.append(MAX_PROBABILITY)
                features.append(SUM_AVERAGE)
                features.append(SUM_ENTROPY)
                features.append(SUM_VARIANCE)
                features.append(DIF_VARIANCE)
                features.append(DIF_ENTROPY)
                features.append(INF_CORRELATION_I)
                # features.append(INF_CORRELATION_II)

        return features

    def preProcesseROI(self, ROI, window_size=48):
        ROI_enhanced = cv2.GaussianBlur(ROI.copy(), ksize=(3, 3), sigmaX=0)
        ROI_enhanced = cv2.resize(ROI_enhanced, (window_size, window_size))
        ROI_enhanced = cv2.equalizeHist(ROI_enhanced)
        return ROI_enhanced

    def preProcesseImage(self, img, positionX, positionY, radius, window_size=48):

        img_enhanced = img.copy()

        if radius == 0:
            radius = window_size / 2

        ROI_original = img_enhanced.copy()[1023 - positionY - radius:1023 - positionY + radius + 1,
                       positionX - radius:positionX + radius + 1]

        ROI_enhanced = self.preProcesseROI(ROI_original.copy(), window_size)

        return ROI_original, ROI_enhanced

    def rotate_image(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return bb_w - 2 * x, bb_h - 2 * y

    def crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]




        # for k in xrange(0, len(database)):
        #     image = cv2.imread(os.getcwd() + '\\MIAS\\' + database[k][0], 0)
        #     image_height, image_width = image.shape[0:2]
        #
        #     ROI_original, ROI_enhanced = obj.preProcesseImage(image.copy(),
        #                                                     database[k][1],
        #                                                     database[k][2],
        #                                                     database[k][4],
        #                                                     window_size=48)
        #
        #     if database[k][4] != 0:
        #         path = os.getcwd() + '\\ROI\\Abnormal\\'
        #     else:
        #         path = os.getcwd() + '\\ROI\\Normal\\'
        #
        #     for i in np.arange(0, 360, 90):
        #         image_orig = np.copy(ROI_original)
        #         image_rotated = obj.rotate_image(image_orig, i)
        #         image_rotated_cropped = obj.crop_around_center(image_rotated, *obj.largest_rotated_rect(image_width, image_height, math.radians(i)))
        #
        #         cv2.imwrite(path + str(i) + '\\' + database[k][0][:6] + '.jpg', image_rotated_cropped)
