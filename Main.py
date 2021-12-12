import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import NeuralNetwork as NN
import ImageProcessing as IT
import math
import datetime
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_curve, auc
import random



if __name__ == '__main__':

    obj = IT.ImageTexture()

    # database = [['mdb001.pgm', 535, 425, 1, 197],
    #             ['mdb080.pgm', 432, 149, 1, 20],
    #             ['mdb002.pgm', 522, 280, 1, 69],
    #             ['mdb005.pgm', 477, 133, 1, 30],
    #             ['mdb005.pgm', 500, 168, 1, 26],
    #             ['mdb010.pgm', 525, 425, 1, 33],
    #             ['mdb012.pgm', 471, 458, 1, 40],
    #             ['mdb013.pgm', 667, 365, 1, 31],
    #             ['mdb015.pgm', 595, 864, 1, 68],
    #             ['mdb017.pgm', 547, 573, 1, 48],
    #             ['mdb019.pgm', 653, 477, 1, 49],
    #             ['mdb021.pgm', 493, 125, 1, 49],
    #             ['mdb058.pgm', 318, 359, 1, 27],
    #             ['mdb072.pgm', 266, 517, 1, 28],
    #             ['mdb075.pgm', 468, 717, 1, 23],
    #             ['mdb090.pgm', 510, 547, 1, 49],
    #             ['mdb092.pgm', 423, 662, 1, 43],
    #             ['mdb206.pgm', 368, 200, 1, 17],
    #             ['mdb207.pgm', 571, 564, 1, 19],
    #             ['mdb209.pgm', 647, 503, 1, 87],
    #             ['mdb213.pgm', 547, 520, 1, 45],
    #             ['mdb214.pgm', 582, 916, 1, 11],
    #             ['mdb030.pgm', 322, 676, 1, 43],
    #             ['mdb032.pgm', 388, 742, 1, 66],
    #             ['mdb063.pgm', 546, 463, 1, 33],
    #             ['mdb069.pgm', 462, 406, 1, 44],
    #             ['mdb081.pgm', 492, 473, 1, 131],
    #             ['mdb095.pgm', 466, 517, 1, 29],
    #             ['mdb083.pgm', 544, 194, 1, 38],
    #             ['mdb091.pgm', 680, 494, 1, 20],
    #             ['mdb097.pgm', 612, 297, 1, 34],
    #             ['mdb099.pgm', 714, 340, 1, 23],
    #             ['mdb102.pgm', 415, 460, 1, 38],
    #             ['mdb104.pgm', 357, 365, 1, 50],
    #             ['mdb105.pgm', 516, 279, 1, 98],
    #             ['mdb107.pgm', 600, 621, 1, 111],
    #             ['mdb110.pgm', 190, 427, 1, 51],
    #             ['mdb111.pgm', 505, 575, 1, 107],
    #             ['mdb115.pgm', 461, 532, 1, 117],
    #             ['mdb117.pgm', 480, 576, 1, 84],
    #             ['mdb120.pgm', 423, 262, 1, 79],
    #             ['mdb121.pgm', 492, 434, 1, 87],
    #             ['mdb124.pgm', 366, 620, 1, 33],
    #             ['mdb125.pgm', 700, 552, 1, 60],
    #             ['mdb126.pgm', 191, 549, 1, 23],
    #             ['mdb127.pgm', 523, 551, 1, 48],
    #             ['mdb130.pgm', 220, 552, 1, 28],
    #             ['mdb132.pgm', 252, 788, 1, 52],
    #             ['mdb132.pgm', 335, 766, 1, 18],
    #             ['mdb134.pgm', 469, 728, 1, 49],
    #             ['mdb141.pgm', 470, 759, 1, 29],
    #             ['mdb142.pgm', 347, 636, 1, 26],
    #             ['mdb144.pgm', 233, 994, 1, 29],
    #             ['mdb144.pgm', 313, 540, 1, 27],
    #             ['mdb145.pgm', 669, 543, 1, 49],
    #             ['mdb148.pgm', 326, 607, 1, 174],
    #             ['mdb150.pgm', 351, 661, 1, 62],
    #             ['mdb152.pgm', 675, 486, 1, 48],
    #             ['mdb155.pgm', 448, 480, 1, 95],
    #             ['mdb158.pgm', 540, 565, 1, 88],
    #             ['mdb160.pgm', 536, 519, 1, 61],
    #             ['mdb163.pgm', 391, 365, 1, 50],
    #             ['mdb165.pgm', 537, 490, 1, 42],
    #             ['mdb167.pgm', 574, 657, 1, 35],
    #             ['mdb122.pgm', 525, 425, -1, 0],
    #             ['mdb166.pgm', 525, 425, -1, 0],
    #             ['mdb168.pgm', 525, 425, -1, 0],
    #             ['mdb169.pgm', 525, 425, -1, 0],
    #             ['mdb164.pgm', 525, 425, -1, 0],
    #             ['mdb161.pgm', 525, 425, -1, 0],
    #             ['mdb162.pgm', 525, 425, -1, 0],
    #             ['mdb151.pgm', 525, 425, -1, 0],
    #             ['mdb159.pgm', 525, 425, -1, 0],
    #             ['mdb156.pgm', 525, 425, -1, 0],
    #             ['mdb157.pgm', 525, 425, -1, 0],
    #             ['mdb153.pgm', 525, 425, -1, 0],
    #             ['mdb154.pgm', 525, 425, -1, 0],
    #             ['mdb149.pgm', 525, 425, -1, 0],
    #             ['mdb143.pgm', 525, 425, -1, 0],
    #             ['mdb146.pgm', 525, 425, -1, 0],
    #             ['mdb147.pgm', 525, 425, -1, 0],
    #             ['mdb133.pgm', 525, 425, -1, 0],
    #             ['mdb131.pgm', 525, 425, -1, 0],
    #             ['mdb123.pgm', 525, 425, -1, 0],
    #             ['mdb135.pgm', 525, 425, -1, 0],
    #             ['mdb136.pgm', 525, 425, -1, 0],
    #             ['mdb137.pgm', 525, 425, -1, 0],
    #             ['mdb138.pgm', 525, 425, -1, 0],
    #             ['mdb139.pgm', 525, 425, -1, 0],
    #             ['mdb128.pgm', 525, 425, -1, 0],
    #             ['mdb129.pgm', 525, 425, -1, 0],
    #             ['mdb100.pgm', 525, 425, -1, 0],
    #             ['mdb116.pgm', 525, 425, -1, 0],
    #             ['mdb118.pgm', 525, 425, -1, 0],
    #             ['mdb119.pgm', 525, 425, -1, 0],
    #             ['mdb112.pgm', 525, 425, -1, 0],
    #             ['mdb113.pgm', 525, 425, -1, 0],
    #             ['mdb114.pgm', 525, 425, -1, 0],
    #             ['mdb106.pgm', 525, 425, -1, 0],
    #             ['mdb108.pgm', 525, 425, -1, 0],
    #             ['mdb109.pgm', 525, 425, -1, 0],
    #             ['mdb103.pgm', 525, 425, -1, 0],
    #             ['mdb101.pgm', 525, 425, -1, 0],
    #             ['mdb096.pgm', 525, 425, -1, 0],
    #             ['mdb098.pgm', 525, 425, -1, 0],
    #             ['mdb082.pgm', 525, 425, -1, 0],
    #             ['mdb172.pgm', 525, 425, -1, 0],
    #             ['mdb173.pgm', 525, 425, -1, 0],
    #             ['mdb174.pgm', 525, 425, -1, 0],
    #             ['mdb176.pgm', 525, 425, -1, 0],
    #             ['mdb177.pgm', 525, 425, -1, 0],
    #             ['mdb003.pgm', 535, 425, -1, 0],
    #             ['mdb004.pgm', 522, 280, -1, 0],
    #             ['mdb006.pgm', 477, 133, -1, 0],
    #             ['mdb007.pgm', 500, 168, -1, 0],
    #             ['mdb008.pgm', 525, 425, -1, 0],
    #             ['mdb064.pgm', 525, 425, -1, 0],
    #             ['mdb065.pgm', 525, 425, -1, 0],
    #             ['mdb066.pgm', 525, 425, -1, 0],
    #             ['mdb067.pgm', 525, 425, -1, 0],
    #             ['mdb068.pgm', 525, 425, -1, 0],
    #             ['mdb070.pgm', 525, 425, -1, 0],
    #             ['mdb071.pgm', 525, 425, -1, 0],
    #             ['mdb073.pgm', 525, 425, -1, 0],
    #             ['mdb074.pgm', 525, 425, -1, 0],
    #             ['mdb031.pgm', 525, 425, -1, 0],
    #             ['mdb031.pgm', 525, 425, -1, 0],
    #             ['mdb033.pgm', 525, 425, -1, 0],
    #             ['mdb034.pgm', 525, 425, -1, 0],
    #             ['mdb035.pgm', 525, 425, -1, 0],
    #             ['mdb036.pgm', 525, 425, -1, 0],
    #             ['mdb037.pgm', 525, 425, -1, 0],
    #             ['mdb038.pgm', 525, 425, -1, 0],
    #             ['mdb039.pgm', 525, 425, -1, 0],
    #             ['mdb040.pgm', 525, 425, -1, 0],
    #             ['mdb041.pgm', 525, 425, -1, 0],
    #             ['mdb042.pgm', 525, 425, -1, 0],
    #             ['mdb043.pgm', 525, 425, -1, 0],
    #             ['mdb044.pgm', 525, 425, -1, 0],
    #             ['mdb045.pgm', 525, 425, -1, 0],
    #             ['mdb046.pgm', 525, 425, -1, 0],
    #             ['mdb047.pgm', 525, 425, -1, 0],
    #             ['mdb048.pgm', 525, 425, -1, 0],
    #             ['mdb049.pgm', 525, 425, -1, 0],
    #             ['mdb050.pgm', 525, 425, -1, 0],
    #             ['mdb051.pgm', 525, 425, -1, 0],
    #             ['mdb052.pgm', 525, 425, -1, 0],
    #             ['mdb053.pgm', 525, 425, -1, 0],
    #             ['mdb054.pgm', 525, 425, -1, 0],
    #             ['mdb055.pgm', 525, 425, -1, 0],
    #             ['mdb056.pgm', 525, 425, -1, 0],
    #             ['mdb057.pgm', 525, 425, -1, 0],
    #             ['mdb060.pgm', 525, 425, -1, 0],
    #             ['mdb061.pgm', 525, 425, -1, 0],
    #             ['mdb062.pgm', 525, 425, -1, 0],
    #             ['mdb076.pgm', 525, 425, -1, 0],
    #             ['mdb077.pgm', 525, 425, -1, 0],
    #             ['mdb078.pgm', 525, 425, -1, 0],
    #             ['mdb079.pgm', 525, 425, -1, 0],
    #             ['mdb084.pgm', 525, 425, -1, 0],
    #             ['mdb085.pgm', 525, 425, -1, 0],
    #             ['mdb086.pgm', 525, 425, -1, 0],
    #             ['mdb087.pgm', 525, 425, -1, 0],
    #             ['mdb088.pgm', 525, 425, -1, 0],
    #             ['mdb089.pgm', 525, 425, -1, 0],
    #             ['mdb093.pgm', 525, 425, -1, 0],
    #             ['mdb094.pgm', 525, 425, -1, 0]]

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



    database_validation = [['mdb178.pgm', 492, 600, 1, 70],
                      ['mdb170.pgm', 489, 480, 1, 82],
                      ['mdb171.pgm', 462, 627, 1, 62],
                      ['mdb175.pgm', 592, 670, 1, 33],
                      ['mdb023.pgm', 538, 681, 1, 29],
                      ['mdb025.pgm', 674, 443, 1, 79],
                      ['mdb028.pgm', 338, 314, 1, 56],
                      ['mdb179.pgm', 600, 514, 1, 67],
                      ['mdb181.pgm', 519, 362, 1, 54],
                      ['mdb184.pgm', 352, 624, 1, 114],
                      ['mdb186.pgm', 403, 524, 1, 47],
                      ['mdb188.pgm', 406, 617, 1, 61],
                      ['mdb190.pgm', 512, 621, 1, 31],
                      ['mdb191.pgm', 594, 516, 1, 41],
                      ['mdb193.pgm', 399, 563, 1, 132],
                      ['mdb195.pgm', 725, 129, 1, 26],
                      ['mdb198.pgm', 568, 612, 1, 93],
                      ['mdb199.pgm', 641, 177, 1, 31],
                      ['mdb202.pgm', 557, 772, 1, 37],
                      ['mdb204.pgm', 336, 399, 1, 21],
                      ['mdb211.pgm', 680, 327, 1, 13],
                      ['mdb209.pgm', 647, 503, 1, 87],
                      ['mdb212.pgm', 687, 882, 1, 3],
                      ['mdb218.pgm', 519, 629, 1, 8],
                      ['mdb219.pgm', 546, 756, 1, 29],
                      ['mdb222.pgm', 398, 427, 1, 17],
                      ['mdb223.pgm', 523, 482, 1, 29],
                      ['mdb223.pgm', 591, 529, 1, 6],
                      ['mdb226.pgm', 287, 610, 1, 7],
                      ['mdb226.pgm', 329, 550, 1, 25],
                      ['mdb226.pgm', 531, 721, 1, 8],
                      ['mdb227.pgm', 504, 467, 1, 9],
                      ['mdb231.pgm', 603, 538, 1, 44],
                      ['mdb236.pgm', 276, 824, 1, 14],
                      ['mdb238.pgm', 522, 553, 1, 17],
                    ['mdb239.pgm', 645, 755, 1, 40],
                           ['mdb240.pgm', 643, 614, 1, 23],
                           ['mdb241.pgm', 453, 678, 1, 38],
                           ['mdb244.pgm', 466, 567, 1, 52],
                           ['mdb248.pgm', 378, 601, 1, 10],
                           ['mdb249.pgm', 544, 508, 1, 48],
                           ['mdb249.pgm', 575, 639, 1, 64],
                           ['mdb252.pgm', 439, 367, 1, 23],
                           ['mdb253.pgm', 733, 564, 1, 28],
                           ['mdb256.pgm', 400, 484, 1, 37],
                           ['mdb264.pgm', 596, 431, 1, 36],
                           ['mdb265.pgm', 593, 498, 1, 60],
                           ['mdb267.pgm', 793, 481, 1, 56],
                           ['mdb270.pgm', 356, 945, 1, 72],
                           ['mdb271.pgm', 784, 270, 1, 68],
                           ['mdb274.pgm', 127, 505, 1, 123],
                           ['mdb290.pgm', 337, 353, 1, 45],
                           ['mdb312.pgm', 240, 263, 1, 20],
                           ['mdb314.pgm', 518, 191, 1, 39],
                           ['mdb315.pgm', 516, 447, 1, 93],
                           ['mdb291.pgm', 525, 425, -1, 0],
                           ['mdb316.pgm', 525, 425, -1, 0],
                           ['mdb317.pgm', 525, 425, -1, 0],
                           ['mdb318.pgm', 525, 425, -1, 0],
                           ['mdb319.pgm', 525, 425, -1, 0],
                           ['mdb320.pgm', 525, 425, -1, 0],
                           ['mdb321.pgm', 525, 425, -1, 0],
                           ['mdb322.pgm', 525, 425, -1, 0],
                           ['mdb313.pgm', 525, 425, -1, 0],
                           ['mdb292.pgm', 525, 425, -1, 0],
                           ['mdb293.pgm', 525, 425, -1, 0],
                           ['mdb294.pgm', 525, 425, -1, 0],
                           ['mdb295.pgm', 525, 425, -1, 0],
                           ['mdb296.pgm', 525, 425, -1, 0],
                           ['mdb297.pgm', 525, 425, -1, 0],
                           ['mdb298.pgm', 525, 425, -1, 0],
                           ['mdb299.pgm', 525, 425, -1, 0],
                           ['mdb300.pgm', 525, 425, -1, 0],
                           ['mdb301.pgm', 525, 425, -1, 0],
                           ['mdb302.pgm', 525, 425, -1, 0],
                           ['mdb303.pgm', 525, 425, -1, 0],
                           ['mdb304.pgm', 525, 425, -1, 0],
                           ['mdb305.pgm', 525, 425, -1, 0],
                           ['mdb306.pgm', 525, 425, -1, 0],
                           ['mdb307.pgm', 525, 425, -1, 0],
                           ['mdb308.pgm', 525, 425, -1, 0],
                           ['mdb309.pgm', 525, 425, -1, 0],
                           ['mdb275.pgm', 525, 425, -1, 0],
                           ['mdb276.pgm', 525, 425, -1, 0],
                           ['mdb277.pgm', 525, 425, -1, 0],
                           ['mdb278.pgm', 525, 425, -1, 0],
                           ['mdb279.pgm', 525, 425, -1, 0],
                           ['mdb280.pgm', 525, 425, -1, 0],
                           ['mdb281.pgm', 525, 425, -1, 0],
                           ['mdb282.pgm', 525, 425, -1, 0],
                           ['mdb283.pgm', 525, 425, -1, 0],
                           ['mdb284.pgm', 525, 425, -1, 0],
                           ['mdb285.pgm', 525, 425, -1, 0],
                           ['mdb286.pgm', 525, 425, -1, 0],
                           ['mdb287.pgm', 525, 425, -1, 0],
                           ['mdb288.pgm', 525, 425, -1, 0],
                           ['mdb289.pgm', 525, 425, -1, 0],
                           ['mdb272.pgm', 525, 425, -1, 0],
                           ['mdb273.pgm', 525, 425, -1, 0],
                           ['mdb257.pgm', 525, 425, -1, 0],
                           ['mdb268.pgm', 525, 425, -1, 0],
                           ['mdb269.pgm', 525, 425, -1, 0],
                           ['mdb266.pgm', 525, 425, -1, 0],
                           ['mdb258.pgm', 525, 425, -1, 0],
                           ['mdb259.pgm', 525, 425, -1, 0],
                           ['mdb260.pgm', 525, 425, -1, 0],
                           ['mdb261.pgm', 525, 425, -1, 0],
                           ['mdb262.pgm', 525, 425, -1, 0],
                           ['mdb263.pgm', 525, 425, -1, 0],
                           ['mdb254.pgm', 525, 425, -1, 0],
                           ['mdb255.pgm', 525, 425, -1, 0],
                           ['mdb250.pgm', 525, 425, -1, 0],
                           ['mdb251.pgm', 525, 425, -1, 0],
                           ['mdb246.pgm', 525, 425, -1, 0],
                           ['mdb247.pgm', 525, 425, -1, 0],
                           ['mdb242.pgm', 525, 425, -1, 0],
                           ['mdb243.pgm', 525, 425, -1, 0],
                      ['mdb228.pgm', 525, 425, -1, 0],
                      ['mdb229.pgm', 525, 425, -1, 0],
                      ['mdb230.pgm', 525, 425, -1, 0],
                      ['mdb234.pgm', 525, 425, -1, 0],
                      ['mdb235.pgm', 525, 425, -1, 0],
                      ['mdb237.pgm', 525, 425, -1, 0],
                      ['mdb232.pgm', 525, 425, -1, 0],
                      ['mdb224.pgm', 525, 425, -1, 0],
                      ['mdb225.pgm', 525, 425, -1, 0],
                      ['mdb215.pgm', 525, 425, -1, 0],
                      ['mdb220.pgm', 525, 425, -1, 0],
                      ['mdb221.pgm', 525, 425, -1, 0],
                      ['mdb217.pgm', 525, 425, -1, 0],
                      ['mdb210.pgm', 525, 425, -1, 0],
                      ['mdb220.pgm', 525, 425, -1, 0],
                      ['mdb221.pgm', 525, 425, -1, 0],
                      ['mdb194.pgm', 525, 425, -1, 0],
                      ['mdb208.pgm', 525, 425, -1, 0],
                      ['mdb196.pgm', 525, 425, -1, 0],
                      ['mdb197.pgm', 525, 425, -1, 0],
                      ['mdb200.pgm', 525, 425, -1, 0],
                      ['mdb201.pgm', 525, 425, -1, 0],
                      ['mdb203.pgm', 525, 425, -1, 0],
                      ['mdb205.pgm', 525, 425, -1, 0],
                      ['mdb009.pgm', 525, 425, -1, 0],
                      ['mdb192.pgm', 525, 425, -1, 0],
                      ['mdb189.pgm', 525, 425, -1, 0],
                      ['mdb187.pgm', 525, 425, -1, 0],
                      ['mdb182.pgm', 525, 425, -1, 0],
                      ['mdb183.pgm', 525, 425, -1, 0],
                      ['mdb185.pgm', 525, 425, -1, 0],
                      ['mdb180.pgm', 525, 425, -1, 0],
                      ['mdb011.pgm', 525, 425, -1, 0],
                      ['mdb014.pgm', 525, 425, -1, 0],
                      ['mdb016.pgm', 525, 425, -1, 0],
                      ['mdb018.pgm', 525, 425, -1, 0],
                      ['mdb020.pgm', 525, 425, -1, 0],
                      ['mdb022.pgm', 525, 425, -1, 0],
                      ['mdb024.pgm', 525, 425, -1, 0],
                      ['mdb026.pgm', 525, 425, -1, 0],
                      ['mdb027.pgm', 525, 425, -1, 0],
                      ['mdb029.pgm', 525, 425, -1, 0]]

    M = 18*4*4

    window_size = 48

    database_train_len = 57 + 57
    X_Train = np.zeros((database_train_len, M), dtype=float)
    Y_Train = np.zeros((database_train_len, 1), dtype=float)
    X_Validation = np.zeros((len(database_validation), M), dtype=float)

    gray_level = 64
    newTraining = False

    print 'Training Set Length: ' + str(database_train_len)
    print 'Validation Set Length: ' + str(len(database_validation))

    # Getting Abnormal Features
    print 'Step: Getting features from Training Set - Abnormal ROIs'
    if newTraining:
        path = os.getcwd() + '\\ROI\\Abnormal\\'
        filenames = os.listdir(path + '0\\')
        # index of X_Train to be incremented
        k = 0
        for i in xrange(0, len(filenames)):
            for angle in xrange(0, 360, 360):
                ROI_original = cv2.imread(path + str(angle) + '\\' + filenames[i], 0)
                ROI_enhanced = obj.preProcesseROI(ROI_original.copy(), window_size)
                X_Train[k, 0: M] = obj.getTextureFeatures(ROI_enhanced.copy(), n=gray_level)
                k += 1
                # if angle == 0:
                #     plt.figure()
                #     plt.subplot(131)
                #     plt.imshow(ROI_original, cmap='gray')
                #     plt.subplot(132)
                #     plt.imshow(ROI_enhanced, cmap='gray')
                #     plt.show()
        for i in xrange(0, 57):
            Y_Train[i] = [1]

        # Getting Normal Features
        print 'Step: Getting features from Training Set - Normal ROIs'
        path = os.getcwd() + '\\ROI\\Normal\\'
        filenames = os.listdir(path + '0\\')
        for i in xrange(0, 57):
            for angle in xrange(0, 360, 360):
                ROI_original = cv2.imread(path + str(angle) + '\\' + filenames[i], 0)
                ROI_enhanced = obj.preProcesseROI(ROI_original.copy(), window_size)

                X_Train[k, 0: M] = obj.getTextureFeatures(ROI_enhanced.copy(), n=gray_level)
                k += 1
                # if angle == 0:
                #     plt.figure()
                #     plt.subplot(131)
                #     plt.imshow(ROI_original, cmap='gray')
                #     plt.subplot(132)
                #     plt.imshow(ROI_enhanced, cmap='gray')
                #     plt.show()
        for i in xrange(57, database_train_len):
            Y_Train[i] = [0]

        np.savetxt(os.getcwd() + '\\X_Train.txt', X_Train.copy(), fmt='%.14f')
        np.savetxt(os.getcwd() + '\\Y_Train.txt', Y_Train.copy())

    else:
        print 'Step: Getting features from Training Set - Abnormal ROIs'
        print 'Step: Getting features from Training Set - Normal ROIs'
        X_Train = np.loadtxt(os.getcwd() + '\\X_Train.txt')
        Y_Train = np.loadtxt(os.getcwd() + '\\Y_Train.txt')

    #Getting Features From Validation Set
    print 'Step: Getting features from Validation Set'
    if newTraining:
        for i in xrange(0, len(database_validation)):
            filename = database_validation[i][0]
            positionX = database_validation[i][1]
            positionY = database_validation[i][2]
            radius = database_validation[i][4]

            img_original = cv2.imread(os.getcwd() + '\\MIAS\\' + filename, 0)
            ROI_original, \
            ROI_enhanced = obj.preProcesseImage(img_original.copy(), positionX, positionY, 0, window_size=window_size)

            X_Validation[i, 0: M] = obj.getTextureFeatures(ROI_enhanced.copy(), n=gray_level)
        np.savetxt(os.getcwd() + '\\X_validation.txt', X_Validation.copy(), fmt='%.14f')
    else:
        X_Validation = np.loadtxt(os.getcwd() + '\\X_validation.txt')

    # Normalization of X_Train e X_validation to be processed by MLP
    mean = []
    std = []
    Max = []
    Min = []
    for i in xrange(0, M):
        # mean.append(X_Train[:, i].mean())
        # std.append(X_Train[:, i].std())
        # X_Train[:, i] = (X_Train[:, i] - mean[i]) / std[i]
        # X_Validation[:, i] = (X_Validation[:, i] - mean[i]) / std[i]
        Max.append(np.amax(X_Train[:, i]))
        Min.append(np.amin(X_Train[:, i]))
        X_Train[:, i] = (X_Train[:, i] - Min[i]) / (Max[i] - Min[i])
        X_Validation[:, i] = (X_Validation[:, i] - Min[i]) / (Max[i] - Min[i])

    print 'Step: Training ANN'
    n_components = 8
    mlp = NN.MLP(inputSize=n_components, hiddenSize=16, outputSize=1, n_iter=10000, eta=0.01, u=0.9)
    pca = PCA(n_components=n_components)
    X_Train = pca.fit_transform(X_Train)
    X_Validation = pca.transform(X_Validation)
    np.savetxt(os.getcwd() + '\\pca_X_Train.txt', X_Train, fmt='%.14f')
    np.savetxt(os.getcwd() + '\\pca_X_Validation.txt', X_Validation, fmt='%.14f')
    # kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.01)
    # X_Train = kpca.fit_transform(X_Train)
    # X_Validation = kpca.transform(X_Validation)

    if newTraining:
        mlp.costFunctionPrime(X_Train, Y_Train)
        np.savetxt(os.getcwd() + '\\W164.txt', mlp.W1, fmt='%.14f')
        np.savetxt(os.getcwd() + '\\W264.txt', mlp.W2, fmt='%.14f')
    else:
        mlp.costFunctionPrime(X_Train, Y_Train.reshape((Y_Train.shape[0], 1)))

    acc = 0
    cancerVerdadeiro = 0
    normalVerdadeiro = 0
    cancerErrado = 0
    normalErrado = 0
    score = []
    y = []
    print 'Step: Working on the Validation Set'
    for i in xrange(0, len(database_validation)):
        out = mlp.forward(X_Validation[i])
        if database_validation[i][3] == 1:
            cancerVerdadeiro += 1
            if out > 0.5:
                acc += 1
            else:
                cancerErrado += 1
        else:
            normalVerdadeiro += 1
            if out <= 0.5:
                acc += 1
            else:
                normalErrado += 1
        score.append(out)
        y.append(database_validation[i][3])

    print 'Accuracy: ' + str(100 * float(acc) / float(len(database_validation)))
    print 'Acertos: ' + str(acc) + ' de ' + str(len(database_validation))
    print 'Cancer classificados como normais: ' + str(cancerErrado) + ' de '+str(cancerVerdadeiro)
    print 'Normal classificados como cancer: ' + str(normalErrado) + ' de '+str(normalVerdadeiro)

    obj.plotGraphic(range(1, len(mlp.cost) + 1),
                    mlp.cost,
                    marker='o',
                    xlabel='Epochs',
                    ylabel='Sum of square-errors')
    #
    #
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()