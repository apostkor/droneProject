import sys
import time
import os
import numpy as np
import cv2
import glob

def printSingle():
    now = time.localtime()
    sys.stdout = open('특정_상태_분석결과({}년{}월{}일_{}시{}분).txt'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min),'w')

    k = 0
    i = glob.glob('./inputEXTRA/*.jpg')

    for fname in i:
        single_path_dir = './inputEXTRA'
        single_file_list = os.listdir(single_path_dir)
        singleImages = glob.glob('./inputEXTRA/*.jpg')
        singleImage = cv2.imread(singleImages[k],1)
        singlePercentage = printSin(singleImages[k], single_file_list[k])
        k += 1

def printSin (imagePath, imageName):
    img = cv2.imread(imagePath)
    
    boundaries = [([3, 75, 1], [60, 254, 65])]
    
    for(lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
    
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask = mask)
    
    tot_pixel = output.size
    green_pixel = np.count_nonzero(output)
    singlePercentage = round(green_pixel * 100 / tot_pixel, 3)

    print("< 특정분석 >")
    print("특정분석 {}사진 추측부분: ".format(imageName) + str(green_pixel) + " pixel")
    print("특정분석 {}사진 전체 사진크기: ".format(imageName) + str(tot_pixel) + " pixel")
    print("특정분석 {}사진 추측비율: ".format(imageName) + str(singlePercentage) + "%")
    return (singlePercentage)
        
if __name__ == "__main__":
    print("특정 상태 분석을 시작합니다. 분석이 완료되면 콘솔창은 자동 종료됩니다.")
    printSingle()
