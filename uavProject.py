import sys
import time
import os
import numpy as np
import cv2
import glob
from PIL import Image, ImageCms
from win32process import CREATE_NO_WINDOW

from imutils import contours
from skimage import measure
import argparse
import imutils

#def clahe():
#    path_dir = './input'
#    file_list = os.listdir(path_dir)
#    i = 0
#    images = glob.glob('input/*.jpg')
#    for fname in images:
#        coloredImg = cv2.imread(fname,1)
#        coloredImg2 = cv2.imread(fname,0)
#        
#        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#        cla = clahe.apply(coloredImg2)
#        clac = cv2.split(cla)
#        clacTotal = int(np.average(clac))

#        hsvc = cv2.split(coloredImg)
#        hsvTotal = int(np.average(hsvc))
#        
#        total = clacTotal - hsvTotal
#        
#        lower_black = np.array([total,total,total])
#        upper_black = np.array([clacTotal,clacTotal,clacTotal])
#        mask = cv2.inRange(coloredImg, lower_black, upper_black)
#        res = cv2.bitwise_and(coloredImg,coloredImg, mask= mask)
#        
#        median = cv2.blur(res,(11,11))
#        ret,medianb = cv2.threshold(median, hsvTotal+1, 255, cv2.THRESH_BINARY)
#        
#        grayImage = cv2.cvtColor(medianb, cv2.COLOR_BGR2GRAY)
#        ret, fImage = cv2.threshold(grayImage, hsvTotal+1, 255, cv2.THRESH_BINARY)
#        
#        cv2.imwrite('./output/3.clahe/' + str(file_list[i]), fImage)
#        i += 1

def gausThresh():
    labVal = []
    path_dir = './input'
    file_list = os.listdir(path_dir)
    i = 0
    images = glob.glob('./input/*.jpg')
    
    for fname in images:
        coloredImg = cv2.imread(fname, 1)
        gausImage = cv2.cvtColor(coloredImg, cv2.COLOR_BGR2LAB)
        l_channel,a_channel,b_channel = cv2.split(gausImage)
        labVal.append(np.max(l_channel)) 
        
    for gname in images:
        fname = cv2.imread(gname, 0)
        gausImg =cv2.adaptiveThreshold(fname,labVal[i],cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,45,2)
        cv2.imwrite('./output/1.gausThresh/' + str(file_list[i]), gausImg)
        i += 1
    

def edgeDet():
    labVal = []
    path_dir = './input'
    file_list = os.listdir(path_dir)
    i = 0
    images = glob.glob('./input/*.jpg')

    for fname in images:
        coloredImg = cv2.imread(fname, 1)
        gausImage = cv2.cvtColor(coloredImg, cv2.COLOR_BGR2LAB)
        l_channel,a_channel,b_channel = cv2.split(gausImage)
        labVal.append(np.max(l_channel)) 
        
    for fname in images:
        
        im = cv2.imread(fname)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, int(labVal[i]), 0)
        contours, hierarchy = cv2.findContours(
            
            
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(im, contours, -1, (0,int(labVal[i]),0), 3)
        cv2.imwrite('./output/2.edgeDet/' + str(file_list[i]), im)
        i += 1
        
def clahetest ():
    path_dir = './input'
    file_list = os.listdir(path_dir)
    i = 0
    images = glob.glob('input/*.jpg')
    for fname in images:
        image = cv2.imread(fname,1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        cla = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(cla, (25, 25), 0)
        
        thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh, None, iterations=4)
        thresh = cv2.dilate(thresh, None, iterations=8)
        
        cv2.imwrite('./output/3.clahe/' + str(file_list[i]), thresh)
        i += 1
        
def findCrack():
    gausThresh()
    edgeDet()
    clahetest()
    
    i = glob.glob('./input/*.jpg')
    k = 0
    for fanme in i:
        path_dir = './input'
        file_list = os.listdir(path_dir)
        Images = glob.glob('./input/*.jpg')
        Image = cv2.imread(Images[k],1)
        
        green = cv2.imread('green.jpg', cv2.IMREAD_COLOR)
        
        gaus_path_dir = './output/1.gausThresh'
        gaus_file_list = os.listdir(path_dir)
        gausImages = glob.glob('./output/1.gausThresh/*.jpg')
        gausImage = cv2.imread(gausImages[k],1)
        
        edge_path_dir = './output/2.edgeDet'
        edge_file_list = os.listdir(path_dir)
        edgeImages = glob.glob('./output/2.edgeDet/*.jpg')
        edgeImage = cv2.imread(edgeImages[k],1)
        
        clahe_path_dir = './output/3.clahe'
        clahe_file_list = os.listdir(path_dir)
        claheImages = glob.glob('./output/3.clahe/*.jpg')
        claheImage = cv2.imread(claheImages[k],1)
        
        res = cv2.bitwise_and(edgeImage,claheImage);
        res2 = cv2.bitwise_and(gausImage,res);
        res3 = cv2.bitwise_and(res2,green);
        
        crack = cv2.addWeighted(res3,float(0.5), Image,float(0.5),0)
        
        cv2.imwrite('./output/4.crackResult/' + str(gaus_file_list[k]), crack)
        
        k += 1

def topHat():
    path_dir = './input'
    file_list = os.listdir(path_dir)
    i = 0
    images = glob.glob('./input/*.jpg')
    
    for fname in images:
        im = cv2.imread(fname, 0)
        egde = cv2.Canny(im,110,150)
        kernel = np.ones((5,5),np.uint8)
        tophat = cv2.morphologyEx(egde, cv2.MORPH_TOPHAT, kernel)
        cv2.imwrite('./output/5.topHat/' + str(file_list[i]), tophat)
        i += 1
        
def findLeak():
    topHat()
    i = glob.glob('./input/*.jpg')
    k = 0
    for fanme in i:
        path_dir = './input'
        file_list = os.listdir(path_dir)
        Images = glob.glob('./input/*.jpg')
        Image = cv2.imread(Images[k],1)
        
        green = cv2.imread('green.jpg', cv2.IMREAD_COLOR)
        
        gaus_path_dir = './output/1.gausThresh'
        gaus_file_list = os.listdir(path_dir)
        gausImages = glob.glob('./output/1.gausThresh/*.jpg')
        gausImage = cv2.imread(gausImages[k],1)
        
        topHat_path_dir = './output/5.topHat'
        topHat_file_list = os.listdir(path_dir)
        topHatImages = glob.glob('./output/5.topHat/*.jpg')
        topHatImage = cv2.imread(topHatImages[k],1)
        
        res = cv2.bitwise_and(gausImage, topHatImage)
        res2 = cv2.bitwise_and(res,green);

        leak = cv2.addWeighted(res2,float(0.5), Image,float(0.5),0)
        
        cv2.imwrite('./output/6.leakResult/' + str(gaus_file_list[k]), leak)
        
        k += 1

def cannyEdge():
    path_dir = './input'
    file_list = os.listdir(path_dir)
    i = 0
    images = glob.glob('./input/*.jpg')
    
    for fname in images:
        im = cv2.imread(fname, 0)
        egde = cv2.Canny(im,0,150)
        cv2.imwrite('./output/7.cannyEdge/' + str(file_list[i]), egde)
        i += 1   

def findCoat():
    cannyEdge()
    i = glob.glob('./input/*.jpg')
    k = 0
    for fanme in i:
        path_dir = './input'
        file_list = os.listdir(path_dir)
        Images = glob.glob('./input/*.jpg')
        Image = cv2.imread(Images[k],1)
        
        green = cv2.imread('green.jpg', cv2.IMREAD_COLOR)
        
        gaus_path_dir = './output/1.gausThresh'
        gaus_file_list = os.listdir(path_dir)
        gausImages = glob.glob('./output/1.gausThresh/*.jpg')
        gausImage = cv2.imread(gausImages[k],1)
        
        canny_path_dir = './output/5.topHat'
        canny_file_list = os.listdir(path_dir)
        cannyImages = glob.glob('./output/7.cannyEdge/*.jpg')
        cannyImage = cv2.imread(cannyImages[k],1)
        
        res = cv2.bitwise_and(gausImage, cannyImage)
        res2 = cv2.bitwise_and(res,green);

        coat = cv2.addWeighted(res2,float(0.5), Image,float(0.5),0)
        
        cv2.imwrite('./output/8.coatResult/' + str(gaus_file_list[k]), coat)
        
        k += 1
        
def findALL():
    labVal = []
    findCrack()
    findLeak()
    findCoat()

def printCrack(imagePath, imageName):
    img = cv2.imread(imagePath)
    boundaries = [([3, 75, 1], [60, 254, 65])]
    
    for(lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
    
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask = mask)
    tot_pixel = output.size
    green_pixel = np.count_nonzero(output)
    crackPercentage = round(green_pixel * 100 / tot_pixel, 3)
    
    print("---------------- {}사진 교량 상태 분석--------------------".format(imageName))
    print("< 균열분석 >")
    print("균열분석 {}사진 균열 추측부분: ".format(imageName) + str(green_pixel) + " pixel")
    print("균열분석 {}사진 전체 사진크기: ".format(imageName) + str(tot_pixel) + " pixel")
    print("균열분석 {}사진 총 균열 추측비율: ".format(imageName) + str(crackPercentage) + "%")
    return (crackPercentage)

def printLeak(imagePath, imageName):
    img = cv2.imread(imagePath)
    
    boundaries = [([3, 75, 1], [60, 254, 65])]
    
    for(lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
    
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask = mask)
    
    tot_pixel = output.size
    green_pixel = np.count_nonzero(output)
    leakPercentage = round(green_pixel * 100 / tot_pixel, 3)
    print("< 누수분석 >")
    print("누수분석 {}사진 누수 추측부분: ".format(imageName) + str(green_pixel) + " pixel")
    print("누수분석 {}사진 전체 사진크기: ".format(imageName) + str(tot_pixel) + " pixel")
    print("누수분석 {}사진 총 누수 추측비율: ".format(imageName) + str(leakPercentage) + "%")
    return (leakPercentage)

    
def printCoat(imagePath, imageName):
    img = cv2.imread(imagePath)
    
    boundaries = [([3, 75, 1], [60, 254, 65])]
    
    for(lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
    
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask = mask)
    
    tot_pixel = output.size
    green_pixel = np.count_nonzero(output)
    leakPercentage = round(green_pixel * 100 / tot_pixel, 3)

    print("< 백태분석 >")
    print("백태분석 {}사진 백테 추측부분: ".format(imageName) + str(green_pixel) + " pixel")
    print("백태분석 {}사진 전체 사진크기: ".format(imageName) + str(tot_pixel) + " pixel")
    print("백태분석 {}사진 총 백태 추측비율: ".format(imageName) + str(leakPercentage) + "%")
    return (leakPercentage)


def printTotal(imageName, crackPercentage, leakPercentage, coatPercentage):
    img = cv2.imread(imageName)
    sumTotal =  (crackPercentage + leakPercentage + coatPercentage)
    totalPercentage =  round((crackPercentage + leakPercentage + coatPercentage)/3, 3)
    print("-교량 상태 분석-")
    print("{}사진은 도합 {}% 의 보수가 필요하며, 평균 {}% 입니다.".format(imageName, sumTotal, totalPercentage))
    print("-------------------------------------------------------------------------")

def printALL():
    now = time.localtime()
    sys.stdout = open('교량_상태_분석결과({}년{}월{}일_{}시{}분).txt'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min),'w')

    k = 0
    i = glob.glob('./input/*.jpg')

    for fname in i:
        crack_path_dir = './output/4.crackResult'
        crack_file_list = os.listdir(crack_path_dir)
        crackImages = glob.glob('./output/4.crackResult/*.jpg')
        crackImage = cv2.imread(crackImages[k],1)
        crackPercentage = printCrack(crackImages[k], crack_file_list[k])
    
        leak_path_dir = './output/6.leakResult'
        leak_file_list = os.listdir(leak_path_dir)
        leakImages = glob.glob('./output/6.leakResult/*.jpg')
        leakImage = cv2.imread(crack_file_list[k],1)
        leakPercentage = printLeak(leakImages[k], leak_file_list[k])
    
        coat_path_dir = './output/8.coatResult'
        coat_file_list = os.listdir(coat_path_dir)
        coatImages = glob.glob('./output/8.coatResult/*.jpg')
        coatImage = cv2.imread(crack_file_list[k],1)
        coatPercentage = printCoat(coatImages[k], coat_file_list[k])
    
        printTotal(coat_file_list[k], crackPercentage, leakPercentage, coatPercentage)
        k += 1

if __name__ == "__main__":
    findALL()
    printALL()
