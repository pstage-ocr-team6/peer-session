#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv  
from collections import Counter
from pprint import pprint
#%%
img_dir='/opt/ml/input/data/train_dataset/images'
# %%
img=cv2.imread(os.path.join(img_dir,'train_00016.jpg'))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=500,lines=np.array([]), minLineLength=100,maxLineGap=80)
a,b,c = lines.shape
plt.imshow(gray)
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
plt.figure()
plt.imshow(gray)
# %%
image=cv2.imread(os.path.join(img_dir,'train_00016.jpg'))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
result = cv2.bitwise_and(image,image,mask=thresh)
result[thresh==0] = (255,255,255)

plt.imshow(thresh)
plt.figure()
plt.imshow(result)
#%%
######## 선 제거
image = cv2.imread(os.path.join(img_dir,'train_63334.jpg'))  # 이미지 로드
h,w=image.shape[:2] # h, w
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # gray 채널
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #OTSU 방법을 이용해 binary 이미지 변환
plt.title('original') 
plt.imshow(image)
plt.figure()
plt.title('binary') 
plt.imshow(thresh, cmap='gray')
plt.figure()

kernel = np.ones((3, 3), np.uint8)
dilation_image = cv2.dilate(thresh, kernel, iterations=1) # dilate 연산 
plt.title('dilation imgae') 
plt.imshow(dilation_image, cmap='gray')

# 가록 선을 찾을 커널 정하기 -> 가로 선을 찾을 꺼나 (x,y)에서 x를 더 크게 잡아야된다.
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1)) 
# 모포로지 연산 (위의 dilate와 유사한 시리즈)
detected_lines = cv2.morphologyEx(dilation_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
plt.figure()
plt.title('detected_lines') 
plt.imshow(detected_lines, cmap='gray')

# 선들의 contour 들을 찾기
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

temp_height=10
for c in cnts: # 모든 contour 마다
    temp=c.flatten()
    if max(temp[::2])-min(temp[::2]) >w/2: # contour의 w가 전체 이미지의 w/2보다 크면 삭제
        temp_height=max(temp[1::2])-min(temp[1::2])
        cv2.drawContours(dilation_image, [c], -1, (0,0,0), -1) # 안을 검은색으로 채워줌
plt.figure()
print(dilation_image.shape)
plt.title('remove_long line') 
plt.imshow(dilation_image, cmap='gray') # 긴 선을 제거 하고 나니 글자에 빈칸이 생김 (remove_long line 참고)

# 빈칸을 채우기 위한 모포로지 연산(CLOSE)
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,temp_height//10))
result = cv2.morphologyEx(dilation_image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
plt.figure()
plt.title('final') 
plt.imshow(result, cmap='gray')
# %%
######## 굽힙 제거(fail)
src = cv2.imread(os.path.join(img_dir,'train_00007.jpg'))
height, width, channel = src.shape

srcPoint = np.array([[100, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(dstPoint, dstPoint)
dst = cv2.warpPerspective(src, matrix, (width, height))
plt.imshow(src)
plt.figure()
plt.imshow(dst)

# %%
# [19282, 77709] , [63334, 76558]
img=cv2.imread(os.path.join(img_dir,'train_83399.jpg'))
plt.imshow(img)
plt.figure()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # gray 채널

#### 이미지 이진화 어떤 기법을 쓸진 미정
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
# edge =cv2.Canny(gray,0,65)
# plt.imshow(edge,cmap='gray')
# plt.figure()
thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 15,2)
        
plt.imshow(thresh,cmap='gray') 
plt.figure()
white_img=np.zeros_like(img)
# kernel = np.ones((3, 3), np.uint8)
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilation_image = cv2.erode(thresh, kernel, iterations=1) 
plt.imshow(cv2.cvtColor(dilation_image,cv2.COLOR_GRAY2RGB) ) 
plt.figure()
opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
plt.imshow(cv2.cvtColor(opening,cv2.COLOR_GRAY2RGB) ) 
plt.figure()
# tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# plt.imshow(tophat)
# plt.figure()

# %%
def img_equal_clahe_yuv(img):
    '''
        설명 : YUV 형식으로 바꾼 다음 작업
        input : opencv BGR  이미지
        output : equalizeHist 적용 이미지, clahe 적용 이미지
    '''
    img_yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

    img_eq=img_yuv.copy()
    img_eq[:,:,0]=cv2.equalizeHist(img_eq[:,:,0])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

    img_clahe = img_yuv.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
    img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

    return img_eq,img_clahe
#%%
def img_normalize(img):
    '''
        input : opencv BGR  이미지
    '''
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm
#%%
def show_hist(img):
    '''
        input : 이미지
    '''
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()
    plt.figure()
#%%
def img_clahe_luminus(img):
    '''
        설명 : LAB 형식으로 바꾼 다음 작업
        input : opencv BGR  이미지
        output : equalizeHist 적용 이미지, clahe 적용 이미지
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # luminosity(명도)채널을 얻기 위해 채널을 BGR->LAB 로 바꿈
    l, a, b = cv2.split(lab) # 채널 분리

    el=cv2.equalizeHist(l)
    img_eq=cv2.merge((el,a,b))
    img_eq=cv2.cvtColor(img_eq, cv2.COLOR_LAB2BGR)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # 히스토그램 균등화 시키기 http://www.gisdeveloper.co.kr/?p=6652
    cl = clahe.apply(l) # 명도 채널에 적용
    limg = cv2.merge((cl, a, b)) # 채널 합치기
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) 
    return img_eq,img_clahe 
# %%

# BGR  -> YUV LAB hsv
img=cv2.imread(os.path.join(img_dir,'train_83399.jpg'))
plt.imshow(img)
plt.figure()
# show_hist(img)

img_eq,img_clahe=img_equal_clahe_yuv(img)
plt.imshow(img_eq)
plt.figure()
# show_hist(img_eq)
plt.imshow(img_clahe)
plt.figure()
# show_hist(img_clahe)
norm_img=img_normalize(img)
plt.imshow(norm_img)
plt.figure()
# show_hist(norm_img)
img_eq_lab,contra_img=img_clahe_luminus(img)
plt.imshow(contra_img)
plt.figure()
plt.imshow(img_eq_lab)
plt.figure()
# show_hist(contra_img)
# %%
