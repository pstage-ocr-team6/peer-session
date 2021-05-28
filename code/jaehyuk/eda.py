#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv  
from collections import Counter
from pprint import pprint

from scipy.signal.signaltools import lfilter
#%%
img_dir='/opt/ml/input/data/train_dataset/images'
# img_dir='images/brightness'
images=os.listdir(img_dir)
# %%
r_image=cv2.imread(os.path.join(img_dir,images[100]))
# %%
plt.imshow(r_image)
#%%
gt_df=pd.read_csv('/opt/ml/input/data/train_dataset/gt.txt',engine='python',sep='.jpg\t',header=None)
print(len(gt_df))
level_df=pd.read_csv('/opt/ml/input/data/train_dataset/level.txt',engine='python',sep='.jpg\t',header=None)
print(len(level_df))
source_df=pd.read_csv('/opt/ml/input/data/train_dataset/source.txt',engine='python',sep='.jpg\t',header=None)
print(len(source_df))
tokens_df=pd.read_csv('/opt/ml/input/data/train_dataset/tokens.txt',sep='\n',header=None, engine='python', quoting=csv.QUOTE_NONE, encoding='utf-8')
print(len(tokens_df))
#%%
def get_unique_count(df):
    df=df.to_numpy()
    get_str=' '.join(df)
    get_str.split()
    unique_dict=Counter(get_str.split())
    unique_dict=dict(sorted(unique_dict.items(), key= lambda item : item[1]))
    return unique_dict
# %%
gt_np=gt_df[1].to_numpy()
gt_str=' '.join(gt_np)
gt_str.split()
unique_dict=Counter(gt_str.split())
unique_dict=dict(sorted(unique_dict.items(), key= lambda item : item[1]))
pprint(unique_dict)
#%%
# pprint(get_unique_count(gt_df[1]))
# for k,v in get_unique_count(gt_df[1]).items():
#     print(k ,':',v)
# %%
level_df.groupby(1).nunique()
#%%
merge_outer=pd.merge(pd.merge(gt_df,level_df,on=0),source_df,on=0)
# %%
merge_outer.columns =['name','gt','level','source']
# %%
pprint(get_unique_count(merge_outer['gt']))
# %%
merge_level=merge_outer.groupby('level').nunique()
merge_level.head()
# %%
# merge_outer[merge_outer['level']==1]['gt'].nunique()
# pprint(get_unique_count(merge_outer[merge_outer['level']==1]['gt']))
lv1_key=set(get_unique_count(merge_outer[merge_outer['level']==1]['gt']).keys())
lv2_key=set(get_unique_count(merge_outer[merge_outer['level']==2]['gt']).keys())
lv3_key=set(get_unique_count(merge_outer[merge_outer['level']==3]['gt']).keys())
lv4_key=set(get_unique_count(merge_outer[merge_outer['level']==4]['gt']).keys())
lv5_key=set(get_unique_count(merge_outer[merge_outer['level']==5]['gt']).keys())
# %%
key_list=[lv1_key,lv2_key,lv3_key,lv4_key,lv5_key]
intersection_ket=set.intersection(*key_list)
pprint(intersection_ket)
# pprint(lv4_key.difference(lv5_key))
# %%
merge_level
# %%
merge_outer[merge_outer['level']==1]['gt']
# %%
img=cv2.imread(os.path.join(img_dir,'train_00016.jpg'))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=500,lines=np.array([]), minLineLength=100,maxLineGap=80)
a,b,c = lines.shape
plt.imshow(gray)
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    # cv2.imwrite('houghlines5.jpg',gray)

    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
# ret, dst = cv2.threshold(r_image, 170, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(gray)
# plt.imshow(img)
#%%
len(lines)
# %%
# image = cv2.imread('3.jpg')
image=cv2.imread(os.path.join(img_dir,'train_00016.jpg'))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
result = cv2.bitwise_and(image,image,mask=thresh)
result[thresh==0] = (255,255,255)

plt.imshow(thresh)
plt.figure()
plt.imshow(result)
# %%
img_cp=img.copy()
for line in lines:
    a,b,c,d=line[0]
    cv2.rectangle(img_cp,(a,b),(c,d))
plt.imshow(img_cp)
# %%
def img_Contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # luminosity(명도)채널을 얻기 위해 채널을 BGR->LAB 로 바꿈
    l, a, b = cv2.split(lab) # 채널 분리
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # 히스토그램 균등화 시키기 http://www.gisdeveloper.co.kr/?p=6652
    cl = clahe.apply(l) # 명도 채널에 적용
    limg = cv2.merge((cl, a, b)) # 채널 합치기
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) 
    return final 
img=cv2.imread(os.path.join(img_dir,'train_00016.jpg'))
plt.title('original') 
plt.imshow(img)
plt.figure()
img = img_Contrast(img) 
plt.title('img_Contrast') 
plt.imshow(img) 
#%%
# img_dir='/opt/ml/input/data/train_dataset/images'
image = cv2.imread(os.path.join(img_dir,'train_83399.jpg'))  # 이미지 로드
h,w=image.shape[:2] # h, w
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # gray 채널
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #OTSU 방법을 이용해 binary 이미지 변환
thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15,2)
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
wrinkle_img=cv2.imread(os.path.join(img_dir,'train_00007.jpg'))
w,h=wrinkle_img.shape[:2]
a = [[0,0],[w,0],[0,h],[w,h]] 
b = [[0, 0],[1000, 0],[0,500],[1000,500]]
pts1 = np.float32(a)
M = cv2.getPerspectiveTransform(pts1,pts1)
dst = cv2.warpPerspective(img, M, (w, h))
plt.imshow(dst)
# %%
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
img=cv2.imread(os.path.join(img_dir,'train_83399.jpg'))
# hist,bins = np.histogram(img.flatten(),256,[0,256])
hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.imshow( img)
plt.figure()
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img = cdf[img]
plt.imshow( img)
plt.figure()
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()
# %%
img=cv2.imread(os.path.join(img_dir,'train_83399.jpg'))
plt.imshow(thresh)
plt.figure()
img = img_Contrast(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) ) 
plt.imshow(img) 
# %%
def img_equal_clahe_yuv(img):
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
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm
#%%
def show_hist(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()
    plt.figure()
#%%
def img_clahe_luminus(img):
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
img=cv2.imread(os.path.join(img_dir,'train_69864.jpg'))
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
plt.imshow(norm_img,cmap='gray')
plt.figure()
# show_hist(norm_img)
img_eq_lu,img_clahe_lu=img_clahe_luminus(img)
plt.imshow(img_clahe_lu)
plt.figure()
plt.imshow(img_eq_lu)
plt.figure()
# show_hist(contra_img)
# %%
img=cv2.imread(os.path.join(img_dir,'train_69864.jpg'))
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh=cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 15,2)
# thresh=cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY, 15,2)
# thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
plt.imshow(thresh,cmap='gray')

# %%
img=cv2.imread(os.path.join(img_dir,'train_69864.jpg'))
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(img_hsv)
plt.imshow(h)
plt.figure()
plt.imshow(s)
plt.figure()
plt.imshow(v)
# %%
img=cv2.imread(os.path.join(img_dir,'train_69864.jpg'))
img_lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
h,s,v=cv2.split(img_lab)
plt.imshow(h)
plt.figure()
plt.imshow(s)
plt.figure()
plt.imshow(v)
# %%
img=cv2.imread(os.path.join(img_dir,'train_69864.jpg'))
img_yuv=cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v=cv2.split(img_yuv)
plt.imshow(y)
plt.figure()
plt.imshow(u)
plt.figure()
plt.imshow(v)

img_eq=img_yuv.copy()
ey=cv2.equalizeHist(y)
img_eq = cv2.cvtColor(cv2.merge((ey, u, v)), cv2.COLOR_YUV2BGR)
plt.figure()
plt.imshow(img_eq)
# img_clahe = img_yuv.copy()
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
# img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
# img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)
# %%
edge =cv2.Canny(y,0,65)
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
plt.imshow(opening,cmap='gray')
# %%
thresh=cv2.adaptiveThreshold(y, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 15,2)
# thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
# plt.imshow(thresh,cmap='gray')
img_bwa = cv2.bitwise_and(thresh,opening)
plt.imshow(img_bwa,cmap='gray')
# %%
kernel = np.ones((2, 2), np.uint8)
erode_image = cv2.dilate(thresh, kernel, iterations=1)
erode_image = cv2.erode(255-erode_image, kernel, iterations=1)
# tophat = cv2.morphologyEx(erode_image, cv2.MORPH_BLACKHAT, kernel)
plt.imshow(cv2.bitwise_xor(erode_image,y),cmap='gray')
# %%
plt.imshow(255-y,cmap='gray')
plt.figure()
plt.imshow(y,cmap='gray')
# %%
thresh=cv2.adaptiveThreshold(255-y, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 15,2)
thresh = cv2.threshold(255-y, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
plt.imshow(thresh[thresh>100],cmap='gray')
# %%
plt.imshow(cv2.inRange(255-y, (50), (70)),cmap='gray')
# %%
plt.imshow(cv2.inRange(255-y, (50), (70))+cv2.inRange(y, (45), (70)),cmap='gray')
# %%
img=cv2.imread(os.path.join(img_dir,'train_03760.jpg'))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 1)
# thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
thresh=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 15,2)
# dst_TELEA = cv2.inpaint(img,thresh,3,cv2.INPAINT_TELEA)
kernel = np.ones((3, 3), np.uint8)
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(255-thresh, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(255-opening, cv2.MORPH_CLOSE, kernel)
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
result = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
# erode_image = cv2.erode(opening, kernel, iterations=1)
dilate_image = cv2.dilate(result, kernel, iterations=2)
plt.imshow(img)
plt.figure()
plt.imshow(thresh,cmap='gray')
plt.figure()
plt.imshow(opening,cmap='gray')
plt.figure()
plt.imshow(255-result,cmap='gray')
plt.figure()
plt.imshow(255-dilate_image,cmap='gray')
#%%
def remove_line(gray):
    h,w=img.shape[:2] # h, w
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # gray 채널
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #OTSU 방법을 이용해 binary 이미지 변환

    kernel = np.ones((3, 3), np.uint8)
    dilation_image = cv2.dilate(thresh, kernel, iterations=1) # dilate 연산 

    # 가록 선을 찾을 커널 정하기 -> 가로 선을 찾을 꺼나 (x,y)에서 x를 더 크게 잡아야된다.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1)) 
    # 모포로지 연산 (위의 dilate와 유사한 시리즈)
    detected_lines = cv2.morphologyEx(dilation_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # 선들의 contour 들을 찾기
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    temp_height=10
    for c in cnts: # 모든 contour 마다
        temp=c.flatten()
        if max(temp[::2])-min(temp[::2]) >w/2: # contour의 w가 전체 이미지의 w/2보다 크면 삭제
            temp_height=max(temp[1::2])-min(temp[1::2])
            cv2.drawContours(dilation_image, [c], -1, (0,0,0), -1) # 안을 검은색으로 채워줌
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,temp_height//10))
    result = cv2.morphologyEx(dilation_image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
    return 255-result
# %%
def remove_brightness(gray):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w=img.shape[:2]
    if h*w>1000*100:
        gray=cv2.resize(gray, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)
    blurred = cv2.GaussianBlur(gray, (11,11), 1)
    # thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
    thresh=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15,2)
    # dst_TELEA = cv2.inpaint(img,thresh,3,cv2.INPAINT_TELEA)
    kernel = np.ones((3, 3), np.uint8)
    kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(255-thresh, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(255-opening, cv2.MORPH_CLOSE, kernel)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    result = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    # erode_image = cv2.erode(opening, kernel, iterations=1)
    kernel = np.ones((1, 5), np.uint8)
    dilate_image = cv2.dilate(result, kernel, iterations=1)
    return 255-result
# %%
image_list=['03760','14180','23372','25827','31345','47703',
'69864','71023','83399','90301','25827', '23372', '03760', '45307', '66036', '57674', '40377',
'83399', '14180', '71023', '90301', '69864',
'23372', '23166', '73781', '17609']
for image in set(image_list):
    img=cv2.imread(os.path.join(img_dir,'train_'+image+'.jpg'))
    plt.title(image)
    rm_bright=remove_brightness(img)
    plt.imshow(rm_bright,cmap='gray')
    plt.figure()
    plt.title(image)
    plt.imshow(255-remove_line(rm_bright),cmap='gray')
    plt.figure()
# %%
img=cv2.imread(os.path.join(img_dir,'train_'+str(14180)+'.jpg'))
plt.imshow(img)
# %%
image_list=['83399']
for image in set(image_list):
    img=cv2.imread(os.path.join(img_dir,'train_'+image+'.jpg'))
    plt.title(image)
    rm_bright=remove_brightness(img)
    plt.imshow(rm_bright,cmap='gray')
    plt.figure()
    plt.imshow(255-remove_line(rm_bright),cmap='gray')
    plt.figure()
# %%
# from scipy.interpolate import splrep, splev
img=cv2.imread(os.path.join(img_dir,'train_'+'00525'+'.jpg'))
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,w=gray.shape[:2]
temp=[]
white_img=255+np.zeros_like(gray)
# gray-=255
for i in range(w):
    temp.append(gray[:,i].mean())
plt.title('x')
plt.plot(temp)
plt.figure()
temp=[]
for i in range(h):
    temp.append(gray[i,:].mean())
# print(h,w)
# print(len(temp))
# grad=np.gradient(temp)
plt.title('y')
plt.plot(temp)
plt.figure()
plt.imshow(img)
# f = splrep(gray[0,:],grad,k=5,s=3)
# %%
# %%
img=cv2.imread(os.path.join(img_dir,'train_'+'65867'+'.jpg'))
plt.imshow(img) 
# gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# h,w=gray.shape[:2]

# sliding_size=np.zeros((h//3,w//3))
# %%
import imutils
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
# %%
# image=cv2.imread(os.path.join(img_dir,'train_'+'00525'+'.jpg'))
save_folder='/opt/ml/c_transformed'
image_list=['03760','14180','23372','25827','31345','47703',
'69864','71023','83399','90301','25827', '23372', '03760', '45307', '66036', '57674', '40377',
'83399', '14180', '71023', '90301', '69864',
'23372', '23166', '73781', '17609',
'67774', '05513', '12169', 
'65511', '67496', '67496', '08518',
'52073', 
71594,
58664, 79361, 33776,
73463, 80954,
71152,
88703,
89744, 38256, 66300, 88703, 38538, 76487, 87216, 41498, 78022, 62535, 29298,42042,77860, 81037,
64703,
21807,89744,90858,99892,88991,
42562, 22767, 44015, '02801', 98707, 63021, 68263, 87109, '06959', 85476,
65867 
]
for image in set(image_list):
    img=cv2.imread(os.path.join(img_dir,'train_'+str(image)+'.jpg'))
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w=gray.shape[:2]
    (winW, winH) = (w//3, h//3)
    _max=0
    _min=np.inf
    orig_mean=gray.mean()
    for resized in pyramid(gray, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        # print(resized)
        for (x, y, window) in sliding_window(resized, stepSize=max(w//10,h//10), windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            _mean=resized[y:y+winH,x:x+winW].mean()
            if _max <_mean:
                _max=_mean
            if _min > _mean:
                _min=_mean
            # print(_mean)
            # clone = resized.copy()
            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # plt.title(resized[y:y+winH,x:x+winW].mean())
            # plt.imshow(clone)
            # plt.figure()
    
    if orig_mean < 127:
        show=remove_brightness(gray)
    else:
        if (_max-orig_mean) > 40 or (_min-orig_mean)<-40 :
            show=remove_brightness(gray)
        else:
            show=remove_line(gray)
    # plt.title(((_max-orig_mean),(_min-orig_mean),int(orig_mean)))
    # print(os.path.join(save_folder,str(image)+'.jpg'))
    # break
    cv2.imwrite(os.path.join(save_folder,str(image)+'.jpg'),img)
    cv2.imwrite(os.path.join(save_folder,str(image)+'_bi.jpg'),show)
    # plt.title(str(image))
    # plt.imshow(show,cmap='gray')
    # plt.figure()
# %%
cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
# %%
low_resol_image=[
    '89744', '38256', '66300', '88703', '38538', '76487', 87216, 41498, 78022, 62535, 29298,42042,77860, 81037
]
path = "/opt/edsr_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(path)
for image in low_resol_image:
    img=cv2.imread(os.path.join(img_dir,'train_'+str(image)+'.jpg'))
    result = sr.upsample(img)

    # plt.title(img.shape[:2])
    plt.imshow(img)
    plt.figure()
    plt.imshow(result)
    plt.figure()
# %%
