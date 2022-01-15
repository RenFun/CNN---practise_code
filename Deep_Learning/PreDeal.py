# -*- coding: utf-8 -*-
import os
import cv2
import csv

video_src_path = "E:/Codeself/KTH/KTHVideo/" # 视频的路径
frame_des_path = "E:/Codeself/KTH/KTHFrame/" # 视频帧保存的路径

# 建立一个CSV文件用于保存样本的路径
train_save_path = "E:/Codeself/KTH/DataLable.csv"

def VideoToFrame(videopath,framepath,interval,count,filelabel):
    
    capture = cv2.VideoCapture(videopath)
    # 输出视频的信息
    # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH )
    # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = capture.get(cv2.CAP_PROP_FPS)
    # num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # count = 0
    # print (width,height,fps,num_frames)  #160.0 120.0 25.0 656.0
    
    success, frame = capture.read()
    count += 1  
    while success:
        if count % interval == 0:
            # print("Writing the number %d of frame to src file" % (count//interval))
            
            GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转换图片色彩空间，比如将彩色图片转换为灰度图片cv2.COLOR_BGR2GRAY
            train_file.writerow([framepath+'%d.jpg'%count,filelabel])
            
            cv2.imwrite(framepath+'%d.jpg' % (count//interval), GRAY) # 取整除 count//interval
        success, frame = capture.read()
        count += 1
        
    capture.release()
    
    print("Encoding file %s success!" % video)
    return count

files = os.listdir(video_src_path)  # 获取视频文件夹下的文件

file_train=open(train_save_path,'a',newline='')  # 避免出现空行
train_file=csv.writer(file_train,dialect='excel')

# csv_file.writerow(['Mes','Label'])

for file in files:  #遍历1、2两个文件
    file_to_video = os.path.join(video_src_path, file)  # 组合成一个文件路径 I:/DataDeal/Video/1

    videos = os.listdir(file_to_video)  #获取视频的列表并分别遍历
    if not os.path.isdir(frame_des_path+file):#如果没有则创建文件夹
        os.mkdir(frame_des_path+file)
    
    frame_save_path = frame_des_path+file+'/' # 进入该目录文件夹下用于保存文件
    index = 0
    for video in videos:
        video_cur_path = os.path.join(file_to_video, video) # 每一个视频的地址
        
        count = VideoToFrame(video_cur_path,frame_save_path,1,index,file)
        index = count

file_train.close()