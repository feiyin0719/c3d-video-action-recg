# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     split_video
   Description :
   Author :       iffly
   date：          5/4/18
-------------------------------------------------
   Change Activity:
                   5/4/18:
-------------------------------------------------
"""
import argparse
import cv2
import os
import sys
import time

parser = argparse.ArgumentParser(usage="python split_video.py --video videopath --output outputpath",
                                 description="help info.")
parser.add_argument("--video", default="", help="the video path.", dest="video_path", required=True)
parser.add_argument("--output", default="", help="the output path.", dest="output_path", required=True)
args = parser.parse_args()
video_path = args.video_path
output_path = args.output_path


def split_video(video_path, class_name):
    _, file_name = os.path.split(video_path)
    video_name, _ = os.path.splitext(file_name)
    if not os.path.exists(os.path.join(output_path, 'frame', class_name, video_name)):
        os.makedirs(os.path.join(output_path, 'frame', class_name, video_name))
    if not os.path.exists(os.path.join(output_path, 'flow', class_name, video_name)):
        os.makedirs(os.path.join(output_path, 'flow', class_name, video_name))
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened():
        now_frame = 0
        old_frame = None
        while True:
            success, frame = capture.read()
            if not success:
                break
            cv2.imwrite(os.path.join(output_path, 'frame', class_name, video_name, "{}.jpg".format(now_frame)), frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calc optical
            # if  old_frame is None:
            #     old_frame=frame
            # optical_flow=cv2.calcOpticalFlowFarneback(old_frame, frame,None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # cv2.imwrite(os.path.join(output_path,'flow',class_name,video_name,"{}_h.png".format(now_frame)),optical_flow[:,:,0])
            # cv2.imwrite(os.path.join(output_path,'flow',class_name,video_name,"{}_v.png".format(now_frame)),optical_flow[:,:,1])
            now_frame += 1
        capture.release()
    else:
        print("cannot open " + video_name)


def process_data(video_path):
    class_list = os.listdir(video_path)
    print("class:", class_list)
    print("process start:")
    now_count = 0
    for class_name in class_list:
        if os.path.isdir(os.path.join(video_path, class_name)):
            video_list = os.listdir(os.path.join(video_path, class_name))
            for video_name in video_list:
                split_video(os.path.join(video_path, class_name, video_name), class_name)
                sys.stdout.write("{} process done.".format(now_count) + '\r')
                sys.stdout.flush()
                now_count += 1
    print("process end.")


if __name__ == '__main__':
    process_data(video_path)

    # print("123")
    # for i in range(0,10):
    #     sys.stdout.write("{} process done.".format(i)+'\r')
    #     sys.stdout.flush()
    #     time.sleep(2)
