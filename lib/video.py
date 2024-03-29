#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import os

class VideoReader:

    def __init__(self):
        self.vid = None
        self.out = None
        self.running = False
        self.detection_counter = {}
        return

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()

        return

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return

    def start(self, src, width, height, output_image_dir='output_image', output_movie_dir='output_movie', output_prefix='output', save_to_file=False):
        """
        output_1532580366.27.avi
        output_file[:-4] # remove .avi from filename
        """
        output_file = output_movie_dir + '/' + output_prefix + '_' + str(time.time()) + '.avi'
        self.OUTPUT_MOVIE_DIR = output_movie_dir
        self.OUTPUT_IMAGE_DIR = output_image_dir

        # initialize the video camera vid and read the first frame
        self.vid = cv2.VideoCapture(src)
        if not self.vid.isOpened():
            # camera failed
            raise IOError(("Couldn't open video file or webcam."))
        #self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        #self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.ret, self.frame = self.vid.read()
        if not self.ret:
            self.vid.release()
            raise IOError(("Couldn't open video frame."))

        # initialize the variable used to indicate if the thread should
        # check camera vid shape
        self.real_width = int(self.vid.get(3))
        self.real_height = int(self.vid.get(4))
        print("Start video stream with shape: {},{}".format(self.real_width, self.real_height))
        self.running = True

        """ save to file """
        if save_to_file:
            self.mkdir(output_movie_dir)
            fps = self.vid.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter(output_file, int(fourcc), fps, (int(self.real_width), int(self.real_height)))
        return self

    def getSize(self):
        return (self.real_width, self.real_height)

    def read(self):
        # return the frame most recently read
        self.ret, self.frame = self.vid.read()
        #self.frame = cv2.resize(frame, (600, 600), interpolation = cv2.INTER_AREA)
        if not self.ret:
            self.stop()
            return None
        return self.frame

    def save(self, frame):
        # save to avi
        self.out.write(frame)
        return

    def stop(self):
        self.running = False
        if self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()

    def save_detection_image(self, int_label, cv_bgr, filepath):
        self.mkdir(self.OUTPUT_IMAGE_DIR+"/"+str(int_label))

        dir_path, filename = os.path.split(filepath)
        if not filename in self.detection_counter:
            self.detection_counter.update({filename: 0})
        self.detection_counter[filename] += 1
        # remove .jpg/.jpeg/.png and get filename
        if filename.endswith(".jpeg"):
            filehead = filename[:-5]
            filetype = ".jpeg"
        elif filename.endswith(".jpg"):
            filehead = filename[:-4]
            filetype = ".jpg"
        elif filename.endswith(".png"):
            filehead = filename[:-4]
            filetype = ".png"

        # save to file
        cv2.imwrite(self.OUTPUT_IMAGE_DIR+"/"+str(int_label)+"/"+filehead+"_"+str(self.detection_counter[filename])+filetype, cv_bgr)
        return
