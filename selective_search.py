# import the necessary packages
import argparse
import random
import time
import cv2


def selective_search_fast(path):

# load the input image
	image = cv2.imread(path)
	# initialize OpenCV's selective search implementation and set the
	# input image
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)

	ss.switchToSelectiveSearchFast()

	
	# run selective search on the input image
	start = time.time()
	rects = ss.process()
	end = time.time()
	return rects

def selective_search_quality(path):

# load the input image
	image = cv2.imread(path)
	# initialize OpenCV's selective search implementation and set the
	# input image
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)

	ss.switchToSelectiveSearchQuality()

	
	# run selective search on the input image
	start = time.time()
	rects = ss.process()
	end = time.time()
	return rects


selective_search_fast("orange.jpg")