# import the necessary packages
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
	rects = ss.process()
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
	rects = ss.process()
	return rects