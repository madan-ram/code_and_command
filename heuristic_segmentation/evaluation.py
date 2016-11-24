import os, sys
import numpy as np

def overlap_bin():
	""
	
def overlap_area(A, B, scaled=True):
	XA1, YA1, XA2, YA2 = A
	XB1, YB1, XB2, YB2 = B
	SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
	SA = (max(XA1, XA2) - min(XA1, XA2)) * (max(YA1, YA2) - min(YA1, YA2))
	SB = (max(XB1, XB2) - min(XB1, XB2)) * (max(YB1, YB2) - min(YB1, YB2))
	S = SA+SB-SI
	if scaled:
		return SI/S
	return SI

def accuracy():
	""

def delineation():
	""