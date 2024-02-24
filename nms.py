import torch 
from iou import iou
# Non-maximum suppression function to remove overlapping bounding boxes 
def nms(bboxes, iou_threshold, threshold): 
	# Filter out bounding boxes with confidence below the threshold. 
	bboxes = [box for box in bboxes if box[1] > threshold] 

	# Sort the bounding boxes by confidence in descending order. 
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 

	# Initialize the list of bounding boxes after non-maximum suppression. 
	bboxes_nms = [] 

	while bboxes: 
		# Get the first bounding box. 
		first_box = bboxes.pop(0) 

		# Iterate over the remaining bounding boxes. 
		for box in bboxes: 
		# If the bounding boxes do not overlap or if the first bounding box has 
		# a higher confidence, then add the second bounding box to the list of 
		# bounding boxes after non-maximum suppression. 
			if box[0] != first_box[0] or iou( 
				torch.tensor(first_box[2:]), 
				torch.tensor(box[2:]), 
			) < iou_threshold: 
				# Check if box is not in bboxes_nms 
				if box not in bboxes_nms: 
					# Add box to bboxes_nms 
					bboxes_nms.append(box) 

	# Return bounding boxes after non-maximum suppression. 
	return bboxes_nms
