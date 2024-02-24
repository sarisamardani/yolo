import torch
# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True): 
    
    if is_pred: 
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format
        
        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1) 
        y1 = torch.max(b1_y1, b2_y1) 
        x2 = torch.min(b1_x2, b2_x2) 
        y2 = torch.min(b1_y2, b2_y2) 
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
        union = box1_area + box2_area - intersection 

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon) 

        # Return IoU score
        return iou_score 
    
    else: 
        # IoU score based on width and height of bounding boxes
        
        # Calculate intersection area 
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                            torch.min(box1[..., 1], box2[..., 1]) 

        # Calculate union area 
        box1_area = box1[..., 0] * box1[..., 1] 
        box2_area = box2[..., 0] * box2[..., 1] 
        union_area = box1_area + box2_area - intersection_area 

        # Calculate IoU score 
        iou_score = intersection_area / union_area 

        # Return IoU score 
        return iou_score
