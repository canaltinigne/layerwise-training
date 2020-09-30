from helpers import distance
import torch
import torch.nn.functional as F

# Triplet Ratio Loss
def expLoss(output, target, m, centers, class_num, c_points=None, center_distance=None):
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item() 
        
        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum()
        min_D_xj = distance(out, center_distance[classId]).sum()
        
        loss = loss + torch.exp(-min_D_xj / (D_xc+1e-8))
        
    return loss/target.size()[0]


# Triplet Center Loss
def tripletCenterLoss(output, target, m, centers, class_num, c_points=None, center_distance=None):
    loss = 0.
    
    for i, out in enumerate(output):
        classId = target[i].item()

        if c_points is not None:
            if classId not in c_points:
                c_points[classId] = out.unsqueeze(0).data
            else:
                c_points[classId] = torch.cat((c_points[classId], out.unsqueeze(0).data), dim=0)
        
        D_xc = distance(out, centers[classId]).sum().sqrt()
        min_D_xj = distance(out, center_distance[classId]).sum().sqrt()
                
        loss = loss + F.relu(D_xc - min_D_xj + m)
    
    return loss/target.size()[0]