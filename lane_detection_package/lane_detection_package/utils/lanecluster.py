# -*- coding: utf-8 -*-

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
#import glog as log

import matplotlib.pyplot as plt

COLOR_MAP = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

def embedding_feats_dbscan_cluster(lane_coord):
    """
    dbscan cluster
    :param embedding_image_feats:
    :return:
    """
    db = DBSCAN(eps=0.35, min_samples=50)
    try:
        features = StandardScaler().fit_transform(lane_coord)
        db.fit(features)
    except Exception as err:
        #log.error(err)
        ret = {
            'origin_features': None,
            'cluster_nums': 0,
            'db_labels': None,
            'unique_labels': None,
            'cluster_center': None
        }
        return ret
    db_labels = db.labels_
    unique_labels = np.unique(db_labels)

    num_clusters = len(unique_labels)
    cluster_centers = db.components_

    ret = {
        'origin_features': features,
        'cluster_nums': num_clusters,
        'db_labels': db_labels,
        'unique_labels': unique_labels,
        'cluster_center': cluster_centers
    }

    return ret

def get_lane_embedding_feats(binary_seg_result):
    """
    get lane embedding features according the binary seg result
    :param binary_seg_ret:
    :param instance_seg_ret:
    :return:
    """
    idx = np.where(binary_seg_result == 255)
    lane_coord=np.column_stack((idx[0], idx[1]))
    
    return lane_coord

def lane_mask_coords(binaryimage):
    # get embedding feats and coords
    get_lane_embedding_feats_result = get_lane_embedding_feats(binary_seg_result=binaryimage)
    
    # dbscan cluster
    dbscan_cluster_result = embedding_feats_dbscan_cluster(
        lane_coord=get_lane_embedding_feats_result
    )
    mask = np.zeros(shape=[binaryimage.shape[0], binaryimage.shape[1], 3], dtype=np.uint8)
    db_labels = dbscan_cluster_result['db_labels']
    unique_labels = dbscan_cluster_result['unique_labels']
    coord = get_lane_embedding_feats_result
    
    if db_labels is None:
        return None, None
    
    lane_coords = []
    
    for index, label in enumerate(unique_labels.tolist()):
        if label == -1:
            continue
        idx = np.where(db_labels == label)
        # pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
        pix_coord_idx = tuple((coord[idx][:, 0], coord[idx][:, 1]))
        mask[pix_coord_idx] = COLOR_MAP[index]
        lane_coords.append(coord[idx])
    return mask, lane_coords

# full_img=img_pil
# scale_factor=1
# out_threshold=0.2
# img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
# img = img.unsqueeze(0)
# img = img.to(device=device, dtype=torch.float32)

# with torch.no_grad():
#     output = net(img)
# if net.n_classes > 1:
#     probs = F.softmax(output, dim=1)
# else:
#     probs = torch.sigmoid(output)
# probs = probs.squeeze(0)

# np_arr = probs.cpu().detach().numpy()
# np_arr_reshape = np.reshape(np_arr,[256,512])
# np_arr_reshape[np_arr_reshape<0.2]=0
# np_arr_reshape[np_arr_reshape>=0.2]=1
# binaryimage = (np_arr_reshape*255).astype(np.uint8)
