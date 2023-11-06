import cv2
import os
import os.path as osp
import numpy as np

def imshow_lanes(img, lanes, x_scale, y_scale, show=False, out_file=None):
    img = cv2.resize(img, (int(x_scale * img.shape[1]), int(y_scale * img.shape[0])))
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x = np.round(x * x_scale)
            y = np.round(y * y_scale)
            x, y = int(x), int(y)

            cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(1)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)

