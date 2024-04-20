import os
import json

import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import time as time
import itertools
import math
import random
import seaborn as sns

action_dataset="hake"
generation_model="stablediffusion2"
rotate_list = [8, 7, 14, 13, 6, 5, 12, 11]
H, W = 768, 620
aspect_ratio = 0.75
pixel_std = 200
num_joints = 17

def kp_trans(box, dst_shape=(384, 512), size="small"):
    scale = None
    rotation = None

    src_xmin, src_ymin, src_xmax, src_ymax = box[:4]
    src_w = src_xmax - src_xmin
    src_h = src_ymax - src_ymin

    if src_h / dst_shape[0] > src_w / dst_shape[1]:
        if size == "small":
            h_N = 0.25
        elif size == "middle":
            h_N = 0.5
        elif size == "large":
            h_N = 0.8
        w_N = dst_shape[0] * h_N / src_h
        fixed_size = (dst_shape[0] * h_N, src_w * w_N)
    else:
        if size == "small":
            w_N = 0.25
        elif size == "middle":
            w_N = 0.5
        elif size == "large":
            w_N = 0.8
        h_N = dst_shape[1] * w_N / src_w
        fixed_size = (src_h * h_N, dst_shape[1] * w_N)
    src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
    src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
    src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    # dst_center = np.array([(fixed_size[1] + 1) / 2, (fixed_size[0] + 1) / 2])
    # dst_p2 = dst_center + np.array([(fixed_size[1]) / 2, 0])  # top middle
    # dst_p3 = np.array([fixed_size[1], (fixed_size[0]) / 2])  # right middle
    dst_center = np.array([dst_shape[1] / 2, dst_shape[0] / 2])
    dst_p2 = dst_center + np.array([0, -fixed_size[0] / 2])  # top middle
    dst_p3 = dst_center + np.array([fixed_size[1] / 2, 0])  # right middle

    if scale is not None:
        scale = random.uniform(*scale)
        src_w = src_w * scale
        src_h = src_h * scale
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    if rotation is not None:
        angle = random.randint(*rotation)  # 角度制
        angle = angle / 180 * math.pi  # 弧度制
        src_p2 = src_center + np.array(
            [src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)]
        )
        src_p3 = src_center + np.array(
            [src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)]
        )

    src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
    dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

    trans = cv2.getAffineTransform(src, dst)
    return trans


def kp_trans_dst(box, dst_shape=(384, 512), dst_center=None, rotation=None):
    scale = None

    src_xmin, src_ymin, src_xmax, src_ymax = box[:4]
    src_w = src_xmax - src_xmin
    src_h = src_ymax - src_ymin
    fixed_size = (src_h, src_w)
    src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
    src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
    src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    # dst_center = np.array([(fixed_size[1] + 1) / 2, (fixed_size[0] + 1) / 2])
    # dst_p2 = dst_center + np.array([(fixed_size[1]) / 2, 0])  # top middle
    # dst_p3 = np.array([fixed_size[1], (fixed_size[0]) / 2])  # right middle
    # dst_center = np.array([dst_shape[1] / 2, dst_shape[0] / 2])
    dst_p2 = dst_center + np.array([0, -fixed_size[0] / 2])  # top middle
    dst_p3 = dst_center + np.array([fixed_size[1] / 2, 0])  # right middle

    if scale is not None:
        scale = random.uniform(*scale)
        src_w = src_w * scale
        src_h = src_h * scale
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    if rotation is not None:
        angle = random.randint(*rotation)  # 角度制
        angle = angle / 180 * math.pi  # 弧度制
        src_p2 = src_center + np.array(
            [src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)]
        )
        src_p3 = src_center + np.array(
            [src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)]
        )

    src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
    dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

    trans = cv2.getAffineTransform(src, dst)
    return trans


def affine_points(pt, t):
    npt = pt[:, :2]
    ones = np.ones((npt.shape[0], 1), dtype=float)
    npt = np.concatenate([npt, ones], axis=1).T
    new_pt = np.dot(t, npt)
    return new_pt.T


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print("loading annotations into memory...")
            tic = time.time()
            dataset = json.load(open(annotation_file, "r"))
            assert (
                type(dataset) == dict
            ), "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]

        else:
            cats = self.dataset["categories"]
            cats = (
                cats
                if len(catNms) == 0
                else [cat for cat in cats if cat["name"] in catNms]
            )
            cats = (
                cats
                if len(supNms) == 0
                else [cat for cat in cats if cat["supercategory"] in supNms]
            )
            cats = (
                cats
                if len(catIds) == 0
                else [cat for cat in cats if cat["id"] in catIds]
            )

        ids = [cat["id"] for cat in cats]
        return ids

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            # 根据imgIds找到所有的ann
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            # 通过各类条件如catIds对anns进行筛选
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )
        if not iscrowd == None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

def show_mask_skelenton(img, kpts, thr=0.01):
    stickwidth = 2
    kpts = np.array(kpts).reshape(-1, 3)
    kp6 = kpts[5]
    kp7 = kpts[6]
    if kpts[5][2] != 0 and kpts[6][2] != 0:
        kp18 = [
            (kp6[0] + kp7[0]) / 2,
            (kp6[1] + kp7[1]) / 2,
            1,
        ]  # 找两个肩膀的中间点，如果其中一个肩膀的坐标为0，则将中间点坐标赋值为存在的点坐标
    else:
        kp18 = [(kp6[0] + kp7[0]), (kp6[1] + kp7[1]), 0]

    kpts = np.append(kpts, kp18)
    kpts = np.array(kpts).reshape(-1, 3)
    colors = [
        [0, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [0, 255, 255],
        [0, 255, 85],
        [0, 255, 0],
        [0, 255, 170],
        [85, 255, 0],
        [170, 255, 0],
        [255, 255, 0],
        [255, 170, 0],
        [255, 85, 0],
        [255, 0, 0],
        [170, 0, 255],
        [255, 0, 170],
        [255, 0, 255],
        [85, 0, 255],
        [255, 0, 85],
    ]
    skelenton = [
        [18, 7],
        [18, 6],        
        [7, 9],
        [9, 11],
        [6, 8],
        [8, 10],
        [18, 13],
        [15, 13],
        [17, 15],
        [18, 12],
        [14, 12],
        [16, 14],
        [18, 1],
        [1, 3],
        [3, 5],      
        [1, 2],
        [2, 4],
    ]

    draw_sk = [
        [1,3,4], #左胳膊
        [2,5,6], #右胳膊
        [7,8,9],
        [10,11,12],
        [14,15,16,17], #头部
        [1,3,4,7,8,9],
        [2,5,6,10,11,12],
        [14,16,1,2,7],
        [7,8,9,10,11,12],
        [15,16,13,1,2],
    ]
    drawing = random.choice(draw_sk)
    # for n in range(len(kpts)):
    #     x, y = kpts[n][0:2]
    #     cv2.circle(img, (int(x), int(y)), 3, colors[n], thickness=-1)
    #     # cv2.imwrite("s{}.jpg".format(n), img)
    i = 0
    for index, sk in enumerate(skelenton):
        
        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
        if index+1 in drawing:
            if (
                pos1[0] > 0
                and pos1[1] > 0
                and pos2[0] > 0
                and pos2[1] > 0
                and kpts[sk[0] - 1, 2] > thr
                and kpts[sk[1] - 1, 0] > thr
            ):
                X = [pos1[1], pos2[1]]
                Y = [pos1[0], pos2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                #             ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
                )  # 画连接的椭圆
                cv2.fillConvexPoly(img, polygon, colors[i])  # 填充颜色

                # cv2.imwrite("t{}_gc.jpg".format(i), img)
        i += 1
    return img

def show_mask(img, kpts, thr=0.01):

    kpts = np.array(kpts).reshape(-1, 3)
    kp6 = kpts[5]
    kp7 = kpts[6]
    if kpts[5][2] != 0 and kpts[6][2] != 0:
        kp18 = [
            (kp6[0] + kp7[0]) / 2,
            (kp6[1] + kp7[1]) / 2,
            1,
        ]  # 找两个肩膀的中间点，如果其中一个肩膀的坐标为0，则将中间点坐标赋值为存在的点坐标
    else:
        kp18 = [(kp6[0] + kp7[0]), (kp6[1] + kp7[1]), 0]

    kpts = np.append(kpts, kp18)
    kpts = np.array(kpts).reshape(-1, 3)
    colors = [255, 255, 255]

    skelenton = [
        [18, 7],
        [18, 6],        
        [7, 9],
        [9, 11],
        [6, 8],
        [8, 10],
        [18, 13],
        [15, 13],
        [17, 15],
        [18, 12],
        [14, 12],
        [16, 14],
        [18, 1],
        [1, 3],
        [3, 5],      
        [1, 2],
        [2, 4],
    ]
   
    for n in range(len(kpts)):
        x, y = kpts[n][0:2]
        cv2.circle(img, (int(x), int(y)), 3, colors, thickness=-1)

    i = 0
    for sk in skelenton:

        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))

        if (
            pos1[0] > 0
            and pos1[1] > 0
            and pos2[0] > 0
            and pos2[1] > 0
            and kpts[sk[0] - 1, 2] > thr
            and kpts[sk[1] - 1, 0] > thr
        ):
            cv2.line(img, pos1, pos2, colors, thickness=64)

        i += 1
    return img



def show_skelenton(img, kpts, thr=0.01):
    stickwidth = 4
    kpts = np.array(kpts).reshape(-1, 3)
    kp6 = kpts[5]
    kp7 = kpts[6]
    if kpts[5][2] != 0 and kpts[6][2] != 0:
        kp18 = [
            (kp6[0] + kp7[0]) / 2,
            (kp6[1] + kp7[1]) / 2,
            1,
        ]  # 找两个肩膀的中间点，如果其中一个肩膀的坐标为0，则将中间点坐标赋值为存在的点坐标
    else:
        kp18 = [(kp6[0] + kp7[0]), (kp6[1] + kp7[1]), 0]

    kpts = np.append(kpts, kp18)
    kpts = np.array(kpts).reshape(-1, 3)
    colors = [
        [0, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [0, 255, 255],
        [0, 255, 85],
        [0, 255, 0],
        [0, 255, 170],
        [85, 255, 0],
        [170, 255, 0],
        [255, 255, 0],
        [255, 170, 0],
        [255, 85, 0],
        [255, 0, 0],
        [170, 0, 255],
        [255, 0, 170],
        [255, 0, 255],
        [85, 0, 255],
        [255, 0, 85],
    ]
    skelenton = [
        [18, 7],
        [18, 6],        
        [7, 9],
        [9, 11],
        [6, 8],
        [8, 10],
        [18, 13],
        [15, 13],
        [17, 15],
        [18, 12],
        [14, 12],
        [16, 14],
        [18, 1],
        [1, 3],
        [3, 5],      
        [1, 2],
        [2, 4],
    ]
    # skelenton = [
    #     [16, 14],
    #     [14, 12],
    #     [17, 15],
    #     [15, 13],
    #     [6, 8],
    #     [7, 9],
    #     [8, 10],
    #     [9, 11],
    #     [1, 2],
    #     [1, 3],
    #     [2, 4],
    #     [3, 5],
    #     [18, 1],
    #     [18, 6],
    #     [18, 7],
    #     [18, 12],
    #     [18, 13],
    # ]
    # 18个点
   
    for n in range(len(kpts)):
        x, y = kpts[n][0:2]
        cv2.circle(img, (int(x), int(y)), 3, colors[n], thickness=-1)

    i = 0
    for sk in skelenton:

        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))

        if (
            pos1[0] > 0
            and pos1[1] > 0
            and pos2[0] > 0
            and pos2[1] > 0
            and kpts[sk[0] - 1, 2] > thr
            and kpts[sk[1] - 1, 0] > thr
        ):
            X = [pos1[1], pos2[1]]
            Y = [pos1[0], pos2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            #             ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )  # 画连接的椭圆
            cv2.fillConvexPoly(img, polygon, colors[i])  # 填充颜色
            # print(colors[i])
            # cv2.imwrite("t{}_gc.jpg".format(i), img)
        i += 1
    return img

def draw_hsd_skeleton(image, kpts, thr=0.01):
    humansd_skeleton=[
              [0,0,1],
              [1,0,2],
              [2,1,3],
              [3,2,4],
              [4,3,5],
              [5,4,6],
              [6,5,7],
              [7,6,8],
              [8,7,9],
              [9,8,10],
              [10,5,11],
              [11,6,12],
              [12,11,13],
              [13,12,14],
              [14,13,15],
              [15,14,16],
          ]
    humansd_skeleton_width=2
    humansd_color=sns.color_palette("hls", len(humansd_skeleton)) 
    kpts = np.array(kpts).reshape(-1, 3)
    def plot_kpts(img_draw, kpts, color, edgs,width):     
        for idx, kpta, kptb in edgs:

            line_color = tuple([int(255*color_i) for color_i in color[idx]])
            pos1 = (int(kpts[kpta,0]),int(kpts[kpta,1]))
            pos2 = (int(kpts[kptb,0]),int(kpts[kptb,1]))

            if (
                pos1[0] > 0
                and pos1[1] > 0
                and pos2[0] > 0
                and pos2[1] > 0
                and kpts[kpta , 2] > thr
                and kpts[kptb , 0] > thr
            ):

                cv2.line(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), (int(kpts[kptb,0]),int(kpts[kptb,1])), line_color,width)
                cv2.circle(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), width//2, line_color, -1)
                cv2.circle(img_draw, (int(kpts[kptb,0]),int(kpts[kptb,1])), width//2, line_color, -1)
    plot_kpts(image, kpts,humansd_color,humansd_skeleton,humansd_skeleton_width)
        
    return image


def get_center_keypoints(coco, img_idx, dst_shape=(384, 512, 3), size="small"):
    annIds = coco.getAnnIds(imgIds=img_idx, iscrowd=False)
    objs = coco.loadAnns(annIds)
    for person_id, obj in enumerate(objs):
        if obj["num_keypoints"] < 5:
            continue
        keypoints = obj["keypoints"]
        kpts = np.array(keypoints).reshape(-1, 3)

        mask = np.logical_and(kpts[:, 0] != 0, kpts[:, 1] != 0)

        # 通过坐标点，找框，然后找中心点坐标，从而生成仿射矩阵，进行坐标点的变化
        konghang = []
        for i in range(len(kpts)):
            if kpts[i][2] == 0:
                konghang.append(i)
        kpt_new = np.delete(kpts, konghang, axis=0)

        MAX = np.max(kpt_new, axis=0)
        X_max, Y_max = MAX[0], MAX[1]
        MIN = np.min(kpt_new, axis=0)
        X_min, Y_min = MIN[0], MIN[1]
        box = [X_min, Y_min, X_max, Y_max]

        trans = kp_trans(box, dst_shape=(dst_shape[0], dst_shape[1]), size=size)

        # 大小缩方、位置变化
        kpts1 = affine_points(kpts, trans)

        ones = np.ones((kpts1.shape[0], 1), dtype=float)
        kpts1 = np.concatenate([kpts1, ones], axis=1)
        for i in range(len(kpts)):
            if kpts[i][2] == 0:
                kpts1[i][0] = kpts[i][0]
                kpts1[i][1] = kpts[i][1]
                kpts1[i][2] = kpts[i][2]

        kpts1 = np.array(kpts1).reshape(1, -1).tolist()
        # canvas = np.zeros(dst_shape, dtype=np.uint8)
        # img = show_skelenton(canvas, kpts1)
        # htpath = "test.jpg"
        # cv2.imwrite(htpath, img)
        return kpts1[0]


def get_imgs_id_have_all_keypoints(coco):
    catIds = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=catIds)
    imgs_id_have_all_keypoints = []
    for img_id in img_ids:
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = coco.loadAnns(annIds)
        is_have_all_keypoints = True
        for person_id, obj in enumerate(objs):
            if obj["num_keypoints"] != 17:
                is_have_all_keypoints = False
                break
        if is_have_all_keypoints:
            imgs_id_have_all_keypoints.append(img_id)
    return sorted(imgs_id_have_all_keypoints)

def get_imgs_id_have_keypoints(coco):
    catIds = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=catIds)
    imgs_id = []
    imgs_keypoints = []
    for img_id in img_ids:
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = coco.loadAnns(annIds)
        have_kp = 0
        k_num = 0
        mean_kp = 0
        for person_id, obj in enumerate(objs):
            if obj["num_keypoints"] != 0:
                k_num+=1
                have_kp+=int(obj["num_keypoints"])
        if k_num!=0:
            mean_kp = have_kp/k_num       
                    
            imgs_id.append(img_id)
            imgs_keypoints.append(mean_kp)
    return imgs_id,imgs_keypoints

def get_box(keypoint):
    kpts = np.array(keypoint).reshape(-1, 3)
    mask = np.logical_and(kpts[:, 0] != 0, kpts[:, 1] != 0)
    # 通过坐标点，找框，然后找中心点坐标，从而生成仿射矩阵，进行坐标点的变化
    konghang = []
    for i in range(len(kpts)):
        if kpts[i][2] == 0:
            konghang.append(i)
    kpt_new = np.delete(kpts, konghang, axis=0)

    MAX = np.max(kpt_new, axis=0).tolist()
    X_max, Y_max = MAX[0], MAX[1]
    MIN = np.min(kpt_new, axis=0).tolist()
    X_min, Y_min = MIN[0], MIN[1]
    return [X_min, Y_min, X_max, Y_max]


def get_useful_point(keypoint):
    kp = 0
    for i in range(0, len(keypoint), 3):
        # 计算相邻元素的和
        if i + 1 < len(keypoint):
            sum_of_pair = keypoint[i] + keypoint[i + 1]
            if sum_of_pair>0:
                kp+=1

    kpts = np.array(keypoint).reshape(-1, 3)
    mask = np.logical_and(kpts[:, 0] != 0, kpts[:, 1] != 0)
    # 通过坐标点，找框，然后找中心点坐标，从而生成仿射矩阵，进行坐标点的变化
    konghang = []
    for i in range(len(kpts)):
        if kpts[i][2] == 0:
            konghang.append(i)
    kpt_new = np.delete(kpts, konghang, axis=0)

    MAX = np.max(kpt_new, axis=0).tolist()
    X_max, Y_max = MAX[0], MAX[1]
    MIN = np.min(kpt_new, axis=0).tolist()
    X_min, Y_min = MIN[0], MIN[1]
    X_center = (X_min + X_max) / 2
    Y_center = (Y_min + Y_max) / 2
    return X_min, Y_min, X_max, Y_max, X_center, Y_center, kp


def load_coco(coco_json_path):
    # coco_json_path = "/root/autodl-tmp/data/person_keypoints_train2017.json"
    # coco_json_path = "/root/person_keypoints_coco_controlnet348_new.json"
    # coco_img_path = "."
    return COCO(coco_json_path)


###########################################################
def construct_box(keypoint, connection_keypoint):
    half_w = math.fabs(keypoint[0] - connection_keypoint[0])
    half_h = math.fabs(keypoint[1] - connection_keypoint[1])
    X_min = keypoint[0] - half_w
    Y_min = keypoint[1] - half_h
    X_max = keypoint[0] + half_w
    Y_max = keypoint[1] + half_h
    return [X_min, Y_min, X_max, Y_max]


def rotate_point(center_point, rotate_point, rotation=[-5, 5]):
    # 假设对图片上任意点(x,y)，绕一个坐标点 (rx0,ry0) 逆时针旋转a角度后的新的坐标设为(x0, y0)，有公式：
    # x0 = (x - rx0) * cos(a) - (y - ry0)*sin(a) + rx0;
    # y0 = (x - rx0) * sin(a) + (y - ry0)*cos(a) + ry0;
    angle = random.randint(*rotation)  # 角度制
    angle = angle / 180 * math.pi      # 弧度制
    x, y = rotate_point[0], rotate_point[1]
    rx0, ry0 = center_point[0], center_point[1]
    x0 = (x - rx0) * math.cos(angle) - (y - ry0) * math.sin(angle) + rx0
    y0 = (x - rx0) * math.sin(angle) + (y - ry0) * math.cos(angle) + ry0
    return [x0, y0]


def random_rotate_one_keypoints(origin_keypoints, keypoint_num, image_shape=(H, W)):
    # keypoint num 下标从 0 开始
    assert keypoint_num in rotate_list
    connection_keypoint_table = {
        8: 10,
        7: 9,
        14: 16,
        13: 15,
        6: 8,
        5: 7,
        12: 14,
        11: 13
    }
    origin_keypoints_np = np.array(origin_keypoints).reshape(-1, 3).astype(np.float32)
    connection_keypoint_num = connection_keypoint_table[keypoint_num]

    # 以 keypoint_num 为中心构建 box
    keypoint = origin_keypoints_np[keypoint_num].tolist()
    connection_keypoint = origin_keypoints_np[connection_keypoint_num].tolist()
    if keypoint[2] == 0 or connection_keypoint[2] == 0:
        # 不存在连接点，直接返回
        return origin_keypoints

    rotate_connection_keypoint = rotate_point(keypoint, connection_keypoint)

    origin_keypoints_np[connection_keypoint_num][0] = rotate_connection_keypoint[0]
    origin_keypoints_np[connection_keypoint_num][1] = rotate_connection_keypoint[1]
    if keypoint_num in [6, 5, 12, 11]:
        # 需要平移对应的 [10, 9, 16, 15] 关键点
        special_keypoint_table = {
            6: 10,
            5: 9,
            12: 16,
            11: 15
        }
        special_keypoint_num = special_keypoint_table[keypoint_num]
        special_keypoint = origin_keypoints_np[special_keypoint_num].tolist()
        if special_keypoint[2] != 0:
            special_relative_x = special_keypoint[0] - connection_keypoint[0]
            special_relative_y = special_keypoint[1] - connection_keypoint[1]
            origin_keypoints_np[special_keypoint_num][0] = rotate_connection_keypoint[0] + special_relative_x
            origin_keypoints_np[special_keypoint_num][1] = rotate_connection_keypoint[1] + special_relative_y

    new_keypoints = np.array(origin_keypoints_np).reshape(-1).tolist()
    return new_keypoints


def random_rotate_keypoints(origin_keypoints, image_shape=(H, W)):
    # 固定原始中心点不动，旋转关节
    for keypoint_num in rotate_list:
        origin_keypoints = random_rotate_one_keypoints(origin_keypoints, keypoint_num, image_shape)
    return origin_keypoints


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def box2cs( box):
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)

def xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x
    center[1] = y

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def random_trans_keypoints(origin_keypoints, image_shape=(H, W)):
    # 在区域内随机平移关节
    origin_keypoints_np = np.array(origin_keypoints).reshape(-1, 3).astype(np.float32)
    origin_box = get_box(origin_keypoints)
    origin_X_min, origin_Y_min, origin_X_max, origin_Y_max = origin_box
    w = origin_X_max - origin_X_min
    h = origin_Y_max - origin_Y_min
    ##### 随机变化
    # new_keypoints_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
    # new_keypoints_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
    # new_keypoints_center = np.array([new_keypoints_center_X, new_keypoints_center_Y])

    ##### 放在图片的正中心
    new_keypoints_center_X = W/2
    new_keypoints_center_Y = H/2
    new_keypoints_center = np.array([new_keypoints_center_X, new_keypoints_center_Y])

    trans = kp_trans_dst(origin_box, dst_shape=image_shape, dst_center=new_keypoints_center)
    new_keypoints_np = affine_points(origin_keypoints_np, trans)
    ones = np.ones((new_keypoints_np.shape[0], 1), dtype=float)
    new_keypoints_np = np.concatenate([new_keypoints_np, ones], axis=1)
    for i in range(len(origin_keypoints_np)):
        if origin_keypoints_np[i][2] == 0:
            new_keypoints_np[i][0] = origin_keypoints_np[i][0]
            new_keypoints_np[i][1] = origin_keypoints_np[i][1]
            new_keypoints_np[i][2] = origin_keypoints_np[i][2]
    new_keypoints = np.array(new_keypoints_np).reshape(-1).tolist()
    return new_keypoints_center_X, new_keypoints_center_Y, new_keypoints

