
"""
decode sequential output to visual locations
author: sierkinhane.github.io
"""
import random
from tqdm import tqdm
import json
import numpy as np
import re
import argparse
import cv2
import math
import os

# COCO keypoints
stickwidth = 3

limbSeq_coco = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

limbSeq_cp = [[14, 2], [14, 1], [2, 4], [4, 6], [1, 3], [3, 5], [14, 8], [8, 10], [10, 12], [14, 7], [7, 9], [9, 11], [13, 14]]

num_idx = 0
# CrowdPose
# {'0': 'left shoulder', '1': 'right shoulder', '2': 'left elbow', '3': 'right elbow', '4': 'left wrist', '5': 'right wrist', '6': 'left hip', '7': 'right hip', '8': 'left knee', '9': 'right knee', '10': 'left ankle', '11': 'right ankle', '12': 'head', '13': 'neck'}

# for human pose visualization
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# for box visualization
colors_box = [[217, 221, 116], [137, 165, 171], [230, 126, 175], [63, 157, 5], [107, 51, 75], [217, 147, 152], [129, 132, 8], [232, 85, 249], [254, 98, 33], [89, 108, 230], [253, 34, 161], [91, 150, 30], [255, 147, 26], [209, 154, 205], [134, 57, 11], [143, 181, 122], [241, 176, 87], [104, 73, 26], [122, 147, 59], [235, 230, 229], [119, 18, 125], [185, 61, 138], [237, 115, 90], [13, 209, 111], [219, 172, 212]]

# Plots one bounding box on image
def plot_one_box(x, img, color=None, label=None, line_thickness=None, idx=0):
     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 # line thickness
     color = color or [random.randint(0, 255) for _ in range(3)]
     color = colors_box[idx]
     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
     cv2.rectangle(img, c1, c2, color, thickness=tl)
     if label:
        tf = max(tl - 1, 1) # font thickness
     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
     cv2.rectangle(img, c1, c2, color, -1) # filled
     cv2.putText(img, label, c1, 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
     return img


# decode one sequence to visual locations
def decode(coordinate_str, type='box'):
    # 将str的字符串转化成numpy的格式
    # find numbers
    locations = np.array([int(i) for i in re.findall(r"\d+", coordinate_str)])

    if type == 'box':
        locations = locations.reshape(-1, 4)
    elif type == 'cocokeypoint':
        locations = locations.reshape(-1, 18, 2)
        visible = np.ones((locations.shape[0], 18, 1))
        eq_0_idx = np.where(locations[:, :, 0] * locations[:, :, 1] == 0)
        visible[eq_0_idx] = 0
        locations = np.concatenate([locations, visible], axis=-1)
        for i in range(locations.shape[0]):
            if locations[i, 2, -1] == 0 or locations[i, 5, -1] == 0:
                locations[i, 1, -1] = 0
    elif type == 'crowdpose':
        locations = locations.reshape(-1, 14, 2)
        visible = np.ones((locations.shape[0], 14, 1))
        eq_0_idx = np.where(locations[:, :, 0] * locations[:, :, 1] == 0)
        visible[eq_0_idx] = 0
        locations = np.concatenate([locations, visible], axis=-1)
    elif type == 'mask':
        locations = []
        for c_str in coordinate_str.split('m0'):
            c_str = ''.join(re.split(r'm\d+', c_str))
            mask_coord = np.array([int(i) for i in re.findall(r"\d+ ", c_str)])
            if len(mask_coord) != 0:
                locations.append(mask_coord.reshape(-1, 1, 2))
    else:
        raise NotImplementedError

    return locations


# process raw sequences inferred by VisorGPT 处理原始的预测序列
def to_coordinate_kp(file_path, ctn=True):

    if isinstance(file_path, list):
        texts = [i.strip().replace(' ##', '') for i in file_path]
    else:
        with open(file_path, 'r') as file:
            texts = [i.strip().replace(' ##', '') for i in file.readlines()]

    location_list = []
    classname_list = []
    type_list = []
    valid_sequences = []
    cnt = 0
    print('to coordinate ...')

    for ste in tqdm(texts):
        cnt += 1
        if 'box' in ste:
            type = 'box'
        elif 'key point' in ste:
            type = 'cocokeypoint' if '; 18 ;' in ste else 'crowdpose'  # 判断关键点的类型，如何包含18，则是coco的类型
        elif 'mask' in ste:
            type = 'mask'
        else:
            raise NotImplementedError

        if '[SEP]' not in ste:
            continue

        try:
            if ctn:
                temp = ste[:ste.index('[SEP]')].split(' ; ')[7].split('] ')
                classnames = []
                for t in temp:
                    classnames.append(t.split(' xmin ')[0].split(' m0')[0][2:])
                classnames = classnames[:-1]
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[7], type=type)

            else:
                classnames = ste[:ste.index('[SEP]')].split(' ; ')[7].split(' , ')
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[8], type=type)
        except:
            pass
        else:
            valid_sequences.append(ste[:ste.index('[SEP]')])
            location_list.append(locations)
            classname_list.append(classnames)
            type_list.append(type)

    with open('valid_sequences.txt', 'w') as file:
        [file.write(i.split('[CLS] ')[-1] + '\n') for i in valid_sequences]

    return location_list, classname_list, type_list, valid_sequences

def to_coordinate(file_path, ctn=True):

    if isinstance(file_path, list):
        texts = [i.strip().replace(' ##', '') for i in file_path]
    else:
        with open(file_path, 'r') as file:
            texts = [i.strip().replace(' ##', '') for i in file.readlines()]

    location_list = []
    classname_list = []
    type_list = []
    valid_sequences = []
    cnt = 0
    print('to coordinate ...')

    for ste in tqdm(texts):
        cnt += 1
        if 'box' in ste:
            type = 'box'
        elif 'key point' in ste:
            type = 'cocokeypoint' if '; 18 ;' in ste else 'crowdpose'  # 判断关键点的类型，如何包含18，则是coco的类型
        elif 'mask' in ste:
            type = 'mask'
        else:
            raise NotImplementedError

        if '[SEP]' not in ste:
            continue

        try:
            if ctn:
                temp = ste[:ste.index('[SEP]')].split(' ; ')[5].split('] ') # 先提取ste字符串中[SEP]之前的部分,然后按照;划分，取坐标的位置信息
                classnames = []
                for t in temp:
                    classnames.append(t.split(' xmin ')[0].split(' m0')[0][2:])
                classnames = classnames[:-1]
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[5], type=type)

            else:
                classnames = ste[:ste.index('[SEP]')].split(' ; ')[5].split(' , ')
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[6], type=type)
        except:
            pass
        else:
            valid_sequences.append(ste[:ste.index('[SEP]')])
            location_list.append(locations)
            classname_list.append(classnames)
            type_list.append(type)

    with open('valid_sequences.txt', 'w') as file:
        [file.write(i.split('[CLS] ')[-1] + '\n') for i in valid_sequences]

    return location_list, classname_list, type_list, valid_sequences



def pose_to_coordinate(file_path, ctn=True):

    if isinstance(file_path, list):
        texts = [i.strip().replace(' ##', '') for i in file_path]
    else:
        with open(file_path, 'r') as file:
            texts = [i.strip().replace(' ##', '') for i in file.readlines()]

    location_list = []
    classname_list = []
    type_list = []
    valid_sequences = []
    cnt = 0
    print('to coordinate ...')

    for ste in tqdm(texts):
        cnt += 1
        type = 'cocokeypoint'

        if '[SEP]' not in ste:
            continue

        try:
            if ctn:
                temp = ste[:ste.index('[SEP]')].split(' ; ')[5].split('] ') # 先提取ste字符串中[SEP]之前的部分,然后按照;划分，取坐标的位置信息
                classnames = []
                for t in temp:
                    classnames.append(t.split(' xmin ')[0].split(' m0')[0][2:])
                classnames = classnames[:-1]
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[5], type=type)

            else:
                classnames = ste[:ste.index('[SEP]')].split(' ; ')[5].split(' , ')
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[6], type=type)
        except:
            print("ERRO!!!!!")  # 出现异常的数据时，模型固定一个为0的数据进行存储。后续筛选时能够去除
            ste = '[CLS] 3 ; large ; s ; easy ; m8 m2 m9 ; [ person a 336 205 b 316 207 c 312 203 d 314 231 e 336 239 f 320 210 g 0 0 h 0 0 i 300 269 j 338 289 k 316 330 l 310 275 m 333 294 n 341 339 o 336 199 p 0 0 q 328 193 r 0 0 ] [ person a 48 114 b 24 149 c 3 145 d 3 205 e 45 225 f 45 152 g 68 200 h 96 224 i 12 246 j 37 323 k 23 388 l 52 248 m 58 329 n 64 394 o 44 105 p 52 108 q 24 112 r 0 0 ] [ person a 184 169 b 157 168 c 161 171 d 160 228 e 172 275 f 152 167 g 0 0 h 0 0 i 135 252 j 207 253 k 160 317 l 130 253 m 168 268 n 131 332 o 183 165 p 0 0 q 172 158 r 0 0 ] [SEP] 267 ] [SEP] 230 b 128 251 c 114 256 d 123 316 e 161 343 f 140 249 g 143 292 h 186 312 i 91 324 j 192 333 k 165 384 l 110 317 m 173 286 n 156 354 o 124 225 p 0 0 q 109 230 r 0 0 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 339 195 b 352 204 c 324 208 d 0 0 e 0 0 g 364 221 h 348 207 i 368 233 j 346 260 k 341 306 l 353 235 m 324 263 n 354 310 o 336 199 p 340 196 q 326 196 r 344 194 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 50 o 0 0 p 236 q 0 0 r 99 202 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] p 19 217 r ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] m 539 147 b 65 [SEP] [SEP] [SEP] 131 p 236 q p 48 r 127 r 0 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 203 347 200 m 534 290 p 52 199 p 0 0 q 200 r 86 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 53 ] [SEP] [SEP] 337 155 250 b m [SEP] [SEP] [SEP] ; [SEP] 341 129 b 200 53 119 [SEP] easy [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 285 203 21 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 119 h 335 p 203 [SEP] [SEP] [SEP] 21 128 127 336 q 218 r 121 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 203 225 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 341 q 143 138 b m [SEP] [SEP] 136 [SEP] [SEP] [SEP] [SEP] 144 195 ] [SEP] [SEP] [SEP] [SEP] [SEP] 130 b 128 m [SEP] 142 129 339 200 21 23 l [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 226 q 213 p 313 137 ] [SEP] [SEP] 249 101 [SEP] 286 b 199 r ; 85 ] [SEP] 144 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 233 ] [SEP] [SEP] [SEP] [SEP] 86 na 144 241 d 131 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] easy 281 139 h 127 248 81 [SEP] [SEP] [SEP] [SEP] 203 144 p m 97 21 200 r 144 p 16 [SEP] [SEP] [SEP] 135 easy 288 151 204 p 32 137 hard [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 285 e 29 248 p 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 138 h 250 p [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 250 q 135 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP]'
            locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[5], type=type)
            classnames = ['person a 336 205 b 316 207 c 312 203 d 314 231 e 336 239 f 320 210 g 0 0 h 0 0 i 300 269 j 338 289 k 316 330 l 310 275 m 333 294 n 341 339 o 336 199 p 0 0 q 328 193 r 0 0 ', 'person a 48 114 b 24 149 c 3 145 d 3 205 e 45 225 f 45 152 g 68 200 h 96 224 i 12 246 j 37 323 k 23 388 l 52 248 m 58 329 n 64 394 o 44 105 p 52 108 q 24 112 r 0 0 ', 'person a 184 169 b 157 168 c 161 171 d 160 228 e 172 275 f 152 167 g 0 0 h 0 0 i 135 252 j 207 253 k 160 317 l 130 253 m 168 268 n 131 332 o 183 165 p 0 0 q 172 158 r 0 0 ']
            
            valid_sequences.append(ste[:ste.index('[SEP]')])
            location_list.append(locations)
            classname_list.append(classnames)
            type_list.append(type)

        else:
            valid_sequences.append(ste[:ste.index('[SEP]')])
            location_list.append(locations)
            classname_list.append(classnames)
            type_list.append(type)

    # with open('valid_sequences.txt', 'w') as file:
    #     [file.write(i.split('[CLS] ')[-1] + '\n') for i in valid_sequences]

    return location_list, classname_list, type_list, valid_sequences


# visualize object locations on a canvas
def visualization(location_list, classname_list, type_list, save_dir='debug/', save_fig=False):

    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print('visualizing ...')
    for b, (loc, classnames, type) in tqdm(enumerate(zip(location_list, classname_list, type_list))):
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)

        if len(loc) != len(classnames):
            continue
        
        if type == 'box':
            for i in range(loc.shape[0]):
                canvas = plot_one_box(loc[i], canvas, label=classnames[i], idx=i)

        elif type == 'cocokeypoint':
            for i in range(loc.shape[0]):

                for j in range(loc.shape[1]):
                    x, y, v = loc[i, j]
                    if v != 0:
                        cv2.circle(canvas, (int(x), int(y)), 3, colors[j], thickness=-1)


                for j in range(17):
                    lim = limbSeq_coco[j]
                    cur_canvas = canvas.copy()
                    Y = [loc[i][lim[0] - 1][0], loc[i][lim[1] - 1][0]]
                    X = [loc[i][lim[0] - 1][1], loc[i][lim[1] - 1][1]]

                    if loc[i][lim[0] - 1][-1] == 0 or loc[i][lim[1] - 1][-1] == 0:
                        continue

                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[j])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        elif type == 'crowdpose':
            for i in range(loc.shape[0]):
                for j in range(loc.shape[1]):
                    x, y, _ = loc[i, j]
                    if x != 0 and y != 0:
                        cv2.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)
                for j in range(13):
                    lim = limbSeq_cp[j]
                    cur_canvas = canvas.copy()

                    Y = [loc[i][lim[0] - 1][0], loc[i][lim[1] - 1][0]]
                    X = [loc[i][lim[0] - 1][1], loc[i][lim[1] - 1][1]]

                    if (Y[0] == 0 and X[0] == 0) or (Y[1] == 0 and X[1] == 0):
                        continue

                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[j])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        elif type == 'mask':
            for i in range(len(loc)):
                color = [random.randint(0, 255) for _ in range(3)]
                xmin, ymin, xmax, ymax = loc[i][:, :, 0].min(), loc[i][:, :, 1].min(), loc[i][:, :, 0].max(), loc[i][:, :, 1].max()
                cur_canvas = canvas.copy()
                cv2.fillPoly(cur_canvas, [loc[i]], color)
                cur_canvas = plot_one_box((xmin, ymin, xmax, ymax), cur_canvas, color=color, label=classnames[i])
                canvas = cv2.addWeighted(canvas, 0.5, cur_canvas, 0.5, 0)
        else:
            raise NotImplementedError
        if save_fig:
            cv2.imwrite(f'{save_dir}/test_{b}.png', canvas[..., ::-1])
            
    return canvas[..., ::-1]


# visualize object locations on a canvas
def visualization_oneperson(location_list, classname_list, type_list, save_dir='debug/', save_fig=False):

    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print('visualizing ...')
    for b, (loc, classnames, type) in tqdm(enumerate(zip(location_list, classname_list, type_list))):
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        ######
        save_dir  = "/storage/zhaoliuqing/data/tt1228"
        if len(loc) != len(classnames):
            continue
        
        if type == 'box':
            for i in range(loc.shape[0]):
                canvas = plot_one_box(loc[i], canvas, label=classnames[i], idx=i)

        elif type == 'cocokeypoint':
            #  每次只画一个人
            for i in range(loc.shape[0]):
                canvas_oneperson = np.zeros((512, 512, 3), dtype=np.uint8)
                for j in range(loc.shape[1]):
                    x, y, v = loc[i, j]
                    if v != 0:
                        cv2.circle(canvas, (int(x), int(y)), 3, colors[j], thickness=-1)
                        cv2.circle(canvas_oneperson, (int(x), int(y)), 3, colors[j], thickness=-1)
                for j in range(17):
                    lim = limbSeq_coco[j]
                    cur_canvas = canvas.copy()
                    cur_canvas_oneperson = canvas_oneperson.copy()

                    Y = [loc[i][lim[0] - 1][0], loc[i][lim[1] - 1][0]]
                    X = [loc[i][lim[0] - 1][1], loc[i][lim[1] - 1][1]]

                    if loc[i][lim[0] - 1][-1] == 0 or loc[i][lim[1] - 1][-1] == 0:
                        continue

                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[j])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

                    cv2.fillConvexPoly(cur_canvas_oneperson, polygon, colors[j])
                    canvas_oneperson = cv2.addWeighted(canvas_oneperson, 0.4, cur_canvas_oneperson, 0.6, 0)

                # cv2.imwrite(f'{save_dir}/test_{num_idx}.png', canvas_oneperson[..., ::-1])
                # print(num_idx)
                # num_idx+=1


            
    return canvas_oneperson[..., ::-1]

# to json output
def to_json(location_list, classname_list, type_list, valid_sequences):

    ret_json_box = {'bboxes': [], 'sequences': []}
    ret_json_mask = {'masks': [], 'sequences': []}
    ret_json_keypoint = {'keypoints': [], 'sequences': []}
    print('to json ...')
    for loc, classnames, type, seq in tqdm(zip(location_list, classname_list, type_list, valid_sequences)):
        ins_list = []
        kpt_list = []
        mask_list = []
        seq_list = []
        if len(loc) != len(classnames):# or len(classnames) > 8:
            continue

        if type == 'box':
            for i in range(loc.shape[0]):
                # xmin, ymin, xmax, ymax = loc[i]
                # area = (xmax - xmin) * (ymax - ymin)
                # compute area and omit very small one due to the synthesis ability of AIGC
                # if area < 32**2:
                #     continue

                dic = {classnames[i]: loc[i].tolist()}
                ins_list.append(dic)
                if len(seq_list) == 0:
                    seq_list.append(seq)

        elif type == 'cocokeypoint' or type == 'crowdpose':
            for i in range(loc.shape[0]):
                # compute validate key points and omit the less one, as the synthesis ability of AIGC
                # if loc[i, :, -1].sum() <= 4:
                #     continue

                # compute area and omit very small one due to the synthesis ability of AIGC
                # xmin, ymin, xmax, ymax = loc[i, :, 0].min(), loc[i, :, 1].min(), loc[i, :, 0].max(), loc[i, :, 1].max()
                # area = (xmax - xmin) * (ymax - ymin)
                # if area < 32 ** 2:
                #     continue

                dic = {classnames[i]: loc[i][:, :].tolist()}
                kpt_list.append(dic)
                if len(seq_list) == 0:
                    seq_list.append(seq)

        elif type == 'mask':
            for i in range(len(loc)):

                # xmin, ymin, xmax, ymax = loc[i][:, :, 0].min(), loc[i][:, :, 1].min(), loc[i][:, :, 0].max(), loc[i][:, :, 1].max()
                # area = (xmax - xmin) * (ymax - ymin)
                # if area < 32 ** 2:
                #     continue

                dic = {classnames[i]: loc[i].tolist()}
                mask_list.append(dic)
                if len(seq_list) == 0:
                    seq_list.append(seq)
        else:
            raise NotImplementedError

        if len(ins_list) != 0:
            ret_json_box['bboxes'].append(ins_list)
            ret_json_box['sequences'].append(seq_list)
        if len(kpt_list) != 0:
            ret_json_keypoint['keypoints'].append(kpt_list)
            ret_json_keypoint['sequences'].append(seq_list)
        if len(mask_list) != 0:
            ret_json_mask['masks'].append(mask_list)
            ret_json_mask['sequences'].append(seq_list)

    return [ret_json_box, ret_json_mask, ret_json_keypoint]


def gen_cond_mask(texts, ctn):
    # texts = ['[CLS] 3 ; large ; s ; easy ; m9 m1 m8 ; [ person a 232 176 b 194 212 c 216 221 d 219 246 e 230 259 f 172 203 g 155 203 h 169 194 i 187 303 j 0 0 k 0 0 l 161 296 m 0 0 n 0 0 o 231 171 p 228 169 q 229 178 r 219 175 ] [ person a 0 0 b 462 185 c 482 184 d 511 219 e 0 0 f 441 186 g 447 234 h 471 256 i 492 292 j 51 ##3 349 k 52 ##5 411 l 460 296 m 470 343 n 502 406 o 0 0 p 0 0 q 436 158 r 435 164 ] [ person a 304 260 247 c 294 258 d 344 305 e 373 341 f 225 238 g 171 274 h 194 334 i 258 346 j 321 361 k 0 0 l 201 346 m 262 402 n 0 0 o 310 234 p 300 229 q 0 0 r 274 222 ] [SEP] 142 ] [SEP] 144 p 292 155 b 296 171 c 317 173 d 295 200 f 276 165 g 249 191 h 259 199 i 284 234 j 0 0 k 0 0 l 250 234 m 0 0 n 0 0 o 302 146 p 287 145 q 306 153 r 284 146 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 351 225 b 358 252 c 380 257 d 362 283 e 323 296 f 336 241 h 358 214 i 366 297 j 0 0 k 0 0 l 336 300 m 0 0 n 0 0 o 357 220 p 351 220 q 365 223 r 351 223 ] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] k 497 361 l 418 323 m 0 0 n 0 0 o 202 218 q 212 225 r 203 238 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] l 0 0 m 56 215 n 0 0 0 p 21 q 203 200 q 204 r ; [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 204 p 313 l m [SEP] [SEP] [SEP] [SEP] 142 p 24 249 p 4 r 113 142 r 147 150 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 203 l 204 m [SEP] [SEP] [SEP] [SEP] [SEP] 144 p [SEP] 154 p 342 145 p 341 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 142 189 n [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 199 r ; [SEP] [SEP] 248 q 200 n [SEP] [SEP] [SEP] [SEP] [SEP] l na q 141 204 r 130 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 199 203 r ; [SEP] [SEP] [SEP] [SEP] [SEP] 204 205 r 152 197 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 142 h [SEP] [SEP] 210 q 197 r 144 200 101 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 142 200 195 b [SEP] [SEP] [SEP] [SEP] [SEP] 203 p 47 134 348 197 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 203 p [SEP] [SEP] 231 204 ] [SEP] [SEP] [SEP] [SEP] [SEP] 154 248 204 r ; 127 ] [SEP] 203 ] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 200 l [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] large 197 ] [SEP] [SEP] [SEP] [SEP] 199 194 hard [SEP] [SEP] easy 227 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 144 199 r 142 241 [SEP] 136 203 154 199 197 ] [SEP] [SEP] [SEP] 138 200 ] [SEP] [SEP] [SEP] [SEP] [SEP] 203 l [SEP] 231 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 141 b [SEP] [SEP] [SEP] [SEP] [SEP] 142 256 144 [SEP] [SEP] [SEP] [SEP] 236 200 212 ] [SEP] [SEP] [SEP] [SEP] 152 205 195 p [SEP] [SEP] 152 200 204 [SEP] [SEP] 245 249 249 188 [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 204 r 50 268 130 199 180 208 r 144 198 339 136 199 e 122 339 p n 143 250 138 215 159 231 250 81 248 251 o [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] 245 p [SEP] [SEP] 146 p 139 200 ] [SEP] [SEP] 147 203 189 302 158 240 p 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0']
    location_list, classname_list, type_list, valid_sequences = pose_to_coordinate(texts, ctn)
    ret_mask = visualization(location_list, classname_list, type_list, None, False)  # 关键点图像
    # ret_mask = visualization_oneperson(location_list, classname_list, type_list, None, False)  # 关键点图像  每个人都画
    ret_json = to_json(location_list, classname_list, type_list, valid_sequences)   # 关键点的信息
    return ret_mask, ret_json


def gen_cond_mask_kp(texts, ctn):
    location_list, classname_list, type_list, valid_sequences = pose_to_coordinate(texts, ctn)
    ret_mask = visualization(location_list, classname_list, type_list, None, False)  # 关键点图像
    ret_json = to_json(location_list, classname_list, type_list, valid_sequences)   # 关键点的信息
    return ret_mask, ret_json

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='debug')
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    location_list, classname_list, type_list, valid_sequences = to_coordinate(args.file_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # visualization
    if args.visualize:
        visualization(location_list, classname_list, type_list, args.save_dir)

    # to json data
    rets = to_json(location_list, classname_list, type_list, valid_sequences)

    for ret, flag in zip(rets, ['box', 'mask', 'keypoint']):
        save_path = args.file_path.split('/')[-1].split('.')[0] + f'_{flag}.json'
        with open('files/' + save_path, 'w') as file:
            json.dump(ret, file, indent=2)



