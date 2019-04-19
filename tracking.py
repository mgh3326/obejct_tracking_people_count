#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import warnings
from timeit import time

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from models import *
from optical_flow.applyGeometricTransformation import applyGeometricTransformation
from optical_flow.estimateAllTranslation import estimateAllTranslation
# from optical_flow.optical import optical_flow_tracking
from optical_flow.getFeatures import getFeatures
from tools import generate_detections as gdet
from yolo import YOLO
from keras.utils import get_file
import cvlib as cv
from keras.preprocessing.image import img_to_array

warnings.filterwarnings('ignore')


# transform newbboxs of (n_object,4,2) np array s.t. return_boxs = bbox_transform(newbboxs)
# newbboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
# return_boxs = [], return_boxs.append([x,y,w,h])
def bbox_transform(newbboxs):
    return_boxs = []
    for i in range(newbboxs.shape[0]):
        [x, y, w, h] = [newbboxs[i, 0, 0], newbboxs[i, 0, 1], newbboxs[i, 3, 0] - newbboxs[i, 0, 0],
                        newbboxs[i, 3, 1] - newbboxs[i, 0, 1]]
        return_boxs.append([x, y, w, h])
    return return_boxs


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_segement(p1, p2):
    ret = list()
    ret.append(p1[0])
    ret.append(p2[0])
    ret.append(p1[1])
    ret.append(p2[1])

    return ret


def intersection(s1, s2):
    left = max(min(s1[0], s1[2]), min(s2[0], s2[2]))
    right = min(max(s1[0], s1[2]), max(s2[0], s2[2]))
    top = max(min(s1[1], s1[3]), min(s2[1], s2[3]))
    bottom = min(max(s1[1], s1[3]), max(s2[1], s2[3]))

    if top > bottom or left > right:
        return False
    if (top, left) == (bottom, right):
        return list((left, top))
    return list((left, bottom, right, top))


def draw_people_point_line(image):
    for people in people_list:

        for i in range(len(people.points) - 1):
            cv2.line(image, (people.points[i].x, people.points[i].y), (people.points[i + 1].x, people.points[i + 1].y),
                     people.color,
                     3)
            # 선 그린 이후에 count를 셀수 있다로고 하자
            # person_line = line_segement((people.points[i].x, people.points[i].y),
            #                             (people.points[i + 1].x, people.points[i + 1].y))


def draw_count_text(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX  # normal size serif font

    text = "in man count : " + str(count.get("in_man_count")) + "       out man count : " + str(
        count.get("out_man_count"))
    text += "\nin woman count : " + str(count.get("in_woman_count")) + "       out woman count : " + str(
        count.get("out_woman_count"))
    x = int((frame.shape[1] / 10) * 1)
    y0, dy = int((frame.shape[0] / 10) * 9), 40
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    # cv2.putText(frame, text, org, font, 2, (0, 255, 255), 2)


def draw_count_line(image):
    # center
    cv2.line(image, out_start_point, out_end_point, (0, 0, 255), 2)
    cv2.line(image, center_start_point, center_end_point, (255, 0, 0), 5)
    cv2.line(image, in_start_point, in_end_point, (0, 0, 255), 2)


def append_point(point: Point, people_id: int, image):
    # Get people
    for _people in people_list:
        if _people.id == people_id:
            if not _people.is_in_line:
                # ret = intersection(person_line, in_line)
                ret = intersect(_people.points[len(_people.points) - 1], point,
                                Point(in_start_point[0], in_start_point[1]),
                                Point(in_end_point[0], in_end_point[1]))
                if ret:
                    _people.is_in_line = True

            else:
                ret = intersect(_people.points[len(_people.points) - 1], point,
                                Point(center_start_point[0], center_start_point[1]),
                                Point(center_end_point[0], center_end_point[1]))

                if ret:
                    if _people.gender.name == "woman":
                        count["in_woman_count"] = count.get("in_woman_count") + 1
                    else:
                        count["in_man_count"] = count.get("in_man_count") + 1

                    _people.is_in_line = False
                    _people.is_center_line = False

            if not _people.is_out_line:
                ret = intersect(_people.points[len(_people.points) - 1], point,
                                Point(out_start_point[0], out_start_point[1]),
                                Point(out_end_point[0], out_end_point[1]))
                if ret:
                    _people.is_out_line = True
            else:
                ret = intersect(_people.points[len(_people.points) - 1], point,
                                Point(center_start_point[0], center_start_point[1]),
                                Point(center_end_point[0], center_end_point[1]))
                if ret:
                    if _people.gender.name == "woman":
                        count["out_woman_count"] = count.get("out_woman_count") + 1
                    else:
                        count["out_man_count"] = count.get("out_man_count") + 1

                    _people.is_out_line = False
            _people.points.append(point)
            ###Face detection
            # cv2.imwrite("temp.jpg", image)
            # 크기가 어느정도 작게 되면 예외 처리 하는것도 좋은 방법 같다.
            if image.shape[0] == 0 or image.shape[1] == 0:
                return False
            try:
                face, confidence = cv.detect_face(image)
            except Exception as e:
                return False

            classes = ['man', 'woman']

            # loop through detected faces
            for idx, f in enumerate(face):
                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                if startX < 0 or startY < 0 or endX < 0 or endY < 0:
                    continue
                if startX > image.shape[1] or startY > image.shape[0] or endX > image.shape[1] or endY > image.shape[0]:
                    continue

                # draw rectangle over face
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # crop the detected face region
                face_crop = np.copy(image[startY:endY, startX:endX])

                # preprocessing for gender detection model
                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue
                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0]
                print(conf)
                print(classes)

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]
                if _people.gender.accuracy is not None and _people.gender.accuracy < conf[idx]:
                    _people.gender = Gender(name=classes[idx], accuracy=conf[idx] * 100)

                # label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                #
                # Y = startY - 10 if startY - 10 > 10 else startY + 10
                #
                # # write label and confidence above face rectangle
                # cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 255, 0), 2)
                #
                # cv2.imwrite("result.jpg", image)

            return

    # not found people
    people = People(people_id, [point], False, False, False, Age(20, 0), Gender("man", 0))
    import random
    r = lambda: random.randint(0, 255)
    g = lambda: random.randint(0, 255)
    b = lambda: random.randint(0, 255)
    people.color = (r(), g(), b())
    people_list.append(people)


def detect_face_gender(image):
    face, confidence = cv.detect_face(image)

    classes = ['man', 'woman']

    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY, startX:endX])

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        print(conf)
        print(classes)

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap1 = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    OPTICAL = False

    # video_filename = './dataset/people.mp4'
    # video_filename = 'C:/tensorflow1\models/research\object_detection/videos/IMG_1101.MOV'
    video_filename = 'C:/tensorflow1\models/research\object_detection/videos/IMG_1105-diet.mp4'
    video_capture = cv2.VideoCapture(video_filename)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('C:/tensorflow1\models/research\object_detection/videos/output_0419.avi', fourcc, 30,
                              (w, h))
        list_file = open('detection.txt', 'w')
        list_file2 = open('tracking.txt', 'w')
        frame_index = -1

    fps = 0.0
    firstflag = 1
    while True:
        ok, frame = video_capture.read()  # frame shape 640*480*3
        # cv2.imwrite("test.png", frame)
        # exit()

        if ok != True:
            break;
        t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)  # [x,y,w,h]
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression (NMS)
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
        detections = [detections[i] for i in indices]

        ### Call the tracker
        tracker.predict()
        tracker.update(detections)

        ### Add one more step of optical flow
        # convert detections to bboxs for optical flow
        n_object = len(detections)
        bboxs = np.empty((n_object, 4, 2), dtype=float)
        i = 0
        for det in detections:
            bbox = det.to_tlbr()  # (min x, min y, max x, max y)
            (xmin, ymin, boxw, boxh) = (
                int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))
            bboxs[i, :, :] = np.array(
                [[xmin, ymin], [xmin + boxw, ymin], [xmin, ymin + boxh], [xmin + boxw, ymin + boxh]]).astype(float)
            i = i + 1

        if firstflag:
            oldframe = frame
        else:

            startXs, startYs = getFeatures(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), bboxs, use_shi=False)
            newXs, newYs = estimateAllTranslation(startXs, startYs, oldframe, frame)
            Xs, Ys, newbboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)
            oldframe = frame
            ## generate new detections
            boxs = bbox_transform(newbboxs)
            features = encoder(frame, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
            detections = [detections[i] for i in indices]

            ## Call the tracker again
            tracker.predict()
            tracker.update(detections)
        origin_frame = frame.copy()
        draw_count_line(frame)
        draw_people_point_line(frame)
        draw_count_text(frame)
        boxes_tracking = np.array([track.to_tlwh() for track in tracker.tracks])
        ### Deep sort tracker visualization
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            central_point = Point(int((bbox[0] + bbox[2]) / 2)
                                  , int((bbox[1] + bbox[3]) / 2 + (bbox[3] - bbox[1]) / 3))
            # 원본 이미지를 넣어줌
            crop_img = origin_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            append_point(central_point, track.track_id, crop_img)
            # detect_face_gender(crop_img)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        ### Start from the first frame, do optical flow for every two consecutive frames.
        if OPTICAL:
            if firstflag:
                n_object = len(detections)
                bboxs = np.empty((n_object, 4, 2), dtype=float)
                i = 0
                for det in detections:
                    bbox = det.to_tlbr()  # (min x, min y, max x, max y)
                    (xmin, ymin, boxw, boxh) = (
                        int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))
                    bboxs[i, :, :] = np.array(
                        [[xmin, ymin], [xmin + boxw, ymin], [xmin, ymin + boxh], [xmin + boxw, ymin + boxh]]).astype(
                        float)
                    i = i + 1
                startXs, startYs = getFeatures(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), bboxs, use_shi=False)
                oldframe = frame
                oldbboxs = bboxs
            else:
                ### add new tracking object
                # new_n_object = len(detections)
                # if new_n_object > n_object:
                #     # Run non-maxima suppression (NMS)
                #     tmp_boxes = np.array([d.tlwh for d in detections])
                #     tmp_scores = np.array([d.confidence for d in detections])
                #     tmp_indices = preprocessing.non_max_suppression(tmp_boxes, nms_max_overlap2, tmp_scores)
                #     tmp_detections = [detections[i] for i in indices]
                # if len(tmp_detections)>n_object:

                newXs, newYs = estimateAllTranslation(startXs, startYs, oldframe, frame)
                Xs, Ys, newbboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, oldbboxs)
                # update coordinates
                (startXs, startYs) = (Xs, Ys)

                oldframe = frame
                oldbboxs = newbboxs

                # update feature points as required
                n_features_left = np.sum(Xs != -1)
                print('# of Features: %d' % n_features_left)
                if n_features_left < 15:
                    print('Generate New Features')
                    startXs, startYs = getFeatures(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), newbboxs)

                # draw bounding box and visualize feature point for each object
                for j in range(n_object):
                    (xmin, ymin, boxw, boxh) = cv2.boundingRect(newbboxs[j, :, :].astype(int))
                    cv2.rectangle(frame, (xmin, ymin), (xmin + boxw, ymin + boxh), (255, 255, 255), 2)  # BGR color
                    cv2.putText(frame, str(j), (xmin, ymin), 0, 5e-3 * 200, (0, 255, 0), 2)
                    # red color features
                    # for k in range(startXs.shape[0]):
                    #     cv2.circle(frame, (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0),
                          2)  # BGR color

        # cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            # detection
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
            # tracking
            list_file2.write(str(frame_index) + ' ')
            if len(boxes_tracking) != 0:
                for i in range(0, len(boxes_tracking)):
                    list_file2.write(str(boxes_tracking[i][0]) + ' ' + str(boxes_tracking[i][1]) + ' ' + str(
                        boxes_tracking[i][2]) + ' ' + str(boxes_tracking[i][3]) + ' ')
            list_file2.write('\n')

        firstflag = 0

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dwnld_link = "https://s3.ap-south-1.amazonaws.com/arunponnusamy/pre-trained-weights/gender_detection.model"
    model_path = get_file("gender_detection.model", dwnld_link,
                          cache_subdir="pre-trained", cache_dir=os.getcwd())
    model = load_model(model_path)

    people_list = list()
    # count = dict(in_count=0,
    #              out_count=0)
    count = dict(in_man_count=0,
                 in_woman_count=0,
                 out_man_count=0,
                 out_woman_count=0

                 )
    # save_line
    # center_start_point = (0, 1463)
    # center_end_point = (3200, 1557)
    # # in
    # in_start_point = (0, 1504)
    # in_end_point = (3200, 1609)
    # # out
    # out_start_point = (0, 1392)
    # out_end_point = (3200, 1469)

    center_start_point = (0, 658)
    center_end_point = (1920, 658)
    # in
    in_start_point = (0, 720)
    in_end_point = (1920, 720)
    # out
    out_start_point = (0, 600)
    out_end_point = (1920, 600)
    main(YOLO())
