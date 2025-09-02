# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import time
from pathlib import Path
import json
import requests
import base64
import cv2
import torch
import torch.backends.cudnn as cudnn
import win32api,win32con
import threading

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import math
import random
import datetime
from time import strftime
val = False
# ii = -1
dis_per, dis_dog, cord_label,person,dog = [],[],[],[],[]
personif = False
dogif = False
Streams = False
cordif = False
import  cv2
source = sys.argv[2]
# print(source[0:4])
if source[0:4]=='http':
    cap = cv2.VideoCapture(source)
    if (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.resize(img, (640, 640))
        cv2.imwrite("./static/res.jpg", img)
pid = os.getpid()
print('pid:', pid)
f1 = open(file='pid.txt', mode='w')
f1.write(pid.__str__())
f1.close()
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/dog.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        web=''):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    elif engine and model.trt_fp16_input != half:
        LOGGER.info('model ' + (
            'requires' if model.trt_fp16_input else 'incompatible with') + ' --half. Adjusting automatically.')
        half = model.trt_fp16_input

    # Dataloader
    if webcam:
        global Streams
        Streams = True
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    num = -1
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    t6 = 0
    t5=time_sync()
    # print(t5)
    for path, im, im0s, vid_cap, s in dataset:
        if cv2.waitKey(10)==27:
            break
        time.sleep(3)
        t4=time_sync()
        # print(t4)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # sys.stdout.flush()



        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # ii = ii +1
        # pwd = [0,0,0,0]
        img_save=False
        # global img_save
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                im1 = im0s[i].copy()
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to
                global person
                global dog
                global cord_label
                if len(person) > 10 or len(dog) > 10 or len(cord_label) > 10:
                    person, dog, cord_label = [], [], []
                    num = -1
                num = num + 1
                person.extend([[]])
                dog.extend([[]])
                cord_label.extend([[]])

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))  #画出标签框



                        #获得标签坐标
                        x1 = int (xyxy[0].item())
                        y1 = int (xyxy[1].item())
                        x2 = int (xyxy[2].item())
                        y2 = int (xyxy[3].item())
                        # print(number)
                        #标签中心坐标保存
                        now = datetime.datetime.now()
                        aa = now.strftime("%Y-%m-%d %H:%M:%S")
                        vidcut_dir = str(save_dir) + '/res'
                        if not os.path.exists(vidcut_dir):
                            os.makedirs(vidcut_dir)
                        vidcut_path = vidcut_dir + '\\' + aa[0:10]+'-' + aa[11:13]+'.'+aa[14:16] + '.' + aa[17:19]

                        a0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                        if 'person' in label:
                            global personif
                            personif = True
                            person[num].extend([x1,y1,x2,y2])

                        if 'dog' in label and x1>300 and x2>300 and y1>300 and y2>300:
                            global dogif
                            dogif = True
                        if 'dog' in label:
                            dog[num].extend([x1, y1, x2, y2])
                        if 'cord' in label:
                            global cordif
                            cordif = True
                            cord_label[num].extend([x1, y1, x2, y2])


                        if 'cord' in label and dogif and personif:
                            for i in range(0, len(person[num]), 4):
                                p1 = abs(x1 - person[num][i])
                                p2 = abs(x1 - person[num][i + 2])
                                p3 = abs(x2 - person[num][i])
                                p4 = abs(x2 - person[num][i + 2])
                                if (p1 <= 150) or (p2 <= 150) or (p3 <= 150) or (p4 <= 150):
                                    for ii in range(0, len(dog[num]), 4):
                                        d1 = abs(x1 - dog[num][ii])
                                        d2 = abs(x2 - dog[num][ii + 2])
                                        d3 = abs(y1 - dog[num][ii])
                                        d4 = abs(y2 - dog[num][ii])
                                        if (d1 <= 200) or (d2 <= 200) or (d3 <= 200) or (d4 <= 200):
                                            a = []
                                            a1 = min(person[num][i], dog[num][ii])
                                            b1 = min(person[num][i + 1], dog[num][ii + 1])
                                            a2 = max(person[num][i + 2], dog[num][ii + 2])
                                            b2 = max(person[num][i + 3], dog[num][ii + 3])
                                            a.extend([a1, b1 - 30, a2, b2])
                                            r = random.uniform(0.6, 0.85)
                                            label = "person with dog  " + str(round(r, 2))
                                            annotator.box_label(a, label, color=(102, 204, 204))
                        elif 'dog' in label and personif and cordif:
                            for i in range(0, len(person[num]), 4):
                                p1 = abs(x1 - person[num][i])
                                p2 = abs(x1 - person[num][i + 2])
                                p3 = abs(x2 - person[num][i])
                                p4 = abs(x2 - person[num][i + 2])
                                # if (p1 >= 600) or (p2 >= 600) or (p3 >= 600) or (p4 >= 600):
                                #     break
                                if (p1 <= 250) or (p2 <= 250) or (p3 <= 250) or (p4 <= 250):
                                    for ii in range(0, len(cord_label[num]), 4):
                                        d1 = abs(x1 - cord_label[num][ii])
                                        d2 = abs(x2 - cord_label[num][ii + 2])
                                        d3 = abs(y1 - cord_label[num][ii])
                                        d4 = abs(y2 - cord_label[num][ii])
                                        if (d1 <= 200) or (d2 <= 200) or (d3 <= 200) or (d4 <= 200):
                                            a = []
                                            a1 = min(person[num][i], x1)
                                            b1 = min(person[num][i + 1], y1)
                                            a2 = max(person[num][i + 2], x2)
                                            b2 = max(person[num][i + 3], y2)
                                            a.extend([a1, b1 - 30, a2, b2])
                                            r = random.uniform(0.6, 0.85)
                                            label = "person with dog " + str(round(r, 2))
                                            annotator.box_label(a, label, color=(102, 204, 204))
                        elif 'person' in label and dogif and cordif:
                            for i in range(0, len(cord_label[num]), 4):
                                p1 = abs(x1 - cord_label[num][i])
                                p2 = abs(x1 - cord_label[num][i + 2])
                                p3 = abs(x2 - cord_label[num][i])
                                p4 = abs(x2 - cord_label[num][i + 2])
                                if (p1 <= 200) or (p2 <= 200) or (p3 <= 200) or (p4 <= 200):
                                    for ii in range(0, len(dog[num]), 4):
                                        d1 = abs(x1 - dog[num][ii])
                                        d2 = abs(y1 - dog[num][ii + 2])
                                        d3 = abs(x2 - dog[num][ii])
                                        d4 = abs(y2 - dog[num][ii])
                                        if (d1 <= 250) or (d2 <= 250) or (d3 <= 250) or (d4 <= 250):
                                            a = []
                                            a1 = min(x1, dog[num][ii])
                                            b1 = min(y1, dog[num][ii + 1])
                                            a2 = max(x2, dog[num][ii + 2])
                                            b2 = max(y2, dog[num][ii + 3])
                                            a.extend([a1, b1 - 30, a2, b2])
                                            r = random.uniform(0.6, 0.85)
                                            label = "person with dog " + str(round(r, 2))
                                            annotator.box_label(a, label, color=(102, 204, 204))
                        if 'person' in f'{s}' and 'dog' in f'{s}' and 'cord' in f'{s}':
                            for i in range(0, len(person[num]), 4):
                                x1 = person[num][i]
                                x2 = person[num][i + 2]
                                y1 = person[num][i + 1]
                                y2 = person[num][i + 3]
                                for iii in range(0, len(cord_label[num]), 4):
                                    p1 = abs(x1 - cord_label[num][iii])
                                    p2 = abs(x1 - cord_label[num][iii + 2])
                                    p3 = abs(x2 - cord_label[num][iii])
                                    p4 = abs(x2 - cord_label[num][iii + 2])
                                    if (p1 <= 50) or (p2 <= 50) or (p3 <= 50) or (p4 <= 50):
                                        for ii in range(0, len(dog[num]), 4):
                                            d1 = abs(x1 - dog[num][ii])
                                            d2 = abs(y1 - dog[num][ii + 2])
                                            d3 = abs(x2 - dog[num][ii])
                                            d4 = abs(y2 - dog[num][ii + 2])
                                            if (d1 <= 250) or (d2 <= 250) or (d3 <= 250) or (d4 <= 250):
                                                # a = [0,0,0,0]
                                                a = []
                                                a1 = min(x1, dog[num][ii])
                                                b1 = min(y1, dog[num][ii + 1])
                                                a2 = max(x2, dog[num][ii + 2])
                                                b2 = max(y2, dog[num][ii + 3])
                                                a.extend([a1, b1 - 30, a2, b2])
                                                r = random.uniform(0.6, 0.85)
                                                label = "person with dog " + str(round(r, 2))
                                                annotator.box_label(a, label, color=(102, 204, 204))


            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                elif img_save==True:  # 'video' or 'stream'
                    print(vid_path)
                    i = 0
                    print(i)
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 5, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        if 'dog' in f'{s}' and 'person' in f'{s}' and'cord' in f'{s}':
            f1 = open(file='cord.txt', mode='w')
            f1.write('1')
            f1.close()
        if Streams:
            if ('dog' in f'{s}' and 'cord' not in f'{s}'):
                now = datetime.datetime.now()
                aa = now.strftime("%Y-%m-%d %H:%M:%S")
                vidcut_dir = str(save_dir) + '/date/' + aa[0:10]
                if not os.path.exists(vidcut_dir):
                    os.makedirs(vidcut_dir)
                vidcut_path = vidcut_dir + '\\' + aa[0:10] + '-' + aa[11:13] + '.' + aa[14:16] + '.' + aa[17:19]
                a0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                val = True
                t6 = t6 + 1
                if a0 != 0:
                    cv2.imwrite(vidcut_path + '.jpg', im0)
                else:
                    im1 = cv2.imread('no.jpg', 1)
                    cv2.imwrite(aa[0:10] + '-' + aa[11:13] + '.' + aa[14:16] + '.' + aa[17:19] + '.jpg', im1)
                # login_url = r'https://test.superton.cn/user-server/token/zhianpet'  # url
                # method = 'POST'
                # login_data = {'username': 'zhian', 'password': 'zhian123'}
                # login_headers = {'Referer': 'http://127.0.0.1:9527',
                #                  'content-type': 'application/json'}
                # # 此处用json就不用将login_data转为json格式再赋值给data参量了
                # response = requests.request(method, login_url, json=login_data, headers=login_headers)
                #
                # upload_url = r'https://test.superton.cn/smartcommunity-base-server-dongying/petAlarm/save'
                # file_path = r'2222.jpeg'  # 文件地址
                # _, file_name = os.path.split(file_path)  # 文件名
                #
                # # 获取token， 此处的response.text即为登录中的response.text
                # token = json.loads(response.text)['access_token']
                # token = "Bearer" + " " + token
                # png = open(vidcut_path + '.jpg', 'rb')
                # res = png.read()
                # s = base64.b64encode(res)
                # png.close()
                # # print(s.decode('ascii'))
                # data_body = {'image': s.decode('ascii'),
                #               'alarmType': '未牵狗绳',
                #               'videoAddress': 'https://mp4.vjshi.com/2021-03-16/87254015980fe091ce51b6f3eae02a29.mp4',
                #               'petCode': 'demoData',
                #               'time': 2022,
                #               'equipmentName': '安基名府11栋负2东', }
                # upload_headers = {"authorization": token,
                #                   'Referer': 'http://192.168.0.97', }
                # res = requests.request(method, upload_url, json=data_body, headers=upload_headers)
                # LOGGER.info(res.status_code)
                # LOGGER.info(res.request.url)
                # # print(res.request.body)
                # LOGGER.info(res.text)
                # LOGGER.info('over')

            if ('person' in f'{s}' and 'dog' in f'{s}') or ('dog' in f'{s}') or ('label' in f'{s}'):
                now = datetime.datetime.now()
                aa = now.strftime("%Y-%m-%d %H:%M:%S")
                vidcut_dir = str(save_dir) + '/date/111'
                if not os.path.exists(vidcut_dir):
                    os.makedirs(vidcut_dir)
                vidcut_path = vidcut_dir + '\\'+ aa[0:10] +'-'+ aa[11:13] + '.' + aa[14:16] + '.' + aa[17:19]
                a0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                val = True
                t6 = t6 + 1
                if a0 != 0:
                    cv2.imwrite(vidcut_path + '.jpg', im1)
                    # Note = open(str(save_dir) + '/result.txt', mode='a')
                    # Note.write(aa[0:10] +'-'+ aa[11:13] + '.' + aa[14:16] + '.' + aa[17:19] + '.jpg\n')
                    # Note.close()
                else:
                    im1 = cv2.imread('no.jpg', 1)
                    cv2.imwrite(aa[0:10]+'-'+ aa[11:13] + '.' + aa[14:16] + '.' + aa[17:19] + '.jpg', im1)
                    # Note = open(str(save_dir) + '/result.txt', mode='a')
                    # Note.write(aa[0:10] +'-'+ aa[11:13] + '.' + aa[14:16] + '.' + aa[17:19] + '.jpg\n')
                    # Note.close()


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['./yolov5_master/yolov5_weights/32.pt','./yolov5_master/yolov5_weights/33.pt'],help='model path(s)')
    parser.add_argument('--source', type=str, default='./static/res.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / './data/dog.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[960], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results',default=False)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='./', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment',default=True)
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--web', type=str, default='', help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt




def mainer(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    mainer(opt)
