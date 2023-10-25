import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
from PIL import Image, ImageDraw, ImageFont
import os
import argparse


recyclables = [0,1,4,5,6,7,8,11,12,13,14,15,17,22,23,24,25,26,31,34,35,36,37,38,40,41,42]
kitchen = [3,9,16,28,29,30,43]
other = [18,19,20,27,32,39]
hazardous = [2,10,21,33]

font_path = "SimHei.ttf"
font_name = "SimHei.ttf"



@st.cache_resource
def load_model():
    model = torch.hub.load('C:/Users/DELL/Desktop/1123456/yolov5-trash', 'custom', "runs/train/exp/weights/best.pt",  source='local')
    return model

def identify_garbage(garbage_id):
    if garbage_id in recyclables:
        return "可回收垃圾"
    elif garbage_id in kitchen:
        return "厨余垃圾"
    elif garbage_id in other:
        return "其他垃圾"
    elif garbage_id in hazardous:
        return "有害垃圾"
    else:
        return "未知垃圾种类"

def get_subdirs(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def draw_box_string(img, x_min, y_min, x_max, string):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("simhei.ttf", 25, encoding="utf-8")
    draw.rectangle([(x_min, y_min-25), (x_max, y_min)], fill=(0,255,0))
    draw.text((x_min, y_min - 25), string, (255, 255, 255), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img





parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str,
                    default='runs/train/exp/weights/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str,
                    default='', help='source')
parser.add_argument('--img-size', type=int, default=640,
                    help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float,
                    default=0.35, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float,
                    default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true',
                    help='display results')
parser.add_argument('--save-txt', action='store_true',
                    help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true',
                    help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true',
                    help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int,
                    help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true',
                    help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true',
                    help='augmented inference')
parser.add_argument('--update', action='store_true',
                    help='update all models')
parser.add_argument('--project', default='runs/detect',
                    help='save results to project/name')
parser.add_argument('--name', default='exp',
                    help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true',
                    help='existing project/name ok, do not increment')
opt = parser.parse_args()


app_mode = st.sidebar.selectbox('选择模式',
                                ['关于项目', '图片检测', '视频检测', '实时检测'])

if app_mode == '关于项目':
    st.markdown("<br />", unsafe_allow_html=True)
    st.subheader("前言")
    st.markdown(" 这是一个基于Yolov5实现的垃圾检测系统")
    st.markdown("- <h5>本项目使用的模型为yolov5s</h5>", unsafe_allow_html=True)
    st.image("ui/yolov5per.png")
    st.subheader("快速开始")
    st.markdown("- <h5>系统支持三种类型的检测：图片检测、视频检测和实时检测</h5>", unsafe_allow_html=True)
    st.image("ui/1.jpg")
    st.markdown("- <h5>图片检测</h5>", unsafe_allow_html=True)
    st.markdown("- 在左侧边栏中上传要检测的图片，等待一会儿后结果将展示在右侧")
    st.image("ui/2.jpg")
    st.markdown("- <h5>Live Detection</h5>", unsafe_allow_html=True)
    st.image("Images/fourth_4.png")
    st.markdown("- <h5>Click Start to start the camera</h5>", unsafe_allow_html=True)
    st.markdown("- <h5>Click Stop to stop the camera</h5>", unsafe_allow_html=True)

if app_mode == '图片检测':
    source_index = 0
    st.subheader("检测结果:")
    text = st.markdown("")
    st.sidebar.markdown("---")
    # Input for Image
    uploaded_file = st.sidebar.file_uploader("上传图片", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.sidebar.markdown("---")
        st.sidebar.markdown("**原始图片**")
        st.sidebar.image(uploaded_file)
    else:
        st.sidebar.markdown("请上传图片")
    # predict the image
    if st.button('开始检测'):
        if source_index == 0:
            model = load_model()
            results = model(image)
            length = len(results.xyxy[0])
            label_num = [int(x) for x in results.pred[0][:, -1].tolist()]
            labels = ''
            classify = ''
            st.markdown(f"<h1 style='text-align: center; color:black;'>检测到的数量为：<span style='color:red;'>{length}</span></h1>",
                        unsafe_allow_html=True)
            if len(label_num) == 1:
                labels = results.names[label_num[0]]
                classify = identify_garbage(label_num[0])
                st.markdown(f"<h5 style='text-align: center; color:black;'>图片中的垃圾为：{labels}</h5>"
                           f"<h5 style='text-align: center; color:black;'>属于：{classify}</h5>",
                            unsafe_allow_html=True)
            else:
                for a in label_num:
                    labels = labels+'  '+results.names[a]
                    classify = classify+'  '+identify_garbage(a)
                st.markdown(f"<h5 style='text-align: center; color:black;'>图片中的垃圾分别为：{labels}</h5>"
                           f"<h5 style='text-align: center; color:black;'>分别属于：{classify}</h5>",
                            unsafe_allow_html=True)
            for i in range(length):
                # 获取目标框坐标
                x_min, y_min, x_max, y_max = list(map(int, results.xyxy[0][i]))[:4]
                # 获取目标类别
                label_num = list(map(int, results.pred[0][i]))[-1]
                labels = results.names[label_num]
                # 获取目标置信度
                confidence = round(results.pred[0][i][-2].item(), 2)
                # 在图像上绘制目标框及类别信息
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                # cv2.putText(frame, f"{labels}: {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
                image = draw_box_string(image, x_min, y_min, x_max, f"{labels}: {confidence:.2f}")
            output = image
            st.subheader("输出图片")
            st.image(output, use_column_width=True)

if app_mode == '视频检测':
    st.subheader("检测结果:")
    text = st.markdown("")
    st.sidebar.markdown("---")
    stframe = st.empty()
    # Input for Video
    uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    if uploaded_file is not None:
        tffile.write(uploaded_file.read())
        vid = cv2.VideoCapture(tffile.name)
        st.sidebar.markdown("**载入视频**")
        st.sidebar.video(tffile.name)
    else:
        st.sidebar.markdown("请上传视频")
    # predict the videos

    if st.button('开始检测'):
        while vid.isOpened():
            ret, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])
            st.markdown(f"<h1 style='text-align: center; color:black;'>检测到的数量为：<span style='color:red;'>{length}</span></h1>",
                unsafe_allow_html=True)
            for i in range(length):
                # 获取目标框坐标
                x_min, y_min, x_max, y_max = list(map(int, results.xyxy[0][i]))[:4]
                # 获取目标类别
                label_num = list(map(int, results.pred[0][i]))[-1]
                labels = results.names[label_num]
                # 获取目标置信度
                confidence = round(results.pred[0][i][-2].item(), 2)
                # 在图像上绘制目标框及类别信息
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # cv2.putText(frame, f"{labels}: {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
                frame = draw_box_string(frame, x_min, y_min, x_max, f"{labels}: {confidence:.2f}")
            output = frame
            text.write(
                f"<h1 style='text-align: center; color:black;'>检测到的数量为：<span style='color:red;'>{length}</span></h1>",
                unsafe_allow_html=True)
            stframe.image(output)

if app_mode == '实时检测':
    st.subheader("检测结果:")
    text = st.markdown("")
    st.sidebar.markdown("---")
    stframe = st.empty()
    run = st.sidebar.button("开始")
    stop = st.sidebar.button("暂停")
    st.sidebar.markdown("---")
    cam = cv2.VideoCapture(0)
    if (run):
        while (True):
            if (stop):
                break
            ret, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])
            for i in range(length):
                # 获取目标框坐标
                x_min, y_min, x_max, y_max = list(map(int, results.xyxy[0][i]))[:4]
                # 获取目标类别
                label_num = list(map(int, results.pred[0][i]))[-1]
                labels = results.names[label_num]
                # 获取目标置信度
                confidence = round(results.pred[0][i][-2].item(), 2)
                # 在图像上绘制目标框及类别信息
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # cv2.putText(frame, f"{labels}: {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
                frame = draw_box_string(frame, x_min, y_min, x_max, f"{labels}: {confidence:.2f}")
            output = frame
            text.write(
                f"<h1 style='text-align: center; color:black;'>检测到的数量为：<span style='color:red;'>{length}</span></h1>",
                unsafe_allow_html=True)
            stframe.image(output)