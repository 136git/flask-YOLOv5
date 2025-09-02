人犬牵绳检测系统

基于YOLOv5和Flask的智能人犬牵绳检测系统，能够自动识别图像或视频中的人与狗，并检测是否存在牵绳行为。

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/PyTorch-1.7%252B-red
https://img.shields.io/badge/Flask-2.0%252B-green
https://img.shields.io/badge/YOLOv5-ultralytics-orange
项目概述

本项目使用YOLOv5目标检测模型，结合Flask框架构建了一个Web应用程序，可以检测图像或视频中的人与狗是否使用牵绳。系统能够识别:

    人(person)

    狗(dog)

    牵绳(leash)

功能特点

    🖼️ 支持图像上传和实时检测

    🎥 支持视频流处理

    🌐 基于Web的友好界面

    ⚡ 高性能深度学习推理

    📊 可视化检测结果

环境要求

    Python 3.8+

    PyTorch 1.7+

    Flask 2.0+

    OpenCV

    其他依赖见requirements.txt
