from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import subprocess
import logging
import shutil
import cv2
from ultralytics import YOLO  # 正确导入 YOLO 模块

app = Flask(__name__, static_folder='dist')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载YOLO模型
model = YOLO('/Volumes/XiaoMI_1TB/软件杯作品/17011727源码/runs/segment/train/weights/best.pt')

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def serve_vue_app():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'code': 40000, 'status': 400, 'message': '没有文件部分', 'data': {}}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 40001, 'status': 400, 'message': '未选择文件', 'data': {}}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"文件保存到 {filepath}")

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_filepath = process_image(filepath)
            result_key = 'result_image'
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            processed_filepath = process_video(filepath)
            result_key = 'result_video'
        else:
            return jsonify({'code': 40002, 'status': 400, 'message': '不支持的文件类型', 'data': {}}), 400

        logger.info(f"处理后的文件路径: {processed_filepath}")
        if not processed_filepath or not os.path.exists(processed_filepath):
            return jsonify({'code': 50001, 'status': 500, 'message': '未找到处理后的文件',
                            'data': {'path': processed_filepath}}), 500

        return jsonify(
            {'code': 20000, 'status': 200, 'message': '上传成功',
             'data': {result_key: f'/processed/{os.path.basename(processed_filepath)}'}}
        ), 200

@app.route('/processed/<filename>')
def get_processed_file(filename):
    """提供处理后的文件的端点。"""
    return send_from_directory(PROCESSED_FOLDER, filename)

def process_image(filepath):
    """处理上传的图像，直接返回模型处理后的图像。"""
    results = model.predict(filepath, save=True, save_dir='runs/segment',
                            show_boxes=False,)
    for root, _, files in os.walk('runs/segment'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
                shutil.move(src_path, dst_path)
                logger.info(f"图像处理成功，保存到 {dst_path}")
                return dst_path
    logger.error(f"图像处理失败，未找到保存的文件在 {PROCESSED_FOLDER}")
    return filepath

def process_video(filepath):
    """处理上传的图像，直接返回模型处理后的图像。"""
    results = model.predict(filepath, save=True, save_dir='processed',
                            show_boxes=False,)
    for root, _, files in os.walk('runs/segment'):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')) and not file.startswith('.'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
                shutil.move(src_path, dst_path)
                logger.info(f"图像处理成功，保存到 {dst_path}")
                return dst_path
    logger.error(f"图像处理失败，未找到保存的文件在 {PROCESSED_FOLDER}")
    return filepath

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)