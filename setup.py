"""
Minimal setup.py for YOLOv8 Cigarette Detection Model
Allows installation via: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name='yolov8-cigarette-detector',
    version='1.0.0',
    description='YOLOv8s cigarette butt detection model',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-repo/yanto',
    python_requires='>=3.8',
    install_requires=[
        'ultralytics>=8.0.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy',
        'opencv-python',
        'tqdm',
    ],
    py_modules=['infer_yolo', 'train_yolo'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
