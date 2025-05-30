from utils.cascade import Classifier
from utils.yolo_video import Detector
from utils.yolo_image import Detector as Image_Detector
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Object detection")
    parser.add_argument("--model", type=str, choices=['cascade', 'yolov8', 'yolov11'],
                        default='cascade', help="Model type")
    parser.add_argument("--filepath", type=str)
    parser.add_argument('--confidence', type=float, default=0.02, help="Confidence threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.model == 'cascade':
        my_classifier = Classifier(args.filepath)
        my_classifier.plot()
    elif args.model == 'yolov11':
        my_detector = Detector(args.filepath)
        my_detector.forward()
    elif args.model == 'yolov8':
        my_detector = Image_Detector(args.filepath, args.confidence)
        my_detector.plot()


if __name__ == '__main__':
    main()
