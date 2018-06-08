"""Demo for use yolo v3
"""
import os
import time
# import cv2
import numpy as np
from model.yolo_model import YOLO

from PIL import Image
import requests
from io import BytesIO


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_image_from_url(url):
    response = requests.get(url)
    output = BytesIO(response.content)
    output.seek(0)
    pil_image = Image.open(output).convert('RGB')
    open_cv_image = np.array(pil_image)
    image = open_cv_image[:, :, ::-1].copy()
    return image

def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def reformat_result(boxes, scores, classes, all_classes):
    outp = list()
    for box, score, cl in zip(boxes, scores, classes):

        x, y, w, h = box

        top = x
        left = y
        right = x+w
        bottom = y+h

        cls = all_classes[cl]

        # new_box = [top, left, right, bottom]
        # print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        # print('box coordinate top,left,right,bottom: {0}'.format(new_box))
        res = {"BoundingBox": {"Bottom": str(bottom), "Left": str(left), "Right": str(right), "Top": str(top)},
               "Class": cls, "Confidence": str(score)}
        outp.append(res)
    return outp


def detect_image_draw(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_image_raw(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict_raw(pimage)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        result = reformat_result(boxes, scores, classes, all_classes)
    #     draw(image, boxes, scores, classes, all_classes)

    return result


def detect_video(video, yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    camera = cv2.VideoCapture(video)
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image_draw(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    camera.release()


def demo_test(yolo, all_classes):
    # detect images in test floder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            for f in files:
                print(f)
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = detect_image_draw(image, yolo, all_classes)
                cv2.imwrite('images/res/' + f, image)


def test_one(yolo, all_classes, data):
    final_outp = []
    if data:
        url = data["url"]
        image = get_image_from_url(url)
        final_outp = detect_image_raw(image, yolo, all_classes)
    else:
        for (root, dirs, files) in os.walk('images/test'):
            if files:
                for f in files:
                    print(f)
                    path = os.path.join(root, f)
                    image = cv2.imread(path)
                    final_outp = detect_image_raw(image, yolo, all_classes)
    return final_outp


def test_list(yolo, all_classes, list_data):
    final_outp = []
    if list_data:
        for data in list_data:
            url = data["url"]
            image = get_image_from_url(url)
            single_outp = detect_image_raw(image, yolo, all_classes)   # single_outp is a list
            return_data = {"url": url, "detections": single_outp}
            final_outp.append(return_data)
    else:
        for (root, dirs, files) in os.walk('images/test'):
            if files:
                for f in files:
                    print(f)
                    path = os.path.join(root, f)
                    image = cv2.imread(path)
                    single_outp = detect_image_raw(image, yolo, all_classes)
                    return_data = {"url": f, "detections": single_outp}
                    final_outp.append(return_data)
    return final_outp


def setup():
    # global yolo, all_classes
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)
    return yolo, all_classes


def main_one(data):
    yolo, all_classes = setup()

    # demo_test
    # demo_test(yolo, all_classes)

    # detect video.
    # video = 'E:/video/car.flv'
    # detect_video(video, yolo, all_classes)

    # test_one
    outp = test_one(yolo, all_classes, data)
    return outp


def main_list(list_data):
    yolo, all_classes = setup()

    # demo_test
    # demo_test(yolo, all_classes)

    # detect video.
    # video = 'E:/video/car.flv'
    # detect_video(video, yolo, all_classes)

    # test_one
    outp = test_one(yolo, all_classes, list_data)
    return outp


if __name__ == '__main__':
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)

    # demo_test
    demo_test(yolo, all_classes)

    # detect video.
    # video = 'E:/video/car.flv'
    # detect_video(video, yolo, all_classes)

    # test_one
    # outp = test_one(yolo, all_classes, None)
