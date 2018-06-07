from flask import Flask, request, jsonify
import demo
from model.yolo_model import YOLO
import tensorflow as tf

app = Flask(__name__)
# yolo, all_classes = None, None


@app.route('/')
def hello_world():
    return 'hello world!'


@app.route('/ObjectDetection', methods=['POST'])
def yolo_detection_single():
    # outp = [{
    #             "BoundingBox": {
    #                 "Bottom": 0.6719,
    #                 "Left": 0.9425,
    #                 "Right": 1,
    #                 "Top": 0.5979
    #             },
    #             "Class": "car",
    #             "Confidence": "0.9803",
    #             "ObjectKey": "None/[0.9425,0.5979,1.0000,0.6719]"
    #         },
    #         {
    #             "BoundingBox": {
    #                 "Bottom": 0.6719,
    #                 "Left": 0.9425,
    #                 "Right": 1,
    #                 "Top": 0.5979
    #             },
    #             "Class": "car",
    #             "Confidence": "0.9803",
    #             "ObjectKey": "None/[0.9425,0.5979,1.0000,0.6719]"
    #         }
    #         ]
    data = request.get_json()
    if request.method == "POST":
        if data["url"]:
            print(data)
            # print(data["url"])
            # print("received something ...")
            # data = dict()
            # data["url"] = "https://upload.wikimedia.org/wikipedia/commons/0/09/Shibuya_Station_Crossing.JPG"
            # outp = demo.test_one(yolo, all_classes, data)
            # outp = demo.main_one(data)
            outp = demo.test_one(yolo, all_classes, data)

    return jsonify(outp)


# @app.route('/ObjectDetection_list', methods=['POST'])
# def yolo_detection_list():
#     data = request.get_json()
#     print(data)
#
#     data = dict()
#     # data["url"] = "http://cyprusexcursion.com/wp-content/uploads/2016/04/trip-sharing-cyprus.jpg"
#     data["url"] = "https://upload.wikimedia.org/wikipedia/commons/0/09/Shibuya_Station_Crossing.JPG"
#
#     list_data = [data, data]
#     print("received something ...")
#     outp = demo.main_list(list_data)
#
#     return jsonify(outp)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    global yolo
    global all_classes
    global graph
    yolo, all_classes = demo.setup()
    graph = tf.get_default_graph()
    app.run(debug=True, port=8888)
