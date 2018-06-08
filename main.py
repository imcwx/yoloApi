from flask import Flask, request, jsonify
import demo
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

app = Flask(__name__)


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
    #             "Confidence": "0.9803"
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


@app.route('/ObjectDetection_list', methods=['POST'])
def yolo_detection_list():
    list_data = request.get_json()
    if request.method == "POST":
        if list_data[0]:
            print(list_data)
            # data = dict()
            # data["url"] = "https://upload.wikimedia.org/wikipedia/commons/0/09/Shibuya_Station_Crossing.JPG"
            # list_data = [data , data]
            outp = demo.test_list(yolo, all_classes, list_data)

    return jsonify(outp)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    yolo, all_classes = demo.setup()
    graph = tf.get_default_graph()
    app.run(debug=True, host='0.0.0.0',port=8888)
