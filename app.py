import os
import tensorflow as tf

from datetime import datetime
from flask import Flask, Blueprint, render_template, jsonify, request

application = Flask(__name__)
home_page = Blueprint("index",
                      import_name=__name__,
                      template_folder="templates")

input = tf.keras.layers.Input([224, 224, 3], dtype = tf.float32)
ResNet = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=(224, 224, 3), pooling=None, classes=1000)(input)
MODEL = tf.keras.models.Model(inputs=[input], output=[ResNet])


@home_page.route("/")
def home():
    return render_template("index.html")


@home_page.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return "An Error Occured"

        input_file = request.files["file"]

        input_file_path = os.path.join(
            "uploads", str(input_file.filename))
        input_file.save(input_file_path)
        class_name, score = inference(input_file_path)
        return jsonify({"status": "Success!",
                        "prediction": class_name,
                        "confidence": score,
                        "upload_time": datetime.now()})


def inference(image_path):
    """ResNet50 Pipeline. Preprocesses, runs model, post processes."""
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224))
    image = tf.constant(tf.keras.utils.img_to_array(image))
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    logits = MODEL.predict(image)
    decoded_logits = tf.keras.applications.resnet50.decode_predictions(
        preds=logits, top=1)
    class_name, class_description, score = decoded_logits
    return class_name, score

if __name__ == "__main__":
    application.register_blueprint(home_page)
    application.run(debug=True, host="0.0.0.0", port=5000)
