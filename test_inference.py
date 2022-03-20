import tensorflow as tf

input = tf.keras.layers.Input([224, 224, 3], dtype = tf.float32)
layer = tf.keras.applications.resnet50.preprocess_input(input)
ResNet = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=(224, 224, 3), pooling=None, classes=1000)(layer)
MODEL = tf.keras.models.Model(inputs=[input], outputs=[ResNet])

def inference(image_path):
    """ResNet50 Pipeline. Preprocesses, runs model, post processes."""
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 244))
    image = tf.constant(tf.keras.utils.img_to_array(image))
    image = tf.expand_dims(image, axis=0)
    logits = MODEL.predict(image)
    decoded_logits = tf.keras.applications.resnet50.decode_predictions(
        preds=logits, top=1)
    class_name, class_description, score = decoded_logits
    return class_name, score

if __name__ == "__main__":
    out = inference("uploads/index-001.png")
    print(out)