from flask import Flask, request,render_template
import os
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
class_names=['glioma', 'meningioma', 'notumor', 'pituitary']
model = tf.keras.models.load_model('brain_tumor1.h5')
@app.route('/')
def index():
    return render_template('index.html')


def read_file_as_image(file):
   filename = os.path.join(file.filename)
   file.save(filename)
   image = cv2.imread(filename)
   image = cv2.resize(image, (150, 150))
   image = np.expand_dims(image, -1)
   return image


@app.route('/upload', methods=['POST'])
def predition():
    file = request.files['file']
    img = read_file_as_image(file)
    image_batch = np.expand_dims(img,0)
    pred=  model.predict(image_batch)
    pred_class =  class_names[np.argmax(pred[0])]
    confidene =   round(100 * np.max(pred[0]),2)

    return {
        'class:':pred_class,
        'confidence:':confidene
    }


# if __name__ == '__main__':
#     app.run(host='192.168.0.108',port=8000, debug=True)