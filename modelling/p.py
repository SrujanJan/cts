from flask import Flask, render_template, request
from tensorflow import keras
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('signature.h5')

def preprocess_image(image_path):
    # Implement any necessary preprocessing for the uploaded image
    # For example, resizing, converting to grayscale, etc.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (650, 268))  # Swap dimensions to match the model input shape
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded file (optional)
            file_path = 'uploads/' + file.filename
            file.save(file_path)

            # Preprocess the image
            img = preprocess_image(file_path)

            # Make a prediction
            prediction = model.predict(img)
            result = 'Real' if prediction[0][0] > 0.5 else 'Fake'

            return render_template('result.html', result=result, file_path=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
