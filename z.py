from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
import pytesseract
import cv2
import json
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize Roboflow
rf = Roboflow(api_key="Tr2vfRBoqop4xw7uvCAf")
project = rf.workspace("cheque-automation").project("cheques_segmentation")
model = project.version(2).model

def crop_image(image, x, y, width, height, save_path):
    x1 = int(x - (width/2))
    y1 = int(y - (height/2))
    x2 = int(x + (width/2))
    y2 = int(y + (height/2))
    cropped_image = image[y1:y2, x1:x2]
    # sharpening_filter = np.array([[-1, -1, -1],
    #                               [-1, 9, -1],
    #                               [-1, -1, -1]])
    # sharp_image = cv2.filter2D(cropped_image, -1, sharpening_filter)

    # # Convert to black and white
    # grayscale_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((1, 1), np.uint8)
    # cropped_image = cv2.dilate(cropped_image, kernel, iterations=1)
    # cropped_image = cv2.erode(cropped_image, kernel, iterations=1)
    # cv2.threshold(cv2.GaussianBlur(cropped_image, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.threshold(cv2.bilateralFilter(cropped_image, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.threshold(cv2.medianBlur(cropped_image, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.adaptiveThreshold(cv2.GaussianBlur(cropped_image, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # cv2.adaptiveThreshold(cv2.bilateralFilter(cropped_image, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # cv2.adaptiveThreshold(cv2.medianBlur(cropped_image, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # Ensure the folder "grayimg" exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cropped_image)

    return cropped_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    file = request.files['file']
    result_list = []
     # Generate a new filename with the desired prefix and original extension
    original_filename, original_extension = os.path.splitext(secure_filename(file.filename))
    new_filename = "custom_prefix" + original_extension
    # Save the file locally
    file_path = "C:/Users/sruja/OneDrive/Desktop/ALL FILES/Flask/Images/" + new_filename
    file.save(file_path)
    image = cv2.imread(file_path)

    # Grayscale operation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian filtering
    sigma = 1  # Adjust the value of sigma based on your requirement
    gaussian_filtered = cv2.GaussianBlur(gray_image, (0, 0), sigma)

    # Binary image conversion
    _, binary_image = cv2.threshold(gaussian_filtered, 136, 255, cv2.THRESH_BINARY)

    # Create the 'grayimg' folder if it doesn't exist
    output_folder = 'grayimg'
    os.makedirs(output_folder, exist_ok=True)

    # Save the results in the 'grayimg' folder
    cv2.imwrite(os.path.join(output_folder, 'gray_image.jpg'), gray_image)
    cv2.imwrite(os.path.join(output_folder, 'gaussian_filtered_image.jpg'), gaussian_filtered)
    cv2.imwrite(os.path.join(output_folder, 'binary_image.jpg'), binary_image)
    # Perform prediction
    file_binary_img='C:/Users/sruja/OneDrive/Desktop/ALL FILES/Flask/grayimg/binary_image.jpg'
    json_object = model.predict(file_binary_img, confidence=10, overlap=50).json()
    json_object["predictions"] = sorted(json_object["predictions"], key=lambda x: x["class_id"])

    # Save predictions to predictions.json
    with open("predictions.json", "w") as f:
        json.dump(json_object, f)

    # Example usage for different classes
    classes_of_interest = ["micr", "ifsc", "a_c", "b_name"]

    for class_name in classes_of_interest:
        prediction = next((p for p in json_object['predictions'] if p['class'] == class_name), None)
        if prediction:
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            confidence = prediction['confidence']

            print(f"Class: {class_name}")
            print(f"Bounding Box: (x={x}, y={y}, width={width}, height={height})")
            print(f"Confidence: {confidence}")

            # Example of saving the grayscale image
            save_path = f"grayimg/{class_name}_grayscale.jpg"
            cropped_gray_image = crop_image(cv2.imread(file_path), x, y, width, height, save_path=save_path)
            class_name_att=class_name
            if class_name_att!="micr":
                class_name_att=pytesseract.image_to_string(cropped_gray_image,  lang = 'eng')
                result_list.append(class_name_att)
            else:
                class_name_att=pytesseract.image_to_string(cropped_gray_image,  lang = 'mcr')
                result_list.append(class_name_att)
        else:
            print(f"Class '{class_name}' not found in predictions.")

    return jsonify(result_list)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")
