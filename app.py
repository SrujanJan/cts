from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
import pytesseract
import cv2
import json
import numpy as np

app = Flask(__name__)

# Initialize Roboflow
rf = Roboflow(api_key="Tr2vfRBoqop4xw7uvCAf")
project = rf.workspace("cheque-automation").project("cheques_segmentation")
model = project.version(2).model

def crop_image(image, x, y, width, height,save_path):
    # if isinstance(image, str):
    #     # Load the image if a file path is provided
    #     image = cv2.imread(image)
    x1 = int(x - (width/2))
    y1 = int(y - (height/2))
    x2 = int(x + (width/2))
    y2 = int(y + (height/2))
    cropped_image = image[y1:y2, x1:x2]
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # save_path=r"C:/Users/sruja/OneDrive/Desktop/ALL FILES/Flask/grayimg"
    if save_path:
        cv2.imwrite(save_path, grayscale_image)
    return grayscale_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    file = request.files['file']
    l=[]
    # Save the file locally
    file_path = "C:/Users/sruja/OneDrive/Desktop/ALL FILES/Flask/Images/" + file.filename
    file.save(file_path)

    # Perform prediction
    json_object = model.predict(file_path, confidence=10, overlap=50).json()
    json_object["predictions"] = sorted(json_object["predictions"], key=lambda x: x["class_id"])
# 
    # print(json_object)
    #save to predictions.json
    with open("predictions.json", "w") as f:
        json.dump(json_object, f)
    # Further processing as needed
    # ...
    # Get the x, y, width, and height coordinates of the MICR code
    micr_prediction = json_object['predictions'][1]
    micr_x = micr_prediction['x']
    micr_y = micr_prediction['y']
    micr_width = micr_prediction['width']
    micr_height = micr_prediction['height']
    # return jsonify(json_object)
    micr_img = crop_image(cv2.imread(file_path), micr_x, micr_y, micr_width, micr_height,save_path="grayimg/micr_grayscale.jpg")
    f=pytesseract.image_to_string(micr_img,  lang = 'mcr')
    print(f)
    l.append(f)
    # Get the x, y, width, and height coordinates of the ACCOUNT NUMBER code
    acc_prediction = json_object['predictions'][6]
    acc_x = acc_prediction['x']
    acc_y = acc_prediction['y']
    acc_width = acc_prediction['width']
    acc_height = acc_prediction['height']
    # # Crop the ACCOUNT NUMBER code from the image
    account_number=crop_image(cv2.imread(file_path),acc_x,acc_y,acc_width,acc_height,save_path="grayimg/acc_grayscale.jpg")
    acc=pytesseract.image_to_string(account_number,  lang = 'eng')
    l.append(acc)
    print(pytesseract.image_to_string(account_number,  lang = 'eng'))
    
    # Get the x, y, width, and height coordinates of the IFSC  code
    IFSC_prediction = json_object['predictions'][7]
    IFSC_x = IFSC_prediction['x']
    IFSC_y = IFSC_prediction['y']
    IFSC_width = IFSC_prediction['width']
    IFSC_height = IFSC_prediction['height']
    # # Crop the IFSC  code code from the image
    
    IFSC_number=crop_image(cv2.imread(file_path),IFSC_x,IFSC_y,IFSC_width,IFSC_height,save_path="grayimg/ifsc_grayscale.jpg")
    ifsc=pytesseract.image_to_string(IFSC_number,  lang = 'eng')
    print(ifsc)
    l.append(ifsc)
    
    # Get the x, y, width, and height coordinates of the payee name
    PAYEE_prediction = json_object['predictions'][3]
    PAYEE_x = PAYEE_prediction['x']
    PAYEE_y = PAYEE_prediction['y']
    PAYEE_width = PAYEE_prediction['width']
    PAYEE_height = PAYEE_prediction['height']

    # # Crop the payee from the image
    PAYEE_number=crop_image(cv2.imread(file_path),PAYEE_x,PAYEE_y,PAYEE_width,PAYEE_height,save_path="grayimg/payee_grayscale.jpg")

    # # Print the shape of the cropped payee NUMBER image
    pnum=pytesseract.image_to_string(PAYEE_number,  lang = 'eng')
    print(pnum)
    l.append(pnum)
    return l

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port="5000")
