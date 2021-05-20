from torch import no_grad, from_numpy, Tensor
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from PIL import Image
import numpy as np
from utils.datasets import letterbox

from io import BytesIO
from flask import Flask, request, jsonify


imgsz = 416
device  = select_device('cpu')

weight_file = input('Enter the path to Weights file such as best.pt')
detection_model = attempt_load(weight_file, map_location=device)
imgsz = check_img_size(imgsz, s=detection_model.stride.max())  # check img_size

app = Flask(__name__)  # Flask App


def detect(img_arr:np.ndarray, conf_thres:float = 0.5555, iou_thres:float = 0.5)->Tensor:
    '''
    Apply the Scaled Yolov4 model to an image and get detections
    args:
        image: Pillow Image converted to Numpy Array
        conf_thresh: Consider detections if and only if the score or the prediction is greater than this value
        iou_thresh: Threshold for Non Max Suppression
    returns:
        Torch.Tensor of rank 2 which includes No of Detections. Each detection has format as [x_min, x_min, x_max, y_max, conf, class].
        returns None if no detections are detected
    ''' 
    img = letterbox(img_arr, new_shape=416)[0]
    img = img.transpose(2, 0, 1)  #  to 3x416x416
    img = np.ascontiguousarray(img)
    
    img = from_numpy(img).to(device)
    img = img.float()  # uint8 to fp 32 . If CUDA is there, implement half but removing code here
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: # add a batch dimension
        img = img.unsqueeze(0)

    with no_grad():
        pred = detection_model(img)[0] # Predict from model

    pred = non_max_suppression(pred, conf_thres = conf_thres, iou_thres = iou_thres)  # Apply NMS
    
    det = pred[0] # Just 1 image in the batch
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_arr.shape).round() # Rescale boxes from img_size to img_arr size

    return det


@app.route('/predict', methods = ['POST'])
def extract_features():
    """
    Method to extract features
    args:
     data: {image} containing url of image whose features are to be extracted & input shape of mode
    returns:
     List of List for detected BB in the format [ [x_min, y_min ,x_max, y_max, class, conf_score],  ]
    """
    try:
        data = request.files["img"]
        image = Image.open(BytesIO(data.read()))
        
        if image.mode == 'RGBA' :
            image = image.convert('RGB')
        elif image.mode == 'L':
            return jsonify(["Grayscale Image Found. Needs RGB"]),400
        
        img_array = np.array(image)

    except Exception as e:
        return jsonify(["Corrupt Image or Image object not found in key 'img'"]),400
    
    try:
        detections = detect(img_array).numpy()
        return jsonify(detections = detections.tolist()),200 # Detections are in the format [ [x_min, y_min ,x_max, y_max, class, conf_score],  ]

    except Exception as e:
         return jsonify(["Detection Error"]),400
            
            
if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0')

