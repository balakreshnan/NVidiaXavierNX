import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt
import onnx
import time

print(" Onnx Runtime : " + rt.get_device())

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]



model_path = "models/ssd_mobilenet_v1_12.onnx"
#model_path = "models/tiny-yolov3-11.onnx"


import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt

coco_classes = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
}

def draw_detection(draw, d, c):
    """Draw box and label for 1 detection."""
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    draw.text(text_origin, label, fill=color)  # , font=font)



import cv2
#sess = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
sess = rt.InferenceSession(str(model_path), providers=providers)
# vid = cv2.VideoCapture(0) # For webcam
print(sess.get_inputs()[0].name)
vid = cv2.VideoCapture("rtsp://office:admin1234@192.168.4.54:88/videoMain") # For streaming links
while True:
    rdy,frame = vid.read()
    #print(rdy)
    start = time.process_time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame, (640, 480))
    # image = Image.fromarray(img)
    image = Image.fromarray(img, 'RGB')
    img_data = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    print(img_data.shape)
    #sess = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    #sess = rt.InferenceSession(model_path)
    # we want the outputs in this order
    # outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]
    outputs = []
    result = sess.run(outputs, {"inputs": img_data})
    #print(result)
    detection_boxes, detection_classes, detection_scores, num_detections = result
    # print number of detections
    print(num_detections[0])
    print(detection_classes[0])
    #print(detection_boxes[0])

    #cv2.imshow('Video Live IP cam',frame)
    batch_size = num_detections.shape[0]
    draw = ImageDraw.Draw(Image.fromarray(img, 'RGB'))
    #for batch in range(0, batch_size):
    #    for detection in range(0, int(num_detections[batch])):
    #        c = detection_classes[batch][detection]
    #        d = detection_boxes[batch][detection]
    #        draw_detection(draw, d, c)

    #plt.figure(figsize=(80, 40))
    #plt.axis('off')
    #plt.imshow(img)
    #plt.show()
    cv2.imshow("output", img)
    print(" Time taken = " + str(time.process_time() - start))
    
    key = cv2.waitKey(1) & 0xFF
    if key ==ord('q'):
        break
 
#    try:
#    except:
#       pass

vid.release()
cv2.destroyAllWindows()

