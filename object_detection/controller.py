import sys
sys.path.insert(0, '/home/pallav/yolo/object_detection_v1')

import time
from utilities.config import get_objDetectConfig
from object_detection.object_detect import objectDetection

if __name__ == '__main__':
    st = time.time()
    objdetectconfig_str = get_objDetectConfig()
    object_detect = objectDetection(objdetectconfig_str)
    input_data = object_detect.get_data()
    output_data = object_detect.object_detection(input_data)
    output_data.to_csv('/home/pallav/yolo/object_detection_v1/result.csv')
    ed = time.time()
    print(f"\nTotal time required: {ed - st} sec")