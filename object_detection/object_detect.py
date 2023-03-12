import sys
sys.path.insert(0, '/home/pallav/yolo/object_detection_v1')

### import required libraries
import cv2
import time
import numpy as np
import pandas as pd

class objectDetection():
    def __init__(self, objdetectconfig_str):
        self.objdetectconfig_str = objdetectconfig_str
        ### read classnames from coco.names file
        self.labels = open(self.objdetectconfig_str.coco_data_path).read().strip().split('\n')
        # print(f"Labels: {self.labels}\nTotal no of labels: {len(self.labels)}")
        ### load the object detector
        self.net = cv2.dnn.readNetFromDarknet(self.objdetectconfig_str.yolo_cfg_path,
                                              self.objdetectconfig_str.yolo_wts_path)

    def get_data(self):
        data = pd.read_csv('/home/pallav/yolo/backend_content_details.csv', low_memory = False)
        data = data[['_id', 'contentURL']]
        # data = data.drop_duplicates()
        # data = data.dropna()
        # print(f"Shape of data: {data.shape}")
        # print(f"Columns in data: {data.columns}")
        return data

    def detect_objects(self, video_url):
        st = time.time()
        results = {}
        ### extract yolo-layers from net
        ln = self.net.getLayerNames()
        yolo_layers = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # print(f"yolo layers: {yolo_layers}")

        (W, H) = (None, None)
        cap = cv2.VideoCapture(video_url)
        # try:
        #     cap = cv2.VideoCapture(video_url)
        # except Exception as e:
        #     print(e)
        #     exit()

        ### Find the total number of frames in the video file
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f"Total frames in video: {total_frame}")

        fps = cap.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
        multiplier = fps * self.objdetectconfig_str.sec
        frame_counter = 1

        while frame_counter <= total_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            (grabbed, frame) = cap.read()
            # print(f"frame number: {int(frame_counter)}")
            frame_counter += multiplier
            ### if the frame was not grabbed, then we have reached the end of the stream
            if not grabbed:
                break
            ### if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            
            ### YOLO object detector - obtain bounding boxes and probabilities
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layerOutputs = self.net.forward(yolo_layers)

            ### initialize lists to save detected bounding boxes, confidences, and class ID
            boxes = []
            confidences = []
            classIDs = []

            ### loop over each of the layer outputs
            for output in layerOutputs:
                ### loop over each of the detections
                for detection in output:
                    ### extract the class ID and confidence (i.e., probability) of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > self.objdetectconfig_str.detectionConfThreshold:
                        ### YOLO model returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        ### use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        ### update our list of bounding box coordinates, confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 
                                    self.objdetectconfig_str.detectionConfThreshold, 
                                    self.objdetectconfig_str.suppressionThreshold)

            #  ensure at least one detection exists
            if len(idxs) > 0:
                for i in idxs.flatten():
                    label = self.labels[classIDs[i]]
                    conf = round(confidences[i], 3)
                    ### update labels and confidence in dictionary
                    if label in results.keys():
                        results[label] = max(results[label], conf)
                    else:
                        results[label] = conf
                    # print(f"{label} -> {conf}")
        ed = time.time()
        print(f"Prediction time on video: {ed - st} sec")
        return results

    def get_master_labels(self, results):
        master_list = []
        for label in results.keys():
            for i in self.objdetectconfig_str.master_label_dict.keys():
                if label in self.objdetectconfig_str.master_label_dict[i]:
                    master_list.append(i)
        return list(set(master_list))

    def object_detection(self, input_data):
        output_data = pd.DataFrame(columns = ['_id', 'master_labels', 'detection_results'])

        if input_data.size == 0:
            print('No data found')
            exit()
        else:
            for count, row in input_data.iterrows():
                if count == 6000:
                    break
                video_url = row['contentURL']
                print(f"\n{count+1}.  {video_url}")
                results = self.detect_objects(video_url)
                master_list = self.get_master_labels(results)
                temp_dict = {'_id':row['_id'], 'master_labels' : master_list, 'detection_results' : results}
                print(f"results: {results}\nmaster_list: {master_list}")
                output_data = pd.concat([output_data, 
                                         pd.DataFrame([temp_dict])], 
                                         ignore_index=True)
        return output_data