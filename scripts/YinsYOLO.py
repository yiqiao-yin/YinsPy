class YinsYOLO:

    print("-----------------------------------------------------")
    print(
    """
    Yin's YOLO Package (For Quick Deployment)
    Copyright © YINS CAPITAL, 2009 – Present
    For more information, go to www.YinsCapital.com
    """ )
    print("-----------------------------------------------------")
    
    def Personal_AI_Surveillance(whatObject = 'knife', useAdvancedYOLO = True, confidence=0.1, whichCam = 0, verbose=False):
        
        """
        # READ:
        # Object detection webcam example using tiny yolo
        # Usage: python object_detection_webcam_yolov3_tiny.py
        """

        # Import necessary packages
        import cvlib as cv
        from cvlib.object_detection import draw_bbox
        import cv2

        # Check out laptop cam:
        # 0 is the first camera (on laptop) 
        # 1 is the second camera (ex. I have a usb cam connected to the laptop that is higher resolution)
        # and then you can do 2, 3, ... if you have installed more cameras.
        # webcam = cv2.VideoCapture(1)
        # print(f'Camera resolution is {webcam.get(3)} by {webcam.get(4)}.')

        # Setup *alert()* Function

        import time
        from IPython.core.magics.execution import _format_time
        from IPython.display import display as d
        from IPython.display import Audio
        from IPython.core.display import HTML
        import numpy as np
        import logging as log

        def alert():
            """ makes sound on client using javascript (works with remote server) """      
            framerate = 44100
            duration  = 0.1
            freq      = 300
            t         = np.linspace(0, duration, framerate*duration)
            data      = np.sin(2*np.pi*freq*t)
            d(Audio(data, rate=framerate, autoplay=True))

        # The following code will start a new window with live camera feed from your laptop. 
        # The notebook will print out a long list of results, with objects detected or not. 
        # To shut it down, make sure current window is in the camera feed and press 'q'. 

        # Open Camera
        webcam = cv2.VideoCapture(whichCam)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # Loop through frames
        while webcam.isOpened():

            # Read frame from webcam 
            status, frame = webcam.read()
            if not status:
                break

            # Apply object detection
            # 80 common objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
            if useAdvancedYOLO:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3') # this is very slow
            else:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')
            
            # Print Comment
            if verbose:
                print(bbox)
                print(label)
                print(conf)
            
            # Output
            # print(bbox, label, conf)
            # Set Alert (if see a knife)
            tmp = label
            for i in tmp:
                if i == whatObject:
                    alert()

            # Draw bounding box over detected objects
            # We take output from *cv.detect_common_objects* to print them out on videos
            # by using *draw_bbox()*
            out = draw_bbox(frame, bbox, label, conf, write_conf=True)

            # Display output
            cv2.imshow("Yin's AI Surveillance", out)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End of function
    
        
    def Personal_AI_Surveillance_Basic_Grouping(whatObject = ['cell phone', 'person'], useAdvancedYOLO = True, confidence=0.1, whichCam = 0, useAlert=False, verbose=False):
        
        """
        # READ:
        # Object detection webcam example using tiny yolo
        # Usage: python object_detection_webcam_yolov3_tiny.py
        """

        # Import necessary packages
        import cvlib as cv
        from cvlib.object_detection import draw_bbox
        import cv2

        # Check out laptop cam:
        # 0 is the first camera (on laptop) 
        # 1 is the second camera (ex. I have a usb cam connected to the laptop that is higher resolution)
        # and then you can do 2, 3, ... if you have installed more cameras.
        # webcam = cv2.VideoCapture(1)
        # print(f'Camera resolution is {webcam.get(3)} by {webcam.get(4)}.')

        # SUPPORT FUNCTIONS:
        # Setup *alert()* Function
        import time
        from IPython.core.magics.execution import _format_time
        from IPython.display import display as d
        from IPython.display import Audio
        from IPython.core.display import HTML
        import numpy as np
        import logging as log

        def alert():
            """ makes sound on client using javascript (works with remote server) """      
            framerate = 44100
            duration  = 1
            freq      = 300
            t         = np.linspace(0, duration, framerate*duration)
            data      = np.sin(2*np.pi*freq*t)
            d(Audio(data, rate=framerate, autoplay=True))

        # The following code will start a new window with live camera feed from your laptop. 
        # The notebook will print out a long list of results, with objects detected or not. 
        # To shut it down, make sure current window is in the camera feed and press 'q'. 
        
        def drawBox(img, bbox, labels, confidence, colors=None, write_conf=False):
            classes = None
            COLORS = np.random.uniform(0, 255, size=(80, 3))

            if classes is None:
                classes = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

            for i, label in enumerate(labels):
                if colors is None:
                    color = COLORS[classes.index(label)]
                else:
                    color = colors[classes.index(label)]
                if write_conf:
                    label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
                cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)
                if (len(labels) - 1 > 2) & (i < len(labels) - 1):
                    start_point = (round((bbox[i][0]+bbox[i][1])/2), round((bbox[i][2]+bbox[i][3])/2))
                    end_point   = (round((bbox[i+1][0]+bbox[i+1][1])/2), round((bbox[i+1][2]+bbox[i+1][3])/2))
                else:
                    start_point = (0,0)
                    end_point = (0,0)
                cv2.line(img, start_point, end_point, (0,255,0), 1)
                # print(">>>>>>", start_point, end_point)
                cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return img

        # Open Camera
        webcam = cv2.VideoCapture(whichCam)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        # Loop through frames
        while webcam.isOpened():

            # Read frame from webcam 
            status, frame = webcam.read()
            if not status:
                break

            # Apply object detection
            # 80 common objects: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
            if useAdvancedYOLO:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3') # this is very slow
            else:
                bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')
            
            # Print Comment
            if verbose:
                print(bbox)
                print(label)
                print(conf)
            
            # Output
            # print(bbox, label, conf)
            # Set Alert (if see a knife)
            tmp = label
            for i in tmp:
                if i == whatObject:
                    if useAlert:
                        alert()

            # Draw bounding box over detected objects
            # We take output from *cv.detect_common_objects* to print them out on videos
            # by using *draw_bbox()*
            # sample: drawBox(img, bbox, labels, confidence, colors=None, write_conf=False)
            out = drawBox(img=frame, bbox=bbox, labels=label, confidence=conf, write_conf=True)

            # Display output
            cv2.imshow("Yin's AI Surveillance", out)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End of function