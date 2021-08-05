# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import numpy
from pathlib import Path

from torchreid.utils import FeatureExtractor

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from collections import OrderedDict

from src.tracker import TrackableObject, CentroidTracker, load_coco

net, ln, LABELS, COLORS = load_coco()

root_path = Path("/home/antoine/projects/PeopleCounting")
input_path = root_path / "PeopleCounting/data/raw/test_airport_vid.mp4"
output_path = (
    root_path / "PeopleCounting/data/processed/test_airport_vid_processed.mp4"
)

args = {
    "input": input_path,
    "output": output_path,
    "confidence": 0.5,
    "threshold": 0.3,
}

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(maxDisappeared=5)
(H, W) = (None, None)
trackableObjects = {}

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(str(args["input"]))
writer = None
(W, H) = (None, None)

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# try to determine the total number of frames in the video file
try:
    prop = (
        cv2.cv.CV_CAP_PROP_FRAME_COUNT
        if imutils.is_cv2()
        else cv2.CAP_PROP_FRAME_COUNT
    )
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
rects_img = []
detection_frame_list = []
detection_frame_list = []
frame_number = 0
end_frame_decided = 10
while frame_number < end_frame_decided:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    frame_number = frame_number + 1
    print("frame number: {}".format(frame_number))

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    orig = frame.copy()
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes, rects = [], []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if (
                confidence > args["confidence"]
                and LABELS[classID] == "person"
            ):
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, args["confidence"], args["threshold"]
    )

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # test Antoine
            rects.append((x, y, x + w, y + h))
            rects_img.append(orig[y : y + h, x : x + w])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(
                frame,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    ratio_x_line1 = 500 / 1280
    ratio_x_line2 = 1100 / 1280
    W_line1 = int(ratio_x_line1 * W)
    W_line2 = int(ratio_x_line2 * W)
    cv2.line(frame, (W_line1, 0), (W_line1, H), (0, 255, 255), 2)
    cv2.line(frame, (W_line2, 0), (W_line2, H), (0, 255, 255), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects, orig)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the x-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is positive (indicating the object
                # is moving left) AND the centroid is between my 2 lines
                # the object is counted
                if direction > 0 and W_line1 < centroid[0] < W_line2:
                    totalUp += 1
                    to.counted = True
                    detection_frame_list.append(frame_number)

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(
            frame,
            text,
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Pax", totalUp),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(
            frame,
            text,
            (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
        )

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(args["output"]),
            fourcc,
            30,
            (frame.shape[1], frame.shape[0]),
            True,
        )

        # some information on processing single frame
        if total > 0:
            elap = end - start
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print(
                "[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total
                )
            )

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
