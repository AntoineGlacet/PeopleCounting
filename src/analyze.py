# analyze.py
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import cv2
from pathlib import Path
from tqdm import tqdm

from src.tracker import TrackableObject


def track_distance_and_features(
    input_path: Path,
    output_path: Path,
    ct,
    net,
    ln,
    LABELS,
    COLORS,
    end_frame_decided=300,
    write=True,
    confidence_set: float = 0.5,
    threshold_set: float = 0.3,
    ratio_x_line1: float = 1 / 1280,
    ratio_x_line2: float = 350 / 1280,
    counting_direction=1,  # 1 for left to right and -1 for right to left
):
    # intialize centroid tracker and frame dimensions
    (H, W) = (None, None)
    trackableObjects = {}
    # initialize the video stream, pointer to output_path video file, and
    # frame dimensions
    vs = cv2.VideoCapture(str(input_path))
    writer = None
    (W, H) = (None, None)

    # initialize the total number of people
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
    detection_frame_list = []
    frame_number = 0
    with tqdm(
        total=end_frame_decided, desc="video analysis in progress..."
    ) as runpbar:
        while frame_number < end_frame_decided:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            frame_number = frame_number + 1
            runpbar.update(1)

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
            # start = time.time()
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes, rects = [], []
            confidences = []
            classIDs = []

            # loop over each of the layer outs
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
                        confidence > confidence_set
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
                boxes, confidences, confidence_set, threshold_set
            )

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # store detections boxes to use feature extraction tracking
                    rects.append((x, y, x + w, y + h))

                    if write:
                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(
                            LABELS[classIDs[i]], confidences[i]
                        )
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
            W_line1 = int(ratio_x_line1 * W)
            W_line2 = int(ratio_x_line2 * W)
            if write:
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
                        if (
                            direction * counting_direction > 0
                            and W_line1 < centroid[0] < W_line2
                        ):
                            totalUp += 1
                            to.counted = True
                            detection_frame_list.append(frame_number)

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                if write:
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
                    cv2.circle(
                        frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1
                    )

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
                    str(output_path),
                    fourcc,
                    30,
                    (frame.shape[1], frame.shape[0]),
                    True,
                )

                # end = time.time()
                # some information on processing single frame
                # if total > 0:
                #    elap = end - start
                #    print(
                #        "[INFO] single frame took {:.4f} seconds".format(elap)
                #    )
                #    print(
                #        "[INFO] estimated total time to finish: {:.4f}".format(
                #            elap * total
                #        )
                #    )

            if write:
                # write the output frame to disk
                writer.write(frame)

    # release the file pointers
    print("[INFO] cleaning up...")
    if write:
        writer.release()
    vs.release()

    return detection_frame_list
