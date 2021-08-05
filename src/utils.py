# utils.py
import cv2
from pathlib import Path
from tqdm import tqdm


def divide_video(
    input_path: Path,
    output_path: Path,
    cut_duration: int,
):
    cap = cv2.VideoCapture(str(input_path))  # Get video (using absolute path)
    # cv2.videocapture The absolute path must be correct to read the video, use / instead of \
    if (
        cap.isOpened()
    ):  # Judge whether the video can be opened and read smoothly, True means it can be opened, False means it cannot be opened
        print("video file is readable")
    else:
        print("cannot open video file")

    # Get video information
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )  # Get video frame width and frame height
    print("video width = {}, height ={}".format(width, height))

    if (
        cap.isOpened()
    ):  # When the video is successfully opened cap.isOpened() returns True, otherwise it returns False
        rate = cap.get(
            5
        )  # cap.get() The parameter in parentheses is 5 for obtaining the frame rate
        FrameNumber = int(
            cap.get(7)
        )  # Get the total number of frames of the video file
        duration = (
            FrameNumber / rate
        )  # The total number of video frames divided by the frame rate is equal to the video time, and the unit after dividing by 60 is minutes
        fps = int(
            rate
        )  # The number of frames of each small video (17 frames of 1 second of my video), which is also used for the frame rate of the later written video
        print("rate={}".format(rate))
        print("FrameNumber={}".format(FrameNumber))
        print("duration={}".format(duration))
        print("fps={}".format(fps))

    i = 0
    with tqdm(total=FrameNumber, desc="video processing...") as runpbar:
        while True:
            success, frame = cap.read()
            # cap.read() means to read the video by frame, success, frame get two return values ​​of cap.read()
            # Where success is a boolean value, and the frame is correct, it returns TRUE, if the file is read to the end, its return value is False
            # frame is each frame of image, which is a three-dimensional matrix
            if success:
                runpbar.update(1)
                i += 1
                if (
                    i % (cut_duration * fps) == 1
                ):  # Save a small video every 'duration' seconds, each second of the video has fps frames
                    # cap.read()Read the video frame by frame, every time a frame is read, i is +1,
                    # When the number of frames read can divide the number of frames of each small video, then save the video
                    videoWriter = cv2.VideoWriter(
                        str(output_path)
                        + str(i // (cut_duration * fps))
                        + ".mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (int(width), int(height)),
                    )
                    # videoWriter = cv2.VideoWriter(str(i) + '.mp4', -1, fps, (int(width), int(height)))
                    # Respectively: save file name, encoder, frame rate, video width and height
                    videoWriter.write(frame)  # Write frame image
                else:
                    videoWriter.write(frame)
            else:
                print("done dividing into {} files".format(i))
                break

    cap.release()  # Release video file
