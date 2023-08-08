import numpy as np
import cv2
import os


def concat(original_video, rendered_video, out_path):
    cap_o = cv2.VideoCapture(original_video)
    cap_r = cv2.VideoCapture(rendered_video)

    w = int(cap_o.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_o.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap_o.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

    out = cv2.VideoWriter(os.path.join(out_path, 'Concat_video.mp4'), fourcc, fps, (2 * w, h))

    n_frames = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))

    for ii in range(n_frames):
        frame_o = cap_o.read()
        frame_r = cap_r.read()

        frame_out = cv2.hconcat([frame_o, frame_r])
        out.write(frame_out)
    
    cap_o.release()
    cap_r.release()
    out.release()

    print("All finished!")