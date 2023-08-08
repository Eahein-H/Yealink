import os
import cv2

def Writer(image_path, video_dir):
    fps = 30
    rendered_frames = sorted(os.listdir(image_path))
    rendered_frames.sort(key= lambda x:int(x[:-4]))

    img = cv2.imread(os.path.join(image_path, rendered_frames[0]))
    img_size = (img.shape[1], img.shape[0])
    seq_name = os.path.dirname(image_path).split('/')[-1]

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for renderframe in rendered_frames:
        f_path = os.path.join(image_path, renderframe)
        image = cv2.imread(f_path)
        videowriter.write(image)
        
    print("All rendered images have been written!")

    videowriter.release()

Writer("/home/huangyih/OSX/Yealink_Project/output/202307211422/rendered_images/no_smooth", "/home/huangyih/OSX/Yealink_Project/output/1/video_output/rendered_nosmooth_video_smooth.mp4")