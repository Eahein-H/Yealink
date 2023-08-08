import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import os.path as osp
import argparse
import numpy as np
import tqdm
import copy
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2
import time
from PIL import Image
sys.path.append("/home/huangyih/OSX/ultralytics/yolov5")
from NeWCRFs.newcrfs.test_forOSX import test as depth_test
from NeWCRFs.newcrfs.test_forOSX import convert_arg_line_to_args
from Yealink_Project.videoSmoother import Smoother
from Yealink_Project.videoWriter import Writer
from Yealink_Project.poseSmoother import CalculateIOU, OneEuroFilter
from common.utils.human_models import smpl_x
from Yealink_Project.videoConcat import concat
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='1')
    parser.add_argument('--video_path', type=str, default='/home/huangyih/OSX/Yealink_Project/input_video/202307211422.mp4')
    parser.add_argument('--img_path', type=str, default='/home/huangyih/mmhuman3d_new/vis_results_4/images')
    parser.add_argument('--output_folder', type=str, default='/home/huangyih/OSX/Yealink_Project/output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')

    parser.add_argument('--model_name', type=str, help='model name', default='newcrfs')
    parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--data_path', type=str, help='path to the data')
    parser.add_argument('--output_path', type=str, help='path out to the data')
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file')
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='/home/huangyih/OSX/NeWCRFs/model_nyu.ckpt')
    parser.add_argument('--dataset', type=str, help='dataset to train on', default='nyu')
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--save_viz', help='if set, save visulization of the outputs', action='store_true')
    # args = parser.parse_args()

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

torch.cuda.empty_cache()

# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()

# file_path = []
# for root, ds, fs in os.walk("/home/huangyih/mmhuman3d_new/vis_results_4/images/"):
#         for f in fs:
#             fullname = os.path.join(root, f)
#             file_path.append(fullname)

# video to images
video_name = args.video_path.split('/')[-1]
seg_img_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/original_images")
# print(seg_img_path)
os.makedirs(seg_img_path, exist_ok=True)

# os.system(f'ffmpeg -i {args.video_path} -f image2 -r 30 -b:v 5626k {seg_img_path}/%06d.png')

frames = sorted(os.listdir(seg_img_path))
frames.sort(key= lambda x:int(x[:-4]))

# resize to (640, 480) and save img_path to txt files
print("Start resizing images...")
txt_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/pathfile.txt")
res_img_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/resized_images")
os.makedirs(res_img_path, exist_ok=True)
# txt_file = open(txt_path, 'w+')
# for frame in frames:
#     path = os.path.join(seg_img_path, frame)
#     image = Image.open(path)
#     new_im = image.resize((640, 480))

#     txt_file.write(frame)
#     txt_file.write('\n')
#     save_img = os.path.join(res_img_path, frame)

#     new_im.save(save_img, 'JPEG', quality=95)
# txt_file.close()

# depth map prediction
depth_out_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/depth_images")
args.mode = 'test'
args.data_path = res_img_path
args.filenames_file = txt_path
args.output_path = depth_out_path
pred_depth = depth_test(args)

# 3D pose estimation
counts = 0
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
render_out_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/rendered_images/original_render")
os.makedirs(render_out_path, exist_ok=True)
black_path = "/home/huangyih/OSX/Yealink_Project/resources/wholeblack.PNG"
visual_path = "/home/huangyih/OSX/Yealink_Project/resources/visualmeeting3.PNG"
black_out_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/rendered_images/black_render")
os.makedirs(black_out_path, exist_ok=True)
visual_out_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/rendered_images/visual_render")
os.makedirs(visual_out_path, exist_ok=True)
black_image = load_img(black_path)
visual_image = load_img(visual_path)

# bbox memory
bbox_memory = []
pose_filter = OneEuroFilter(min_cutoff=0.004, beta=0.7)

smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
# zero_pose = np.zeros((1, 3), dtype=float)
zero_pose = torch.zeros((1, 3)).float().cuda() # .repeat(batch_size, 1)

for frame in frames:
    counts += 1
    print(counts)
    # if counts == 101:
    #     break
    # prepare input image
    transform = transforms.ToTensor()
    path = os.path.join(seg_img_path, frame)
    original_img = load_img(path)
    original_img_height, original_img_width = original_img.shape[:2]

    # detect human bbox with yolov5s
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    class_ids, confidences, boxes, centers, boxes_mem = [], [], [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        boxes_mem.append(np.array([x1, y1, x2, y2]))
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        centers.append([x_c, y_c])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vis_img = original_img.copy()
    if black_image.shape[0] == original_img_height:
        vis_img_black = black_image.copy()
    else:
        vis_img_black = black_image.resize((original_img_height, original_img_width))
    if visual_image.shape[0] == original_img_height:
        vis_img_visual = visual_image.copy()
    else:
        vis_img_visual = visual_image.resize((original_img_height, original_img_width))

    # sort the indices by predicted depth map
    sort_indices = []
    for num, indice in enumerate(indices):
        re_x = centers[indice][0] / original_img_width * 640
        re_y = centers[indice][1] / original_img_height * 480
        depth = pred_depth[int(counts - 1)][int(re_y)][int(re_x)]

        sort_indices.append([indice, depth])
    sort_indices.sort(key= lambda x:x[1], reverse=True)
       
    for sort_indice in sort_indices:
        bbox = boxes[sort_indice[0]]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        
        keypoints = []
        root_pose = out['smplx_root_pose'].detach().cpu().numpy().squeeze()
        body_pose = out['smplx_body_pose'].detach().cpu().numpy().squeeze()
        lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy().squeeze()
        rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy().squeeze()
        jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy().squeeze()
        shape = out['smplx_shape'].detach().cpu().numpy().squeeze()
        expr = out['smplx_expr'].detach().cpu().numpy().squeeze()
        cam_trans = out['cam_trans'].detach().cpu().numpy().squeeze()
        mesh = mesh[0]
        keypoints.append(root_pose)
        keypoints.append(body_pose)
        keypoints.append(lhand_pose)
        keypoints.append(rhand_pose)
        keypoints.append(jaw_pose)
        keypoints.append(shape)
        keypoints.append(expr)
        keypoints.append(cam_trans)

        # print(centers[100])
        # print(mesh[0])
        # print(mesh[1])
        if counts == 1:
            length = 0
            for keys in keypoints:
                length += len(keys)

        # check bbox and do mesh smoothing
        new_bbox_memory = []
        dx_prev = np.zeros((int(length), 1), dtype=float)
        if counts == 1:
            bbox_memory.append([boxes_mem[sort_indice[0]], keypoints, dx_prev])
        else:
            max_IOU = 0
            max_IOU_index = 0
            for ii in range(len(bbox_memory)):            
                now_IOU = CalculateIOU(bbox_memory[ii][0], boxes_mem[sort_indice[0]])
                if now_IOU > max_IOU:
                    max_IOU = now_IOU
                    max_IOU_index = ii
            
            # print(max_IOU)
            if max_IOU > 0.5:
                new_keypoints = []
                dx_counts = 0
                key_counts = 0
                for keys in keypoints:
                    new_keys = []
                    item_counts = 0
                    for item in keys:
                        new_item, prev = pose_filter.filter_signal(item, bbox_memory[max_IOU_index][1][key_counts][item_counts],
                                                                   bbox_memory[max_IOU_index][2][dx_counts])
                        new_keys.append(new_item)
                        dx_prev[dx_counts] = prev
                        item_counts += 1
                        dx_counts += 1
                    new_keypoints.append(new_keys)
                    key_counts += 1
                new_bbox_memory.append([boxes_mem[sort_indice[0]], new_keypoints, dx_prev])
                root_pose_t = torch.Tensor(new_keypoints[0]).float().cuda().reshape((1, 3))
                body_pose_t = torch.Tensor(new_keypoints[1]).float().cuda().reshape((1, 63))
                lhand_pose_t = torch.Tensor(new_keypoints[2]).float().cuda().reshape((1, 45))
                rhand_pose_t = torch.Tensor(new_keypoints[3]).float().cuda().reshape((1, 45))
                jaw_pose_t = torch.Tensor(new_keypoints[4]).float().cuda().reshape((1, 3))
                shape_pose_t = torch.Tensor(new_keypoints[5]).float().cuda().reshape((1, 10))
                expr_pose_t = torch.Tensor(new_keypoints[6]).float().cuda().reshape((1, 10))
                cam_pose_t = torch.Tensor(new_keypoints[7]).float().cuda().reshape((1, 3))
                smplx_output = smplx_layer(global_orient=root_pose_t, body_pose=body_pose_t, left_hand_pose=lhand_pose_t,
                                           right_hand_pose=rhand_pose_t, jaw_pose=jaw_pose_t, betas=shape_pose_t,
                                           expression=expr_pose_t, leye_pose=zero_pose, reye_pose=zero_pose)
                mesh_cam = smplx_output.vertices
                mesh_cam = mesh_cam + cam_pose_t[:, None, :]
                mesh = mesh_cam.detach().cpu().numpy().squeeze()
            else:
                new_bbox_memory.append([boxes_mem[sort_indice[0]], keypoints, dx_prev])

        # print(mesh[0])
        # print(mesh[1])
        # if counts == 2:
        #     print(centers[100])
        # start_time = time.time()

        # save each mesh
        # save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))

        # render mesh
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        # vis_img_visual = render_mesh(vis_img_visual, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        # vis_img_black = render_mesh(vis_img_black, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})

    # save rendered image
    image_path = str(counts) + ".jpg"
    cv2.imwrite(os.path.join(render_out_path, image_path), vis_img[:, :, ::-1])
    # cv2.imwrite(os.path.join(black_out_path, image_path), vis_img_black[:, :, ::-1])
    # cv2.imwrite(os.path.join(visual_out_path, image_path), vis_img_visual[:, :, ::-1])

# output video creation
video_out_path = os.path.join(args.output_folder, video_name.split('.')[0] + "/video_output")
os.makedirs(video_out_path, exist_ok=True)

video_dir = os.path.join(video_out_path, 'rendered_video.mp4')
Writer(render_out_path, video_dir)
# video_dir_black = os.path.join(video_out_path, 'rendered_black_video.mp4')
# Writer(black_out_path, video_dir_black)
# video_dir_visual = os.path.join(video_out_path, 'rendered_visual_video_smooth.mp4')
# Writer(visual_out_path, video_dir_visual)

# video smoothing
print("Start video concating...")
concat(args.video_path, video_dir, video_out_path)
# Smoother(video_dir, video_dir_black, video_out_path, args.video_path)
