
from mmdet.apis import Inferencer
import mmcv
import cv2
from mmcv.runner import  obj_from_dict
from mmdet import datasets
from mmdet.datasets.transforms import ImageTransform

config_file = 'configs/sipmask/sipmask_r50_caffe_fpn_gn_1x.py'
checkpoint_file = 'checkpoints/vis_sipmask_ms_1x_final.pth'
save_path = 'results/images'
device='cuda:0'

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

inferencer = Inferencer(cfg.data.test.img_scale, cfg.data.test.img_norm_cfg, size_divisor=cfg.data.test.size_divisor, test_mode=True, with_label=False, with_mask=False)
# build the model from a config file and a checkpoint file
model = inferencer.init_detector(config_file, checkpoint_file, device=device)




# test a single image and show the results
# img = 'data/YouTubeVIS/valid/JPEGImages/fd5bf99712/00090.jpg'  
# img = mmcv.imread(img)
# result = inference_detector(model, img)
# show_result(img, result, model.CLASSES)

# test a list of images and write the results to image files
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

# test a video and show the results
video = mmcv.VideoReader('/SipMask-VIS/data/abc2_cuted.mp4')

for frame in video:
    (result, data) = inferencer.inference_detector(model, frame, video.position -1 )

    model.show_result(data, result, cfg.data.test.img_norm_cfg,
                                     dataset=model.CLASSES,
                                     save_vis = True,
                                     save_path = save_path,
                                     is_video = True)
