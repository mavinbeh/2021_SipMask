from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2
from mmcv.runner import  obj_from_dict
from mmdet import datasets
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets import to_tensor
from mmcv.parallel import DataContainer as DC

config_file = 'configs/sipmask/sipmask_r50_caffe_fpn_gn_1x.py'
checkpoint_file = 'checkpoints/vis_sipmask_ms_1x_final.pth'
save_path = 'results/images'
device='cuda:0'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=device)

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))


def prepare_data(img, img_transform, cfg, device, frame_id):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    _img_meta = [[
                    dict(
                        ori_shape=ori_shape,
                        img_shape=img_shape,
                        pad_shape=pad_shape,
                        scale_factor=scale_factor,
                        flip=False,
                        is_first=frame_id==1,
                        video_id=100,
                        frame_id=frame_id)
                ]]
                    
    img_metas = []
    img_metas.append(DC(_img_meta, cpu_only=True))
    
        
    
    return dict(img=[img], img_meta=img_metas)



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
    result = inference_detector(model, frame)

    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    data = prepare_data(frame, img_transform, model.cfg, device, video.position)

    model.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES,
                                     save_vis = True,
                                     save_path = save_path,
                                     is_video = True)
    model.show_result(data, result, cfg.data.test.img_norm_cfg,
                                     dataset=model.CLASSES,
                                     save_vis = True,
                                     save_path = save_path,
                                     is_video = True)
