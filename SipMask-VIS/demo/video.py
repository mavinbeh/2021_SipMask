from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2

config_file = 'configs/sipmask/sipmask_r50_caffe_fpn_gn_1x.py'
checkpoint_file = 'checkpoints/vis_sipmask_ms_1x_final.pth'
print("x")
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

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
video = mmcv.VideoReader('/SipMask-VIS/data/test.mp4')

for frame in video:
    print("y")
    result = inference_detector(model, frame)
    show_result(frame, result, model.CLASSES, wait_time=1, out_file="results/images/1.png")
