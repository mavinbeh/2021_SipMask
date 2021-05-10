import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import (ImageTransform, BboxTransform, MaskTransform, Numpy2Tensor)
from mmdet.models import build_detector
from mmcv.parallel import DataContainer as DC

class Inferencer:
    def __init__(self,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        self.frame_id_counter = 0
    

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio


    def init_detector(self, config, checkpoint=None, device='cuda:0'):
        """Initialize a detector from config file.

        Args:
            config (str or :obj:`mmcv.Config`): Config file path or the config
                object.
            checkpoint (str, optional): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        if isinstance(config, str):
            config = mmcv.Config.fromfile(config)
        elif not isinstance(config, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            'but got {}'.format(type(config)))
        config.model.pretrained = None
        model = build_detector(config.model, test_cfg=config.test_cfg)
        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint)
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.warn('Class names are not saved in the checkpoint\'s '
                            'meta data, use COCO classes by default.')
                model.CLASSES = get_classes('coco')
        model.cfg = config  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model


    def inference_detector(self, model, imgs, frame_id):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
        """
        cfg = model.cfg
        img_transform = ImageTransform(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

        device = next(model.parameters()).device  # model device
        if not isinstance(imgs, list):
            return self._inference_single(model, imgs, img_transform, device, frame_id)
        else:
            return self._inference_generator(model, imgs, img_transform, device)


    def _prepare_data(self, img, img_transform, cfg, device, frame_id):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid = 0
        proposal = None
        

        def prepare_single(img, frame_id, scale, flip, proposal=None):
            ori_shape = img.shape
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            ori_shape = img.shape
            _img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                is_first=(frame_id == 0),
                video_id=vid,
                frame_id =frame_id,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, frame_id, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data


            



    def _inference_single(self, model, img, img_transform, device, frame_id):
        img = mmcv.imread(img)
        data = self._prepare_data(img, img_transform, model.cfg, device, frame_id)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        return (result, data)


    def _inference_generator(self, model, imgs, img_transform, device):
        for img in imgs:
            yield self._inference_single(model, img, img_transform, device)


    # TODO: merge this method with the one in BaseDetector
    def show_result(self, img,
                    result,
                    class_names,
                    score_thr=0.3,
                    wait_time=0,
                    show=True,
                    out_file=None):
        """Visualize the detection results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            class_names (list[str] or tuple[str]): A list of class names.
            score_thr (float): The threshold to visualize the bboxes and masks.
            wait_time (int): Value of waitKey param.
            show (bool, optional): Whether to show the image with opencv or not.
            out_file (str, optional): If specified, the visualization result will
                be written to the out file instead of shown in a window.

        Returns:
            np.ndarray or None: If neither `show` nor `out_file` is specified, the
                visualized image is returned, otherwise None is returned.
        """
        assert isinstance(class_names, (tuple, list))
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
        if not (show or out_file):
            return img


    def show_result_pyplot(self, img,
                        result,
                        class_names,
                        score_thr=0.3,
                        fig_size=(15, 10)):
        """Visualize the detection results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            class_names (list[str] or tuple[str]): A list of class names.
            score_thr (float): The threshold to visualize the bboxes and masks.
            fig_size (tuple): Figure size of the pyplot figure.
            out_file (str, optional): If specified, the visualization result will
                be written to the out file instead of shown in a window.
        """
        img = show_result(
            img, result, class_names, score_thr=score_thr, show=False)
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
