import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
    resent的骨架
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(16, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    # x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = KL.Conv2D(32, (7, 7), strides=(2, 2), name='conv2_1', use_bias=True)(x)
    x = KL.Conv2D(64, (5, 5), strides=(2, 2), name='conv2_2', use_bias=True)(x)
    x = KL.Conv2D(32, (1, 1), strides=(2, 2), name='conv2_3', use_bias=True)(x)
    x = KL.Activation('relu')(x)
    C2 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 3
    x = KL.Conv2D(32, (5, 5), strides=(2, 2), name='conv3_1', use_bias=True)(x)
    x = KL.Conv2D(128, (3, 3), strides=(2, 2), name='conv3_2', use_bias=True)(x)
    x = KL.Conv2D(128, (3, 3), strides=(2, 2), name='conv3_3', use_bias=True)(x)
    x = KL.Activation('relu')(x)
    C3 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 4
    x = KL.Conv2D(128, (3, 3), strides=(2, 2), name='conv4_1', use_bias=True)(x)
    x = KL.Conv2D(128, (3, 3), strides=(2, 2), name='conv4_2', use_bias=True)(x)
    x = KL.Conv2D(128, (3, 3), strides=(2, 2), name='conv4_3', use_bias=True)(x)
    x = KL.Activation('relu')(x)
    C4 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 5
    if stage5:
        x = KL.Conv2D(256, (3, 3), strides=(2, 2), name='conv2_1', use_bias=True)(x)
        x = KL.Conv2D(256, (3, 3), strides=(2, 2), name='conv2_2', use_bias=True)(x)
        x = KL.Conv2D(256, (3, 3), strides=(2, 2), name='conv2_3', use_bias=True)(x)
        x = KL.Activation('relu')(x)
        C5 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

def build(self, mode, config):
    # Image size must be dividable by 2 multiple times
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    # Inputs
    input_image = KL.Input(
        shape=[None, None, 3], name="input_image")
    input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                name="input_image_meta")
    if mode == "training":
        # RPN GT
        input_rpn_match = KL.Input( shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input( shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
        input_gt_class_ids = KL.Input( shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        input_gt_boxes = KL.Input( shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = KL.Lambda(lambda x: norm_boxes_graph( x, K.shape(input_image)[1:3]))(input_gt_boxes)
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        if config.USE_MINI_MASK:
            input_gt_masks = KL.Input(
                shape=[config.MINI_MASK_SHAPE[0],
                       config.MINI_MASK_SHAPE[1], None],
                name="input_gt_masks", dtype=bool)
        else:
            input_gt_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                name="input_gt_masks", dtype=bool)
    elif mode == "inference":
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

    # Build the shared convolutional layers.
    # Bottom-up Layers
    # Returns a list of the last layers of each stage, 5 in total.
    # Don't create the thead (stage 5), so we pick the 4th item in the list.
    _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
    # 第一步：用resnet_101 生成 图片的高维特征值 _, C2, C3, C4, C5
    ###########构造FPN金字塔特征,用了金字塔(FPN)的网络方式进行特征的提取##################
    # Top-down Layers 上采样
    # TODO: add assert to varify feature map sizes match what's in config
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                   KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                   KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                   KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    mrcnn_feature_maps = [P2, P3, P4, P5]  # 两个特征图
    '''
    用这些特征图生成anchor
    Anchors 如果是training，就在金字塔的特征图上生成anchor，
    如果是inference,则anchor就是输入的anchor
    '''
    if mode == "training":
        '''
        根据config 获取金字塔层面上所有的anchor，实在原始图的大小上生成的anchor
        在金字塔特征图上以每个像素为中心，一配置文件的anchor大小为宽高
        生成anchor，并根据特征图，相对原图缩小的比例，还原到原始的输入
        图片上，则anchor是在原始图片上的坐标
        '''
        anchors = self.get_anchors(config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        # 将anchors复制到batch size的维度（2*261888*4）
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    else:
        anchors = input_anchors

    '''
    RPN Model 
    搭建RPN网络结构 输出中包含了rpn_class_logits,rpn_probs,rpn_bbox
    box的前景色和背景色的分类
    box框体的回归修正
    第二部：采用rpn网络对上采样得到高维特征图:rpn_feature_maps做处理，
    得到rpn_class_logits, rpn_class, rpn_bbox
    '''
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS),
                          config.TOP_DOWN_PYRAMID_SIZE)
    '''
    Loop through pyramid layers
    list of lists
    对rpn_feature_maps 特征数据逐一处理，得到"rpn_class_logits", "rpn_class", "rpn_bbox"
    '''
    layer_outputs = []
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]
    # rpn网络的输出值
    rpn_class_logits, rpn_class, rpn_bbox = outputs
    '''
    rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.

    Generate proposals layer
    Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
    and zero padded.
    ProposalLayer 层来修正anchor
    '''
    proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
        else config.POST_NMS_ROIS_INFERENCE
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    if mode == "training":
        # Class ID mask to mark class IDs supported by the dataset the image
        # came from.
        active_class_ids = KL.Lambda(
            lambda x: parse_image_meta_graph(x)["active_class_ids"]
        )(input_image_meta)
        if not config.USE_RPN_ROIS:
            # Ignore predicted ROIs and use ROIs provided as an input.
            input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
            # Normalize coordinates
            target_rois = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_rois)
        else:
            target_rois = rpn_rois

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask = \
            DetectionTargetLayer(config, name="proposal_targets")([
                target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
        '''
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinements.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
               boundaries and resized to neural network output size.
        '''
        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        '''
        从对应的特征图中取出坐标对应的区域，利用双线性插值的方式进行pooling操作。
        PyramidROIAlign会返回resize成相同大小的rois。
        将得到的特征块输入到fpn_classifier_graph网络中，得到分类和回归值
        '''
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
            fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, config.NUM_CLASSES,
                                 train_bn=config.TRAIN_BN,
                                 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
        '''
        从对应的特征图中取出坐标对应的区域，利用双线性插值的方式进行pooling操作。
        PyramidROIAlign会返回resize成相同大小的rois。
        对mrcnn_feature_maps做fpn处理
        '''
        mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                          input_image_meta,
                                          config.MASK_POOL_SIZE,
                                          config.NUM_CLASSES,
                                          train_bn=config.TRAIN_BN)

        # TODO: clean up (use tf.identify if necessary)
        output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
            [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
            [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
            [target_class_ids, mrcnn_class_logits, active_class_ids])
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
            [target_bbox, target_class_ids, mrcnn_bbox])
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
            [target_mask, target_class_ids, mrcnn_mask])

        # Model 总体模型
        inputs = [input_image, input_image_meta,
                  input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        if not config.USE_RPN_ROIS:
            inputs.append(input_rois)
        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                   rpn_rois, output_rois,
                   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        model = KM.Model(inputs, outputs, name='mask_rcnn')
    else:
        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
            fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, config.NUM_CLASSES,
                                 train_bn=config.TRAIN_BN,
                                 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
        # normalized coordinates
        detections = DetectionLayer(config, name="mrcnn_detection")(
            [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                          input_image_meta,
                                          config.MASK_POOL_SIZE,
                                          config.NUM_CLASSES,
                                          train_bn=config.TRAIN_BN)

        model = KM.Model([input_image, input_image_meta, input_anchors],
                         [detections, mrcnn_class, mrcnn_bbox,
                          mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                         name='mask_rcnn')

    # Add multi-GPU support.
    if config.GPU_COUNT > 1:
        from mrcnn.parallel_model import ParallelModel
        model = ParallelModel(model, config.GPU_COUNT)

    return model