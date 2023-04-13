# from DRP.config import MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH
# from DRP.car_motion import CarMotion
# from DRP.load_sensor_data import load_sensor_data
from DRP.utils import AdamOpt
# from DRP.models import create_model
# from DRP.replay_bicycle import ReplayBicycle
from loss import compute_path_pinv, loss_func
# from DRP.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
#                         IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
#                         MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
#                         YUV_MIN, YUV_MAX
#                         )
# import os
# import pickle
# from logging import getLogger

import numpy as np
# import pandas as pd
# import cv2
from tqdm import tqdm
# import tensorflow as tf

CAMERA_IMG_HEIGHT = 484
CAMERA_IMG_WIDTH = 1164

# BEV_BASE_HEIGHT = int(CAMERA_IMG_HEIGHT * 1.1)
# BEV_BASE_WIDTH = CAMERA_IMG_WIDTH

BEV_BASE_HEIGHT = 968
BEV_BASE_WIDTH = 1000  # 1408

MODEL_IMG_HEIGHT = 128
MODEL_IMG_WIDTH = 256
MODEL_IMG_CH = 6

IMG_CROP_HEIGHT = 256
IMG_CROP_WIDTH = 512


PIXELS_PER_METER = 29 / 3.6576  # bev pixels / lane width
SKY_HEIGHT = 390

IMG_INPUT_SHAPE = (1, 6, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)
IMG_INPUT_MASK_SHAPE = (1, 1, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)
RNN_INPUT_SHAPE = (1, 512)
MODEL_DESIRE_INPUT_SHAPE = (1, 8)
MODEL_OUTPUT_SHAPE = (1, 1760)

YUV_MIN = -8.68289960e-01
YUV_MAX = 4.87138199e-01

DTYPE = np.float32


class CarMotionAttack:

    # replicating the init function
    def __init__(
        self,
        # sess,
        list_bgr_img,
        # df_sensors,
        global_bev_mask,
        base_color,
        # roi_mat,
        n_epoch=10000,
        # model_path="data/model_keras_20191230_v0.7/driving_model.h5",
        # # model_path="data/model_keras_20191125/driving_model.h5",
        learning_rate_patch=1.0e-1,
        # learning_rate_color=1.0e-3,
        # scale=1,
        # result_dir='./result/',
        perturbable_area_ratio=50,
        is_attack_to_right=True,
        # left_lane_pos=4,
        # right_lane_pos=36,
        # src_corners=None,
        # target_deviation=0.5,
        l2_weight=0.01
    ):
        # self.sess = sess
        self.list_bgr_img = list_bgr_img
        self.n_frames = len(list_bgr_img)
        self.list_model_patches = [np.zeros(IMG_INPUT_SHAPE, dtype=DTYPE) for _ in range(self.n_frames)]
        # self.df_sensors = df_sensors
        # self.result_dir = result_dir
        self.perturbable_area_ratio = perturbable_area_ratio
        self.base_color = base_color
        # self.roi_mat = roi_mat
        self.is_attack_to_right = is_attack_to_right
        # self.left_lane_pos = left_lane_pos
        # self.right_lane_pos = right_lane_pos
        # self.target_deviation = target_deviation
        self.l2_weight = l2_weight

        # self.last_epoch = None

        self.global_bev_mask = global_bev_mask
        # self.car_motion = CarMotion(
        #     self.list_bgr_img,
        #     self.df_sensors,
        #     self.global_bev_mask,
        #     self.roi_mat,
        #     left_lane_pos=left_lane_pos,
        #     right_lane_pos=right_lane_pos,
        #     scale=scale,
        #     src_corners=src_corners
        # )

        self.global_bev_perturbation = (
            np.ones(
                (self.global_bev_mask.shape[0], self.global_bev_mask.shape[1], 6),
                dtype=DTYPE,
            )
            * 1.0e-10
        )
        self.masked_global_bev_perturbation = self.global_bev_perturbation.copy()

        # self.global_base_color = np.array(
        #     [base_color, 0, 0], dtype=DTYPE
        # )  # np.zeros(3, dtype=DTYPE)

        self.global_base_color = np.array[self.base_color, 0, 0]    # 1 x 3
        self.expand_base_color = np.array([self.global_base_color[0]] * 4 + [self.global_base_color[1]] + [self.global_base_color[2]])  # 1 x 6
        self.ops_tiled_base_color = np.stack([self.expand_base_color for _ in range(IMG_INPUT_SHAPE[2])], axis=-1)      # 1 x 6 x H
        self.ops_tiled_base_color = np.stack([self.ops_tiled_base_color for _ in range(IMG_INPUT_SHAPE[3])], axis=-1)   # 1 x 6 x H x W

        # self.model = create_model()
        # self.model.load_weights(model_path)

        self.n_epoch = n_epoch
        self.learning_rate_patch = learning_rate_patch
        # self.learning_rate_color = learning_rate_color

        self.prev_ops_obj = 0
        self.prev_list_model_patches = self.list_model_patches


    # replicating CarMotionAttack.run()
    def run(
        self, 
        lateral_shift=4, 
        starting_meters=60, 
        starting_steering_angle=True,
        starting_patch_dir=None,
        starting_patch_epoch = None
    ):

        # the original definition of the gradient of the objective function
        # self.list_ops_gradients = tf.gradients(
        #     self.ops_obj, self.list_tf_model_patches + [self.tf_base_color]
        # )
        # we don't need to attach the base color here as it won't be changed anyway

        # adam optimizer
        adam = AdamOpt(
            self.global_bev_perturbation.shape, lr=self.learning_rate_patch
        )

        for epoch in tqdm(range(self.n_epoch), desc="optimization loop"):
            if epoch % 100 == 0:
                adam.lr *= 0.9

            # step 1: calculate the new inputs: images with patch, the first patch will be a patch with only the base color
            self.list_model_imgs = 

            self.list_ops_model_img = [
                np.where(
                    np.isnan(self.list_model_patches[i]),
                    self.list_model_imgs[i],                # perspective tilted image, according to the steering, not BEV
                    np.clip(self.list_model_patches[i] + self.ops_tiled_base_color, YUV_MIN, YUV_MAX)   # perturbation + base_color (asphalt)
                )
                for i in range(self.n_frames)
            ]

            # step 2: run the model to get the new output, specifically the prediction of lane lines, as listed in the next line
            # pred_x_left, pred_x_right, pred_y_left, pred_y_right, pred_prob

            # call bridge.py as if it's a function

            self.model_attack_outputs = 

            self.pred_x_left[i] = 
            self.pred_x_right[i] = 
            self.pred_y_left[i] = 
            self.pred_y_right[i] = 
            self.pred_prob[i] = 

            # then calculate the new objective function result/loss
            # first part of the objective function
            self.ops_obj_shifting = sum(
                loss_func(                  # size
                    self.pred_x_left[i],    # 33
                    self.pred_x_right[i],   # 33
                    self.pred_y_left[i],    # 33
                    self.pred_y_right[i],   # 33
                    self.pred_prob[i],      # 2
                    is_attack_to_right=self.is_attack_to_right
                )
                for i in tqdm(range(self.n_frames), desc="loss_p1")
            )

            # second part of the objective function
            self.ops_obj_l2 = sum(
                np.nansum(
                    np.square(
                        self.list_model_patches[i]  # 1 x 6 x H x W (IMG_INPUT_SHAPE)
                    )
                )
                for i in tqdm(range(self.n_frames), desc="loss_p2")
            )/2

            # the objective function/loss
            self.ops_obj = self.ops_obj_shifting + (self.l2_weight * self.ops_obj_l2)   # constant

            # step 3: calculate the new gradient, y = objective function, x = patch
            grads = self.get_experimental_gradient()
            list_var_grad = [g[0].transpose((1, 2, 0)) for g in grads[:]]
            patch_grad = self.aggregate_grad(list_var_grad)
        
            # step 4: update the patch through Adam optimizer
            self.global_bev_perturbation -= adam.update(patch_grad)
            self.global_bev_perturbation = self.global_bev_perturbation.clip(
                0, - self.base_color
            )

            # stealthiness postprocessing
            self.global_bev_perturbation[:, :, 4:] = 0

            patch_diff = self.global_bev_perturbation.sum(axis=2)
            threshold = np.percentile(patch_diff, 100 - self.perturbable_area_ratio)
            mask_bev_perturbation = patch_diff >= threshold

            self.masked_global_bev_perturbation = self.global_bev_perturbation.copy()
            self.masked_global_bev_perturbation[~mask_bev_perturbation] = 0.

            # step 5: apply the new patch perturbation to each frame
            list_patches = self.car_motion.conv_patch2model(
                self.masked_global_bev_perturbation, self.global_base_color
            )

            # update the patch perturbation for next iteration
            self.prev_list_model_patches = self.list_model_patches
            self.list_model_patches = [np.expand_dims(list_patches[i].transpose((2, 0, 1)), axis = 0) for i in range(self.n_frames)]


    def get_experimental_gradient(self):
        # here, we can hardly use the same idea as that in the code of DRP paper, which employs tensorflow to get the gradient
        # the problem is that supercombo is an onnx model and we don't have the backward functions of this model
        # instead, we'll use (loss-prev_loss)/(patch-prev_ptach) to get the trivial gradient
        # the gradient will be calculated against each pixel in a patch, resulting in a grad matrix of the same shape as list_model_patches
        diff_patches = self.list_model_patches - self.prev_list_model_patches

        # if diff_patches is zero at some entry, the corresponding gradient will be 0. For example, the first epoch will get an all-zero gradient
        # but since the initial value of self.global_bev_perturbation is not zero, there will be a valid perturbation to the patch
        grads = np.divide(self.ops_obj-self.prev_ops_obj, diff_patches, out=np.zeros_like([0]), where=diff_patches!=0)

        return grads
    d
    def aggregate_grad(self, list_var_grad):
        # step 1: calculate the weight of the patch of each frame
        model_mask_areas = np.array(
            [m.sum() for m in self.car_motion.get_all_model_masks()]
        )
        weights = model_mask_areas / model_mask_areas.sum()

        # step 2: transform the gradient matrix from model input size (IMG_INPUT_SHAPE) to BEV shape
        list_patch_grad = self.car_motion.conv_model2patch(
            list_var_grad
        )  # zero is missing value

        # step 3: aggregation
        for i in range(len(list_patch_grad)):
            list_patch_grad[i] *= weights[i]

        tmp = np.stack(list_patch_grad)

        tmp = np.nanmean(tmp, axis=0)

        # step 4: clean NaN values
        tmp[np.isnan(tmp)] = 0

        return tmp






def main():
    return




