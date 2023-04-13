import numpy as np
import math

# reserved. copied from https://github.com/MTammvee/openpilot-supercombo-model/blob/main/openpilot_onnx.py
X_IDXS = np.array([0., 0.1875, 0.75, 1.6875, 3., 4.6875, 6.75, 
                   9.1875, 12., 15.1875, 18.75, 22.6875, 27.,
                   31.6875, 36.75, 42.1875, 48., 54.1875,60.75, 
                   67.6875, 75., 82.6875, 90.75, 99.1875, 108., 
                   117.1875, 126.75, 136.6875, 147., 157.6875, 
                   168.75, 180.1875, 192.])

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# pred_x_*, pred_y_* contains x-y locations of sample points of the predicted lane lines; expected shape: [33]
# pred_std_x_* and pred_std_y_* contains the standard deviations of the predicted x-y locations of the points in pred; expected shape: [33]
# Note: standard deviations are reserved for other methods to calculate the polynomial fitting
# pred_prob contains the probabilities/confidence in the prediction of the left and right lane lines; expected shape: [2]
# n contains the number of sample points to be used when calculating the sum of gradients
def loss_func(
    pred_x_left, pred_x_right, 
    pred_y_left, pred_y_right,
    pred_prob,
    pred_x_std_left=None, pred_x_std_right=None,
    pred_y_std_left=None, pred_y_std_right=None,
    polyfit_degree = 4, is_attack_to_right=True,
    n = 25
):
    # step 1: preprocessing; as we don't want the constant in polyfit result, divide y by x first
    for i in range(len(pred_x_left)):
        pred_y_left[i] /= pred_x_left[i]
        pred_y_right[i] /= pred_x_right[i]

    # step 2: polynomial fitting
    # degree used here is polyfit_degree - 1 because we have preprocessed the data and assume that there is not a constant in the fit result
    left_poly = np.polyfit(pred_x_left, pred_y_left, polyfit_degree-1)
    right_poly = np.polyfit(pred_x_right, pred_y_right, polyfit_degree-1)

    # step 3: use the probability to combine these two fittings into one polynomial fit for the center of the lane lines/path
    left_prob = sigmoid(pred_prob[0])
    right_prob = sigmoid(pred_prob[1])
    center_poly = (left_prob * left_poly + right_prob * right_poly)/(left_prob+right_prob)

    # step 4: calculate the sum of gradients at n sample points
    obj = 0
    for i in range(n):
        gradients = 0
        for j in range(polyfit_degree):
            gradient += (polyfit_degree-j) * center_poly[j] * (i ** (polyfit_degree-j-1))
        obj += gradients

    if is_attack_to_right:
        return obj # Attack to right
    else:
        return - obj
    



