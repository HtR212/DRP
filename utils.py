import numpy as np

class AdamOpt:

    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.exp_avg = np.zeros(size, dtype=dtype)
        self.exp_avg_sq = np.zeros(size, dtype=dtype)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.step = 0

    def update(self, grad):

        self.step += 1

        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)

        denom = (np.sqrt(self.exp_avg_sq) / np.sqrt(bias_correction2)) + self.eps

        step_size = self.lr / bias_correction1

        return step_size / denom * self.exp_avg
