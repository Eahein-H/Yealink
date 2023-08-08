import numpy as np
import math


def CalculateIOU(bbox_pre, bbox):
    assert len(bbox_pre)==4 and len(bbox)==4,"bounding box coordinate size must be 4"
    bxmin = max(bbox_pre[0],bbox[0])
    bymin = max(bbox_pre[1],bbox[1])
    bxmax = min(bbox_pre[2],bbox[2])
    bymax = min(bbox_pre[3],bbox[3])
    bwidth = bxmax-bxmin
    bhight = bymax-bymin
    inter = bwidth*bhight
    union = (bbox_pre[2]-bbox_pre[0])*(bbox_pre[3]-bbox_pre[1])+(bbox[2]-bbox[0])*(bbox[3]-bbox[1])-inter

    return inter/union


class OneEuroFilter:
    def __init__(self, min_cutoff=0.004, beta=0.7, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter_signal(self, x, x_prev, dx_prev):
        """Compute the filtered signal."""
        t_e = 1

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, x_prev)

        return x_hat, dx_hat

# if __name__ == '__main__':
#     bbox1 = np.array([5, 5, 10, 10])
#     bbox2 = np.array([6, 4, 10, 9])
#     print(CalculateIOU(bbox1, bbox2))
