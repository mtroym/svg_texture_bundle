import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


# random.seed(10)

# INIT CANVAS = 360(H), 720(W)
__all__ = ['BGGenTony']
DEBUG = False


class BGGenTony:
    """
    : HOW TO USE
    from BGGenTony import BGGenTony
    H ,W = 1080, 1920
    RGB = np.array([[[100, 160, 255]]])
    RGB2 = np.array([[[90, 60, 200]]])
    tonyer = BGGenTony(H, W) # BGgen agent
    for style in tonyer.all:
        bg = tonyer.__getattribute__(style)(RGB, RGB2)
        plt.imshow(bg)
        plt.show()
    """
    def __init__(self, h: int, w: int, layout=None):
        """
        :param h: height of bg
        :param w: width of bg
        """
        self.h = h
        self.w = w
        self.all = ['sea_bg', 'jp_bg', 'point_bg']

    def sea_bg(self, RGB1, RGB2=None):
        """
        :param RGB1: numpy of shape 1 1 3
        :param RGB2: optional, numpy of shape 1 1 3
        :return: np.array of shape [H, W, 3]
        """
        # Gen 2 pure color bgs.
        img = (np.ones((self.h, self.w, 3)) * np.array(RGB1)).astype(np.uint8)
        if RGB2 is None:
            color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # random shift color.
            color_converted[:, :, 0] += random.randrange(90)
            if random.random() > 0.5:
                color_converted[:, :, 1] += random.randint(50, 100)  # became brighter.
            else:
                color_converted[:, :, 1] -= random.randint(80, 100)  # become darker.
            img_1 = cv2.cvtColor(color_converted, cv2.COLOR_HSV2BGR)
        else:
            img_1 = (np.ones((self.h, self.w, 3)) * np.array(RGB2)).astype(np.uint8)

        # combine together.
        alpha = self.gen_wave_alpha()
        mask_img = (alpha * img_1 + (1.0 - alpha) * img).astype(np.uint8)
        return mask_img

    def jp_bg(self, RGB1, RGB2=None):
        """
        :param RGB1: numpy of shape 1 1 3
        :param RGB2: optional, numpy of shape 1 1 3
        :return: np.array of shape [H, W, 3]
        """
        bg = np.ones((self.h, self.w, 3)) * np.array(RGB1)
        if RGB2 is None:
            RGB2 = self.rgb2hsv2rgb(RGB1)
        # generate strips.
        times = random.randint(20, 40)
        rad = 360 / times / 2
        x, y = random.randrange(-400, -100), random.randrange(-200, 1000)
        for i in range(times):
            rotate = i * rad * 2
            img = cv2.ellipse(bg, (x, y), (15000, 15000), rotate, 0, rad, RGB2.squeeze().tolist(), -1)
        img = img.astype(np.uint8)

        color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # shift color.
        color_converted[:, :, 1] += 50
        color_converted[:, :, 0] += random.randint(0, 10)
        img_1 = cv2.cvtColor(color_converted, cv2.COLOR_HSV2BGR)

        # combine together.
        alpha = self.gen_alpha()
        mask_img = (alpha * img_1 + (1.0 - alpha) * img).astype(np.uint8)
        return mask_img

    def point_bg(self, RGB1, RGB2=None):
        """
        :param RGB1: numpy of shape 1 1 3
        :param RGB2: optional, numpy of shape 1 1 3
        :return: np.array of shape [H, W, 3]
        """
        bg = np.ones((self.h, self.w, 3)) * np.array(RGB1)
        if RGB2 is None and True:
            RGB2 = self.rgb2hsv2rgb(RGB1)
        img_1 = np.ones_like(bg) * RGB2

        def gen_gaussian_kernel(side, r=100, sigma=-1):
            gaussian_kernel = cv2.getGaussianKernel(side, sigma) * cv2.getGaussianKernel(side, sigma).T
            gaussian_kernel = gaussian_kernel * 255 * random.randrange(r/10, r)
            gaussian_kernel[gaussian_kernel > 1] = 1
            return gaussian_kernel

        def gen_random_point(H, W, side):
            x, y = random.randint(side, H - side), random.randint(side, W - side)
            sigma = 0
            gaussian_kernel = gen_gaussian_kernel(side, self.w, sigma)
            x0, x1, y0, y1 = int(x - side / 2), int(x + side / 2), int(y - side / 2), int(y + side / 2)
            return gaussian_kernel, slice(x0, x1), slice(y0, y1)

        side = int(self.h / 5)
        alpha = np.zeros((self.h + 2 * side, self.w + 2 * side))

        num_kernel = 0
        N = random.randrange(15, 30)
        while num_kernel < N:
            gaussian_kernel, sx, sy = gen_random_point(self.h + 2 * side, self.w + 2 * side, side)
            if np.all(alpha[sx, sy] == 0):
                alpha[sx, sy] = np.maximum(gaussian_kernel, alpha[sx, sy])
                num_kernel += 1
        alpha = alpha[side: side + self.h, side: side + self.w][..., np.newaxis]
        mask_img = (alpha * img_1 + (1 - alpha) * bg).astype(np.uint8)
        return mask_img


    def gen_wave_alpha(self):
        """:return: np.array of shape h, w, 3"""
        return self.gen_alpha(wave=True)

    def gen_alpha(self, wave=False):
        """
        Generate an alpha mask.
        :param wave: boolean. if alpha in wave style
        :return: np.array of shape h, w, 3
        """
        revert = random.randint(0, 1)
        shift = random.randint(0, self.w / 2.0)
        mag = random.randrange(20) * 1.0 / 100.0 + 0.9
        alpha = np.ones((self.h, self.w, 3))
        the1 = 1.0 * random.randrange(5, 10) / self.h
        the2 = 1.0 * random.randrange(5, 10) / self.w
        for i in range(self.w):
            alpha[:, i, :] *= np.abs(revert - max(min(mag * (i - shift) / self.w, 1.), 0.0))
            if wave:
                x = int(np.ceil(0.5 * (1 + 0.5 * (np.sin(i * the2) + np.sin(i * the1))) * self.h))
                alpha[x:, i, :] = 1 - alpha[x:, i, :]
        return alpha

    @staticmethod
    def rgb2hsv2rgb(rgb):
        """transfer RGB to HSV space and shift color and transfer back."""
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] - 30
        # hsv[:, :, 0] = hsv[:, :, 0] + random.randint(10, 10)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb


class BGDecoTony:
    """
    __doc_str__ : HOW TO USE
    """
    def __init__(self, h: int, w: int):
        """
        :param h: height of bg
        :param w: width of bg
        """
        self.h = h
        self.w = w
        self.all = []


if __name__ == '__main__':
    DEBUG = True
    RGB = np.array([[[255, 160, 100]]])
    RGB2 = None
    # RGB2 = np.array([[[90, 60, 200]]])
    H, W = 1080, 1920
    tonyer = BGGenTony(H, W)
    decolar = BGDecoTony(H, W)
    print(help(BGGenTony))
    for f_name in tonyer.all:
        rgb2 = None if random.random() > 0.5 else RGB2
        index = 100 + len(tonyer.all)*10 + tonyer.all.index(f_name) + 1
        plt.imshow(tonyer.__getattribute__(f_name)(RGB, rgb2))
        plt.show()
    # rgb2hsv(RGB)
