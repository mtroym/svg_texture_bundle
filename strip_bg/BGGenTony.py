import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations import GridDistortion

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
    for style in tonyer.released:
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
        self.all = ['sea_bg', 'jp_bg', 'combine_bg']
        self.released = ['sea_bg', 'jp_bg', 'comb_bg']
        self.declor = BGDecTony(h, w)

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
            # color_converted[:, :, 0] += random.randrange(90)
            if random.random() > 0.3:
                color_converted[:, :, 1] += random.randint(50, 100)  # became brighter.
            else:
                color_converted[:, :, 1] -= random.randint(80, 100)  # become darker.
            img_1 = cv2.cvtColor(color_converted, cv2.COLOR_HSV2BGR)
        else:
            img_1 = (np.ones((self.h, self.w, 3)) * np.array(RGB2)).astype(np.uint8)

        # combine together.
        # alpha = self.gen_wave_alpha()
        alpha = self.gen_alpha(wave=random.random() > 0.8)
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
            gaussian_kernel = gaussian_kernel * 255 * random.randrange(r / 10, r)
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

    def combine_bg(self, RGB, RGB2=None):
        bg = self.sea_bg(RGB, None)
        watermark = self.declor.gen_watermark(disable='random')
        alpha = self.declor.gen_mask() / 255.0
        canvas = self.declor.gen_canvas()
        coverted = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float64)
        coverted[:, :, 1] *= 1.0 * alpha[:, :, 0]
        coverted = coverted.astype(np.uint8)
        bg1 = cv2.cvtColor(coverted, cv2.COLOR_HSV2BGR)
        bg1[canvas == 0] = 0
        coverted = cv2.cvtColor(bg1, cv2.COLOR_BGR2HSV).astype(np.float64)
        coverted[watermark == 255, 1] -= 40
        coverted = np.maximum(coverted, 0)
        coverted = coverted.astype(np.uint8)
        bg1 = cv2.cvtColor(coverted, cv2.COLOR_HSV2BGR).astype(np.uint8)
        return bg1


class BGDecTony:
    """
    __doc_str__ : HOW TO USE
    img = decagent.XXdec(img, **kargs)
    """

    def __init__(self, h: int, w: int):
        """
        :param h: height of bg
        :param w: width of bg
        """
        self.h = h
        self.w = w
        self.all = []

    def gen_watermark(self, disable='random'):
        """
        Generate water mark. from sucai library.
        // TODO: WATERMARKS' POSITION NEED UPDATE to RANDOM!
        :param disable: if random, have 0.5 possibility to enable. if yes. then disabled.
        :return: np.array of shape (h, w) type: np.uint8
        """
        root = 'dec'
        watermark = np.ones((self.h, self.w))
        if disable == 'yes':
            return watermark
        elif disable == 'random':
            if random.random() > 0.5:
                return watermark
        # root = '...Like'
        path = os.path.join(root, 'pointball{}.png'.format(random.randint(0, 5)))
        element = cv2.imread(path, -1)
        watermark = np.ones((self.h, self.w))
        mask = element[:, :, 3]
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        # ruihua
        mask[mask > 128] = 255
        mask[mask <= 128] = 0
        mask = cv2.resize(mask, (self.h, self.h), cv2.INTER_CUBIC)
        eleH, eleW = mask.shape[:2]
        # center = (random.randint(0, int(W)),
        #           random.randint(H * i - i * int(H / 100), (i == 1) * (H - int(H / 100)) + (i == 0) * int(H / 4)))
        center = (int(self.w * 2 / 3), int(self.h * 5 / 6))
        # gen slice of origin pic.
        x1 = int(max(0, center[0] - int(eleW / 2)))
        x2 = int(min(self.w, center[0] + int(eleW / 2)))
        y1 = int(max(0, center[1] - int(eleH / 2)))
        y2 = int(min(self.h, center[1] + int(eleH / 2)))
        # print(center)
        # print(x1, x2, y1, y2)
        cir_slice = mask[
                    int(y1 + int(eleH / 2) - center[1]):int(y2 + int(eleH / 2) - center[1]),
                    int(x1 + int(eleW / 2) - center[0]):int(x2 + int(eleW / 2) - center[0])]
        # print(cir_slice.shape)
        #     plt.imshow(cir_slice)
        #     plt.show()
        watermark[y1:y2, x1:x2] = cir_slice
        return watermark

    def gen_canvas(self):
        """
        generate strip circle. many params TBD/
        :return: np.array of shape (h, w, 3)
        """
        H = self.h
        W = self.w

        def gen_circle(outR=200, angle=-45, w=2):
            """
            generate a strip circle.
            :param outR: output Radius.
            :param angle: the rotation angle.
            :param w: the strip width.
            :return: np.array of shape (outR, outR, 3)
            """
            eleR = outR
            num = 30
            step = int(eleR / num)
            halfR = int(eleR / 2)
            circle = (np.ones((eleR, eleR, 3)) * 255).astype(np.uint8)
            cv2.circle(circle, (halfR, halfR), halfR, (0, 0, 0), -1)
            strip = (np.ones((eleR, eleR, 3)) * 255).astype(np.uint8)
            for i in range(int(eleR / step)):
                cv2.line(strip, (0 + i * step, 0), (0 + i * step, eleR), (0, 0, 0), w)
            rotateMatrix = cv2.getRotationMatrix2D(center=(strip.shape[1] / 2, strip.shape[0] / 2), angle=angle,
                                                   scale=1.2)
            rot_strip = cv2.warpAffine(strip, rotateMatrix, (strip.shape[1], strip.shape[0]))

            rot_strip[circle != 0] = 255
            circle[strip != 0] = 255

            return rot_strip

        R = random.randint(int(self.h / 3 * 2), int(self.h / 5 * 4))
        cir = gen_circle(outR=R, angle=random.randint(15, 60), w=3)
        eleH, eleW = cir.shape[:2]
        canvas = (np.ones((H, W, 3)) * 255).astype(np.uint8)
        for i in range(1):
            center = (random.randint(0, int(self.h / 5 * 1)),
                      random.randint(H * i - i * int(H / 100),
                                     (i == 1) * (H - int(H / 100)) + (i == 0) * int(H / 4)))
            # gen slice of origin pic
            x1 = int(max(0, center[0] - eleW / 2))
            x2 = int(min(W, center[0] + eleW / 2))
            y1 = int(max(0, center[1] - eleH / 2))
            y2 = int(min(H, center[1] + eleH / 2))
            cir_slice = cir[int(y1 + eleH / 2 - center[1]):int(y2 + eleH / 2 - center[1]),
                            int(x1 + eleW / 2 - center[0]):int(x2 + eleW / 2 - center[0])]
            canvas[y1:y2, x1:x2] = cir_slice
        return canvas

    def gen_mask(self):
        """
        generate mask to transfer color in Saturation. (HSV)
        :return: np.array of shape(h, w, 3) of dtype np.uint8
        """
        H, W = self.h, self.w
        alpha = (np.ones((H, W, 3)) * 255).astype(np.uint8)
        # gen four circles.
        for _ in range(2):
            R = H / 3
            center = (random.randint(int(0 + R), int(W - R)), random.randint(0, H / 6))
            cv2.circle(alpha, center, random.randint(int(R / 3 * 2), R), (0, 0, 0), -1)
            center = (random.randint(int(W / 3 * 2), int(W)), random.randint(int(H / 6 * 5), int(H)))
            cv2.circle(alpha, center, random.randint(int(R / 3 * 2), R), (0, 0, 0), -1)

        # distortion.
        alpha = cv2.GaussianBlur(alpha, (151, 151), 0)
        aug = GridDistortion(p=1)
        alpha1 = aug(image=alpha)['image']
        alpha[alpha < 220] = 64
        alpha[alpha >= 220] = 128
        alpha1[alpha1 < 220] = 64
        alpha1[alpha1 >= 220] = 127
        alpha = alpha1 + alpha

        # add small circle.
        for i in range(2):
            center = (random.randint(int(0), int(W / 5 * 1)),
                      random.randint(int(H / 6 * 5), int(H)))
            r = random.randint(50, 100)
            cv2.circle(alpha, center, r, (100, 100, 100), int(r / 4))
        #     cv2.circle(alpha, center, int(r*0.8), (255,255,255), -1)
        return alpha


if __name__ == '__main__':
    DEBUG = True
    RGB = np.array([[[120, 77, 200]]])
    RGB2 = None
    # RGB2 = np.array([[[90, 60, 200]]])
    H, W = 720, 1280
    tonyer = BGGenTony(H, W)
    decolar = BGDecTony(H, W)
    print(help(BGGenTony))
    for f_name in tonyer.all:
        rgb2 = None if random.random() > 0.9 else RGB2
        index = 100 + len(tonyer.all) * 10 + tonyer.all.index(f_name) + 1
        plt.imshow(tonyer.__getattribute__(f_name)(RGB, rgb2))
        plt.show()
    # rgb2hsv(RGB)
