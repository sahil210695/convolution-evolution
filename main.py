import numpy as np
from PIL import Image


class Conv2D:
    def __init__(self, padding=0, stride=1):
        self.img = ""
        self.img_x = 0
        self.img_y = 0
        self.padding = padding
        self.stride = stride
        self.out_img = ""
        self.out_img_x = 0
        self.out_img_y = 0
        self.kernel = ""
        self.kernel_x = 0
        self.kernel_y = 0

    def set_input_image(self, image_path):
        img = Image.open(image_path).convert("L")
        self.img = np.asarray(img)
        self.img_x, self.img_y = self.img.shape

    def set_conv_kernel(self, kernel):
        self.kernel = kernel
        self.kernel_x, self.kernel_y = self.kernel.shape

    def set_output_image(self):
        self.out_img_x = ((self.img_x - self.kernel_x + 2 * self.padding) // self.stride) + 1
        self.out_img_y = ((self.img_y - self.kernel_y + 2 * self.padding) // self.stride) + 1

        self.out_image = np.zeros((self.out_img_x, self.out_img_y))

    def forward_pass(self):
        for ix in range(self.img_x):
            if ix > self.img_x - self.kernel_x: break

            for iy in range(self.img_y):
                if iy > self.img_y - self.kernel_y: break

                self.out_image[ix, iy] = np.sum(
                    self.kernel *
                    self.img[ix:ix + self.kernel_x, iy:iy + self.kernel_y])

    def process(self, image_path, kernel):
        self.set_input_image(image_path)
        self.set_conv_kernel(kernel)
        self.set_output_image()
        self.forward_pass()

        return self.out_image
