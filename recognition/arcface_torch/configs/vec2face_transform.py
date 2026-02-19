import random
from PIL import Image
########################################################################
# https://github.com/mk-minchul/AdaFace/blob/master/dataset/augmenter.py
########################################################################
import numpy as np
import cv2
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms


class Augmenter():

    def __init__(self, crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob):
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    def augment(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            sample, crop_ratio = self.crop_augment(sample)

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = self.low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)

        return sample

    def crop_augment(self, sample):
        new = np.zeros_like(np.array(sample))
        if hasattr(F, '_get_image_size'):
            orig_W, orig_H = F._get_image_size(sample)
        else:
            # torchvision 0.11.0 and above
            orig_W, orig_H = F.get_image_size(sample)
        i, j, h, w = self.random_resized_crop.get_params(sample,
                                                         self.random_resized_crop.scale,
                                                         self.random_resized_crop.ratio)
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h,j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))
        crop_ratio = min(h, w) / max(orig_H, orig_W)
        return sample, crop_ratio

    def low_res_augmentation(self, img):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

        return aug_img, side_ratio

    def photometric_augmentation(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample

import random
import torch
from PIL import Image as PILImage

import torchvision.transforms.functional as TF  # v1 functional: to_pil_image
import torchvision.transforms.v2 as transforms_v2

try:
    from torchvision.tv_tensors import Image as TVImage
except Exception:
    TVImage = None


class AdaFaceAugmentTransformV2:
    """
    torchvision.transforms.v2.Compose で使えるラッパ。
    入力が tv_tensors.Image / torch.Tensor / PIL.Image のどれでも受けて、
    内部では PIL に変換して Augmenter を適用し、必要なら tv_tensors.Image に戻す。
    """
    def __init__(
        self,
        p: float = 0.3,
        crop_augmentation_prob: float = 0.2,
        photometric_augmentation_prob: float = 0.2,
        low_res_augmentation_prob: float = 0.2,
    ):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.augmenter = Augmenter(
            crop_augmentation_prob=crop_augmentation_prob,
            photometric_augmentation_prob=photometric_augmentation_prob,
            low_res_augmentation_prob=low_res_augmentation_prob,
        )
        # PIL -> tv_tensors.Image へ戻す用
        self._to_image = transforms_v2.ToImage()

    def __call__(self, img):
        # 入力タイプを覚える（最後に戻す）
        want_tv_image = (TVImage is not None and isinstance(img, TVImage))
        want_tensor = torch.is_tensor(img) and not want_tv_image

        # --- to PIL ---
        if isinstance(img, PILImage.Image):
            pil = img
        elif torch.is_tensor(img):
            # tv_tensors.Image も Tensor 扱いでここに入る（to_pil_imageはOK）
            pil = TF.to_pil_image(img)
        else:
            raise TypeError(f"Unsupported input type: {type(img)}")

        # --- augment ---
        if random.random() < self.p:
            pil = self.augmenter.augment(pil)

        # --- back to original-ish type ---
        if isinstance(img, PILImage.Image):
            return pil

        # v2パイプラインで ToDtype/Normalize につなげるなら tv_tensors.Image に戻すのが安全
        tv = self._to_image(pil)  # -> torchvision.tv_tensors.Image (uint8)
        if want_tv_image:
            return tv
        if want_tensor:
            # もし「純Tensorで返したい」ならここ。通常は不要（v2ならtvでOK）
            return tv.as_subclass(torch.Tensor)

        return tv

    def __repr__(self):
        return (f"{self.__class__.__name__}(p={self.p}, "
                f"crop={self.augmenter.crop_augmentation_prob}, "
                f"photo={self.augmenter.photometric_augmentation_prob}, "
                f"lowres={self.augmenter.low_res_augmentation_prob})")


