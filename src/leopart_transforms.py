import random
import torch
import torchvision

from PIL import ImageFilter, Image
from typing import List, Tuple, Dict
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
import numpy as np
from cvtorchvision import cvtransforms
import cv2
from torchvision.transforms.functional import adjust_hue


# class GaussianBlur:
#     """
#     Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709 following
#     https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/src/multicropdataset.py#L64
#     """
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x: Image):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x


# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         #return x
#         # return cv2.GaussianBlur(x,(0,0),sigma)
#         return cv2.GaussianBlur(x,(0,0),sigma)
#
#
# class RandomBrightness(object):
#     """ Random Brightness """
#
#     def __init__(self, brightness=0.4):
#         self.brightness = brightness
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
#         img = sample * s
#
#         return img.astype(np.uint8).clip(0, 255)
#
# class RandomContrast(object):
#     """ Random Contrast """
#
#     def __init__(self, contrast=0.4):
#         self.contrast = contrast
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
#         mean = np.mean(sample, axis=(0, 1))
#
#         return ((sample - mean) * s + mean).astype(np.uint8).clip(0, 255)
#
# class RandomSaturation(object):
#     """ Random Contrast """
#
#     def __init__(self, saturation=0.4):
#         self.saturation = saturation
#
#     def __call__(self, sample):
#         s = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
#         mean = np.expand_dims(sample.mean(-1), -1)
#         return ((sample - mean) * s + mean).astype(np.uint8).clip(0, 255)
#
#
# class RandomHue(object):
#     """ Random Contrast """
#     def __init__(self, hue=0.1):
#         self.hue = hue
#
#     def __call__(self, sample):
#         sample[:, :, 1:4] = np.flip(np.array(
#             adjust_hue(Image.fromarray(np.flip(sample[:, :, 1:4], 2)), hue_factor=self.hue)
#         ), 2)
#         return sample.astype(np.uint8).clip(0, 255)
#
# class ToGray(object):
#     def __init__(self, out_channels):
#         self.out_channels = out_channels
#     def __call__(self,sample):
#         # sample_np = sample.permute(1, 2, 0).numpy()
#         sample_np = np.transpose(sample, (1, 2, 0))
#         gray_img = np.mean(sample_np, axis=-1)
#         gray_img = np.tile(gray_img, (self.out_channels, 1, 1))
#         return gray_img
#         # gray_img = np.transpose(gray_img, [1, 2, 0])
#         # return gray_img.astype(np.uint8)


class RandomBrightness(object):
    """ Random Brightness """

    def __init__(self, brightness=0.4):
        self.brightness = brightness

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        img = sample * s

        return img.clip(0, 1)

class RandomContrast(object):
    """ Random Contrast """

    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = sample.mean(axis=(2, 3))[:, :, None, None]
        return ((sample - mean) * s + mean).clip(0, 1)


class RandomSaturation(object):
    """ Random Contrast """

    def __init__(self, saturation=0.4):
        self.saturation = saturation

    def __call__(self, sample):

        s = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        mean = sample.mean(axis=1)[:, None, :, :]
        return ((sample - mean) * s + mean).clip(0, 1)


class RandomHue(object):
    """ Random Contrast """
    def __init__(self, hue=0.1):
        self.hue = hue

    def __call__(self, sample):
        rgb_channels = sample[:, 1:4, :, :].flip(1)
        h = np.random.uniform(0 - self.hue, self.hue)
        rgb_channels_hue_mod = adjust_hue(rgb_channels, hue_factor=h)
        sample[:, 1:4, :, :] = rgb_channels_hue_mod.flip(1)
        return sample

class ToGray(object):
    def __init__(self, out_channels):
        self.out_channels = out_channels
    def __call__(self,sample):
        nc = sample.shape[1]
        return sample.mean(axis=1)[:, None, :, :].expand(-1, nc, -1, -1)

class RandomChannelDrop(object):
    """ Random Channel Drop """

    def __init__(self, min_n_drop=1, max_n_drop=8):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, sample):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

        for c in channels:
            sample[c, :, :] = 0
        return sample


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
        self.transform = None

    def __call__(self, x):
        if self.transform is None:
            img_size = x.shape[-1]
            kernel_size = int(img_size * 0.1)
            # Make kernel size odd
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            self.transform = torchvision.transforms.GaussianBlur(kernel_size, self.sigma)
        return self.transform(x)



class Solarize(object):

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        x1 = x.clone()
        one = torch.ones(x.shape, device='cuda')
        bool_check = x > self.threshold
        x1[bool_check] = one[bool_check] - x[bool_check]
        return x1



class LeopartTransforms:

    def __init__(self,
                 size_crops: List[int],
                 nmb_crops: List[int],
                 min_scale_crops: List[float],
                 max_scale_crops: List[float],
                 jitter_strength: float = 0.2,
                 min_intersection: float = 0.01,
                 blur_strength: float = 1):
        """
        Main transform used for fine-tuning with Leopart. Implements multi-crop and calculates the corresponding
        crop bounding boxes for each crop-pair.
        :param size_crops: size of global and local crop
        :param nmb_crops: number of global and local crop
        :param min_scale_crops: the lower bound for the random area of the global and local crops before resizing
        :param max_scale_crops: the upper bound for the random area of the global and local crops before resizing
        :param jitter_strength: the strength of jittering for brightness, contrast, saturation and hue
        :param min_intersection: minimum percentage of intersection of image ares for two sampled crops from the
        same picture should have. This makes sure that we can always calculate a loss for each pair of
        global and local crops.
        :param blur_strength: the maximum standard deviation of the Gaussian kernel
        """
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        assert 0 < min_intersection < 1
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops
        self.min_intersection = min_intersection
        self.interpolation_mode_map = {'BILINEAR': InterpolationMode.BILINEAR}

        # self.color_jitter = cvtransforms.ColorJitter(
        #     0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength,
        #     0.2 * jitter_strength
        # )

        self.color_jitter = cvtransforms.RandomApply([
            RandomBrightness(0.8),
            RandomContrast(0.8),
            RandomSaturation(0.8),
            RandomHue(0.2)
        ], p=0.8)

        # # Construct color transforms
        # self.color_jitter = torchvision.transforms.ColorJitter(
        #     0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength,
        #     0.2 * jitter_strength
        # )

        # color_transform = [torchvision.transforms.RandomApply([self.color_jitter], p=0.8),
        #                    torchvision.transforms.RandomGrayscale(p=0.2)]

        color_transform = [self.color_jitter,
                           cvtransforms.RandomApply([ToGray(13)], p=0.2)]

        blur = GaussianBlur(sigma=[blur_strength * .1, blur_strength * 2.])

        # color_transform.append(torchvision.transforms.RandomApply([blur], p=0.5))
        color_transform.append(cvtransforms.RandomApply([blur], p=0.5))
        self.color_transform = cvtransforms.Compose(color_transform)
        self.to_tensor = cvtransforms.ToTensor()
        # self.color_transform = torchvision.transforms.Compose(color_transform)

        # Construct final transforms
        # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.final_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

        # self.final_transform = cvtransforms.Compose([cvtransforms.ToTensor()])

        # Construct randomly resized crops transforms
        self.rrc_transforms = []
        for i in range(len(self.size_crops)):
            random_resized_crop = cvtransforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )
            # random_resized_crop = torchvision.transforms.RandomResizedCrop(
            #     self.size_crops[i],
            #     scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            # )
            self.rrc_transforms.extend([random_resized_crop] * self.nmb_crops[i])

    def __call__(self, sample: torch.Tensor) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        This method creates the global and local crops, enforcing the intersection requirements and applying a separate
        transform to each crop.
        """
        multi_crops = []
        crop_bboxes = torch.zeros(len(self.rrc_transforms), 4)
        sample_original = sample.copy()
        for i, rrc_transform in enumerate(self.rrc_transforms):
            # Reshape and change image data type
            sample = self.to_tensor(np.transpose(sample_original[np.random.randint(0, 4), :, :, :], (1, 2, 0)))
            # Get the minimum intersection for the crops
            sample_intersection_calc = int((sample.shape[1] * sample.shape[2]) * self.min_intersection)
            if i:
                # Check whether crop has min overlap with existing global crops. If not resample.
                while True:
                    # Get a candidate bounding box
                    candidate_bbox = self.get_candidate_bbox(sample, rrc_transform)

                    # Get the intersection of the candidate with the current global crops.
                    inter = self.calculate_bbox_intersection(candidate_bbox, crop_bboxes, i)

                    # Set the minimum intersection to enforce
                    if i < self.nmb_crops[0]:
                        # Global crops should have at least twice the min_intersection with each other
                        min_intersection = 2 * sample_intersection_calc
                    else:
                        # set min intersection to at least 1% of image area
                        min_intersection = sample_intersection_calc

                    if torch.all(inter > min_intersection):
                        break
            else:
                candidate_bbox = self.get_candidate_bbox(sample, rrc_transform)


            # Store bounding box coordinates
            crop_bboxes[i] = candidate_bbox

            # Crop image
            img_crop = F.resized_crop(
                img=sample,
                top=int(candidate_bbox[0]),
                left=int(candidate_bbox[1]),
                height=int(candidate_bbox[2] - candidate_bbox[0]),
                width=int(candidate_bbox[3] - candidate_bbox[1]),
                size=rrc_transform.size,
                interpolation=self.interpolation_mode_map[rrc_transform.interpolation]
            )


            # Apply color transforms
            # img = torch.from_numpy(self.color_transform(img.numpy()))
            #
            # img = img.float() / 255

            # img_raw = img.clone()

            # Apply colour transformations to crop
            img_crop = self.color_transform(img_crop.unsqueeze(0)).squeeze()

            # Apply final transform
            # img = self.final_transform(img)
            multi_crops.append(img_crop)

        # Calculate relative bboxes for each crop pair from absolute bboxes
        gc_bboxes, otc_bboxes = self.calculate_bboxes(crop_bboxes)

        return multi_crops, {"gc": gc_bboxes, "all": otc_bboxes}

    def calculate_bbox_intersection(self, candidate_bbox, crop_bboxes, crop_index):
        # Calculate intersection between sampled crop and all sampled global crops
        current_gc_count = min(crop_index, self.nmb_crops[0])
        left_top = torch.max(candidate_bbox[None, None, :2], crop_bboxes[:current_gc_count, :2])
        right_bottom = torch.min(candidate_bbox[None, None, 2:], crop_bboxes[:current_gc_count, 2:])
        wh = _upcast(right_bottom - left_top).clamp(min=0)
        return wh[:, :, 0] * wh[:, :, 1]

    def get_candidate_bbox(self, sample, rrc_transform):
        y1, x1, h, w = rrc_transform.get_params(sample.permute(1, 2, 0), rrc_transform.scale, rrc_transform.ratio)
        return torch.Tensor([x1, y1, x1 + w, y1 + h])

    def calculate_bboxes(self, crop_bboxes: Tensor):
        # 1. Calculate two intersection bboxes for each global crop - other crop pair
        gc_bboxes = crop_bboxes[:self.nmb_crops[0]]
        left_top = torch.max(gc_bboxes[:, None, :2], crop_bboxes[:, :2])  # [nmb_crops[0], sum(nmb_crops), 2]
        right_bottom = torch.min(gc_bboxes[:, None, 2:], crop_bboxes[:, 2:])  # [nmb_crops[0], sum(nmb_crops), 2]
        # Testing for non-intersecting crops. This should always be true, just as safe-guard.
        assert torch.all((right_bottom - left_top) > 0)

        # 2. Scale intersection bbox with crop size
        # Extract height and width of all crop bounding boxes. Each row contains h and w of a crop.
        heights = crop_bboxes[:, 2] - crop_bboxes[:, 0]
        widths = crop_bboxes[:, 3] - crop_bboxes[:, 1]
        ws_hs = torch.stack((heights, widths)).T[:, None]

        # Stack global crop sizes for each bbox dimension
        temp_tensor = torch.repeat_interleave(torch.Tensor([self.size_crops[0]]), self.nmb_crops[0] * 2)
        crops_sizes = temp_tensor.reshape(self.nmb_crops[0], 2)
        if len(self.size_crops) == 2:
            temp_tensor = torch.repeat_interleave(torch.Tensor([self.size_crops[1]]), self.nmb_crops[1] * 2)
            crops_sizes = torch.cat((crops_sizes, temp_tensor.reshape(self.nmb_crops[1], 2)))[:, None]  # [sum(nmb_crops), 1, 2]

        # Calculate x1s and y1s of each crop bbox
        x1s_y1s = crop_bboxes[:, None, :2]

        # Scale top left and right bottom points by percentage of width and height covered
        left_top_scaled_gc = crops_sizes[:2] * ((left_top - x1s_y1s[:2]) / ws_hs[:2])
        right_bottom_scaled_gc = crops_sizes[:2] * ((right_bottom - x1s_y1s[:2]) / ws_hs[:2])
        left_top_otc_points_per_gc = torch.stack([left_top[i] for i in range(self.nmb_crops[0])], dim=1)
        right_bottom_otc_points_per_gc = torch.stack([right_bottom[i] for i in range(self.nmb_crops[0])], dim=1)
        left_top_scaled_otc = crops_sizes * ((left_top_otc_points_per_gc - x1s_y1s) / ws_hs)
        right_bottom_scaled_otc = crops_sizes * ((right_bottom_otc_points_per_gc - x1s_y1s) / ws_hs)

        # 3. Construct bboxes in x1, y1, x2, y2 format from left top and right bottom points
        gc_bboxes = torch.cat((left_top_scaled_gc, right_bottom_scaled_gc), dim=2)
        otc_bboxes = torch.cat((left_top_scaled_otc, right_bottom_scaled_otc), dim=2)
        return gc_bboxes, otc_bboxes


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
