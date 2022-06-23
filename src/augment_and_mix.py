# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
from augmentations import augmentations, augmentations_all
import albumentations
import numpy as np
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# def normalize(image, mean: np.array, std: np.array):
#     """Normalize input image channel-wise to zero mean and unit variance."""
#     image = image.transpose(2, 0, 1)  # Switch to channel-first
#     image = (image - mean[:, None, None]) / std[:, None, None]
#     return image.transpose(1, 2, 0)

def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)

def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.


def dual_augment_and_mix(
        image, mask, transforms, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
      image: Raw input image(0-255) as uint8 `np.ndarray` of shape (h, w, c)
      mask: Onehot encoded mask as `np.ndarray` of shape (h, w, c)
      transforms: albumentations augmentations as `list`
                  with parameterized intensity
      width: Width of augmentation chain
      depth: Depth of augmentation chain.
             -1 enables stochastic depth uniformly from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
      image_mixed: Augmented and mixed image.
      mask_mixed: Augmented and mixed mask.
    """

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    image_mix = np.zeros_like(image)
    mask_mix = np.zeros_like(mask, dtype=np.float)

    for i in range(width):
        image_aug, mask_aug = image.copy(), mask.copy()

        # decide augmentations of depth at random
        depth = np.random.randint(1, depth + 1 if depth > 0 else 4)
        op_depth = [np.random.choice(transforms) for _ in range(depth)]

        # apply augmentation
        augmentations = albumentations.Compose(op_depth, p=1)
        sample = augmentations(image=image_aug, mask=mask_aug)
        image_aug, mask_aug = sample['image'], sample['mask']

        # Preprocessing commutes since all coefficients are convex
        image_mix = image_mix + ws[i] * normalize(image_aug)
        mask_mix = mask_mix + ws[i] * mask_aug

    # add augmentated image with normalization
    image_mixed = (1 - m) * normalize(image) + m * image_mix
    mask_mixed = (1 - m) * mask + m * mask_mix

    # denormalization
    # image_mixed = np.clip(image_mixed * 255, 0, 255).astype(np.uint8)
    # mask_mixed = np.clip(mask_mixed, 0, 1)  # 色変換系で1以上になったピクセルを1にクリップ
    return image_mixed, mask_mixed


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1., augall=False):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10)
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
      mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    image_mix = np.zeros_like(image)

    for i in range(width):
        image_aug = image.copy()

        depth = np.random.randint(1, depth + 1 if depth > 0 else 4)
        for _ in range(depth):
            op = np.random.choice(
                augmentations_all if augall else augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        image_mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * normalize(image) + m * image_mix
    return mixed