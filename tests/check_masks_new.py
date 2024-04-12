

# Set index
image_index = 12
brightness_factor=2

####################################

colour_map = {
    0: (15,82,186),
    1: (80,200,120),
    2: (128,0,128),
    3: (224,17,95),
    4: (0,0,128),
    5: (145, 149, 246),
    6: (249, 240, 122),
    7: (251, 136, 180),
    8: (244,96,54),
    9: (49,73,94),
    10: (138,145,188),
    11: (137,147,124)
}

# To run
import numpy as np
import matplotlib.pyplot as plt

brightness_factor=2

img_0 = img_raw
img_0_np = np.flip((img_0.to('cpu').numpy() * 255)[1:4, :, :], 0)

img_0_np_bright = (img_0_np * brightness_factor).clip(0, 255).astype(int)

plt.imshow(np.transpose(img_0_np_bright, (1, 2, 0)))
plt.show()


img_0 = img
img_0_np = np.flip((img_0.to('cpu').numpy() * 255)[1:4, :, :], 0)

img_0_np_bright = (img_0_np * brightness_factor).clip(0, 255).astype(int)

plt.imshow(np.transpose(img_0_np_bright, (1, 2, 0)))
plt.show()

img_0 = self.to_tensor(np.transpose(sample_original[3, :, :, :], (1, 2, 0)))
img_0_np = np.flip((img_0.to('cpu').numpy() * 255)[1:4, :, :], 0)

img_0_np_bright = (img_0_np * brightness_factor).clip(0, 255).astype(int)

plt.imshow(np.transpose(img_0_np_bright, (1, 2, 0)))
plt.show()






mask = masks_1[image_index]
mask_index = mask.to('cpu').argmax(0)

img_0_np_bright_copy = img_0_np_bright.copy()
img_rgb_first_channel = img_0_np_bright[0, :, :].copy()
img_rgb_second_channel = img_0_np_bright[1, :, :].copy()
img_rgb_third_channel = img_0_np_bright[2, :, :].copy()

for cluster_id, rgb_values in colour_map.items():
    bool_mask = (mask_index == cluster_id)
    img_rgb_first_channel[bool_mask] = rgb_values[0]
    img_rgb_second_channel[bool_mask] = rgb_values[1]
    img_rgb_third_channel[bool_mask] = rgb_values[2]

coloured_mask = np.concatenate([
    img_rgb_first_channel.reshape(-1, 224, 224),
    img_rgb_second_channel.reshape(-1, 224, 224),
    img_rgb_third_channel.reshape(-1, 224, 224)
], 0)

# Combine the bright image with the coloured mask
coloured_image = ((img_0_np_bright_copy + coloured_mask) / 2).astype(int)

plt.imshow(np.transpose(coloured_image, (1, 2, 0)))
plt.show()

####################################


# brightness_factor=2
colour_map = {
    0: (15,82,186),
    1: (80,200,120),
    2: (128,0,128),
    3: (224,17,95),
    4: (0,0,128),
    5: (145, 149, 246),
    6: (249, 240, 122),
    7: (251, 136, 180),
    8: (244,96,54),
    9: (49,73,94),
    10: (138,145,188),
    11: (137,147,124)
}

# To run
import numpy as np
import matplotlib.pyplot as plt

img_0 = x_aug2[image_index]
img_0_np = np.flip((img_0.to('cpu').numpy() * 255)[1:4, :, :], 0)

img_0_np_bright = (img_0_np * brightness_factor).clip(0, 255).astype(int)

plt.imshow(np.transpose(img_0_np_bright, (1, 2, 0)))
plt.show()

mask = masks_2[image_index]
mask_index = mask.to('cpu').argmax(0)

img_0_np_bright_copy = img_0_np_bright.copy()
img_rgb_first_channel = img_0_np_bright[0, :, :].copy()
img_rgb_second_channel = img_0_np_bright[1, :, :].copy()
img_rgb_third_channel = img_0_np_bright[2, :, :].copy()

for cluster_id, rgb_values in colour_map.items():
    bool_mask = (mask_index == cluster_id)
    img_rgb_first_channel[bool_mask] = rgb_values[0]
    img_rgb_second_channel[bool_mask] = rgb_values[1]
    img_rgb_third_channel[bool_mask] = rgb_values[2]

coloured_mask = np.concatenate([
    img_rgb_first_channel.reshape(-1, 224, 224),
    img_rgb_second_channel.reshape(-1, 224, 224),
    img_rgb_third_channel.reshape(-1, 224, 224)
], 0)

# Combine the bright image with the coloured mask
coloured_image = ((img_0_np_bright_copy + coloured_mask) / 2).astype(int)

plt.imshow(np.transpose(coloured_image, (1, 2, 0)))
plt.show()

