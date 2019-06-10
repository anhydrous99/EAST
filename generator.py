from tensorflow.python.keras.utils.data_utils import Sequence
from data_processor import get_images, get_text_file, load_annotation, check_and_validate_polys, crop_area, pad_image, \
    resize_image, generate_rbox
import numpy as np
import cv2
import os


class EastSequence(Sequence):

    def __init__(self, FLAGS, input_size=512, background_ratio=3. / 8, is_train=True,
                 random_scale=np.array([0.5, 1.0, 2.0, 3.0])):
        self.FLAGS = FLAGS
        self.batch_size = FLAGS.batch_size
        self.geometry = FLAGS.geometry
        self.epoch = 1
        self.input_size = input_size
        self.background_ratio = background_ratio
        self.is_train = is_train
        self.random_scale = random_scale
        self.image_list = np.array(get_images(FLAGS.training_data_path))
        self.index = np.arange(0, self.image_list.shape[0])
        self.length = int(np.ceil(len(self.index) / float(self.batch_size)))
        np.random.shuffle(self.index)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        overly_small_text_region_training_masks = []
        text_region_boundary_training_masks = []
        end_idx = idx + self.batch_size
        end_idx = end_idx if end_idx < len(self.index) else len(self.index)
        for i in self.index[idx:end_idx]:
            im_fn = self.image_list[i]
            im = cv2.imread(im_fn)
            h, w, _ = im.shape
            txt_fn = get_text_file(im_fn)
            if not os.path.exists(txt_fn):
                if not self.FLAGS.suppress_warning_and_error_messages:
                    raise Exception('text file {} does not exists'.format(txt_fn))

            text_polys, text_tags = load_annotation(txt_fn)
            text_polys, text_tags = check_and_validate_polys(self.FLAGS, text_polys, text_tags, (h, w))

            # random scale this image
            rd_scale = np.random.choice(self.random_scale)
            x_scale_variation = np.random.randint(-10, 10) / 100.
            y_scale_variation = np.random.randint(-10, 10) / 100.
            im = cv2.resize(im, dsize=None, fx=rd_scale + x_scale_variation, fy=rd_scale + y_scale_variation)
            text_polys[:, :, 0] *= rd_scale + x_scale_variation
            text_polys[:, :, 1] *= rd_scale + y_scale_variation

            # random crop a area from image
            if np.random.rand() < self.background_ratio:
                # crop background
                im, text_polys, text_tags = crop_area(self.FLAGS, im, text_polys, text_tags, crop_background=True)
                if text_polys.shape[0] > 0:
                    continue
                # pad and resize image
                im, _, _ = pad_image(im, self.input_size, self.is_train)
                im = cv2.resize(im, dsize=(self.input_size, self.input_size))
                score_map = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
                geo_map_channels = 5 if self.geometry == 'RBOX' else 8
                geo_map = np.zeros((self.input_size, self.input_size, geo_map_channels), dtype=np.float32)
                overly_small_text_region_training_mask = np.ones((self.input_size, self.input_size), dtype=np.uint8)
                text_region_boundary_training_mask = np.ones((self.input_size, self.input_size), dtype=np.uint8)
            else:
                im, text_polys, text_tags = crop_area(self.FLAGS, im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    continue
                h, w, _ = im.shape
                im, shift_h, shift_w = pad_image(im, self.input_size, self.is_train)
                im, text_polys = resize_image(im, text_polys, self.input_size, shift_h, shift_w)
                new_h, new_w, _ = im.shape
                score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = \
                    generate_rbox(self.FLAGS, (new_h, new_w), text_polys, text_tags)

            im = (im / 127.5) - 1.
            images.append(im[:, :, ::-1].astype(np.float32))
            image_fns.append(im_fn)
            score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
            geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
            overly_small_text_region_training_masks.append(
                overly_small_text_region_training_mask[::4, ::4, np.newaxis].astype(np.float32)
            )
            text_region_boundary_training_masks.append(
                text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32)
            )
        print(len(images))
        return [np.array(images), np.array(overly_small_text_region_training_masks),
                np.array(text_region_boundary_training_masks), np.array(score_maps)], \
               [np.array(score_maps), np.array(geo_maps)]

    def on_epoch_end(self):
        np.random.shuffle(self.index)
        self.epoch += 1
