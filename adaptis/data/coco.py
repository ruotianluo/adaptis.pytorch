import cv2
import json
import numpy as np
from pathlib import Path
from .base import BaseDataset


class COCODataset(BaseDataset):
    def __init__(self, dataset_path, split='train', use_jpeg=False, **kwargs):
        super(COCODataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split

        images_path = self.dataset_path / 'images' / f'{split}2017'
        gt_path = self.dataset_path / 'annotations' / f'panoptic_{split}2017'
        annotations_path = self.dataset_path / 'annotations' / f'panoptic_{split}2017.json'
        
        annotations = json.load(open(annotations_path, 'r'))
        annotations['annotations'] = {_['image_id']: _ for _ in annotations['annotations']}

        self.dataset_samples = []
        for image in annotations['images']:
            image_path = str(images_path / image['file_name'])
            panoptic_path = str(gt_path / image['file_name'].replace('.jpg', '.png'))

            self.dataset_samples.append((str(image_path), panoptic_path, annotations['annotations'][image['id']]))

        total_classes = max([_['id'] for _ in annotations['categories']]) + 1
        self._stuff_labels = [_['id'] for _ in annotations['categories'] if _['isthing'] == 0]
        self._things_labels = [_['id'] for _ in annotations['categories'] if _['isthing'] == 1]
        self._semseg_mapping = np.ones(total_classes, dtype=np.int32)
        for i in range(total_classes):
            self._semseg_mapping[i] = self.from_dataset_mapping.get(i, -1)

    def get_sample(self, index):
        image_path, panoptic_path, ann = self.dataset_samples[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_map = cv2.imread(panoptic_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        instance_map = instance_map[..., 0] * 256 ** 2 + instance_map[..., 1] * 256 + instance_map[..., 2]
        
        label_map = np.zeros_like(instance_map).astype(np.int32) - 1

        instances_info = dict()

        for segments in ann['segments_info']:
            obj_id = segments['id']
            class_id = segments['category_id']
            mapped_class_id = self._semseg_mapping[class_id]

            isthing = class_id in self.things_labels
            iscrowd = segments['iscrowd'] == 1
            ignore = mapped_class_id == -1 or iscrowd

            label_map[instance_map == obj_id] = mapped_class_id

            if iscrowd or not isthing:
                instance_map[instance_map == obj_id] = 0

            if isthing:
                instances_info[obj_id] = {
                    'class_id': mapped_class_id, 'ignore': ignore
                }

        sample = {
            'image': image,
            'instances_mask': instance_map,
            'instances_info': instances_info,
            'semantic_segmentation': label_map
        }

        return sample

    @property
    def stuff_labels(self):
        return self._stuff_labels

    @property
    def things_labels(self):
        return self._things_labels
