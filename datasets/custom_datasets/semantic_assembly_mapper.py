# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import logging
import os

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize
import pycocotools.mask as mask_util
from PIL import Image
import PIL
__all__ = ["AssemblySemanticDatasetMapper"]


class AssemblySemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by OneFormer custom semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        name,
        num_queries,
        meta,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        task_seq_len,
        max_seq_len,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.meta = meta
        self.name = name
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.num_queries = num_queries

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        # adjust meta for tehse class names
        self.class_names = self.meta.stuff_classes

        # defining the tokenizers
        self.text_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=task_seq_len)
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        
        # THIS IS WHERE THE MAIN METHODS FOR FINDING AGUMENTATION AND OTHERWISE ARE CALLED
        # comes from presets in the config files

        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "meta": meta,
            "name": dataset_names[0],
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    # sends in the unique classes and the dict of the class names
    # returns texts which is used in the architecture
    def _get_texts(self, classes, num_class_obj):
        
        classes = list(np.array(classes))
        texts = ["an semantic photo"] * self.num_queries
        
        # counts the number of objects per class in each binary mask
        for class_id in classes:
            cls_name = self.class_names[class_id]
            num_class_obj[cls_name] += 1
        
        num = 0
        for i, cls_name in enumerate(self.class_names):
            # checks if the number of objects for the class is greater than 0
            # updates different objects of the same class with same input text with "a photo"
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        return texts
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "AssemblyDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # read the image with Detectron Utils
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # raises error if image size is not the same as said in the dict
        utils.check_image_size(dataset_dict, image)

        # converts label to type double (should be uint16 at first), and removes it from the dictionary
        # sem_seg_gt is the label used
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")

            mask = Image.open(dataset_dict.pop("sem_seg_file_name"))

            transform = transforms.Compose ([
                # transforms.Resize(size=(161, 161), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor()
            ])

            mask = transform(mask)

            sem_seg_gt = mask[0, :, :] + mask[1, :, :] + torch.mul(mask[2, :, :], 2)


        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        
        # create an input that can be used with augumentation calls, supplying the image and respective mask
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)

        # same as augumentation list, this is just a but outdated
        # supplying the transforms we want to make, and the input to make those transforms on
        # tfm_gens are the augumentation supplied at top of __init__ -> adjust this for your transforms
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg


        # Pad image and segmentation label here!

        # matching to (C, W, H) -> this is good
        # ascontigarr creates a new array -> new block of memory
        # converts to tensor
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # if sem_seg_gt is not None:
        #     sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long")) # makes sure the segmentation masks are of type long

        # if you want padding, the following gets executed
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        # transformed image and label are stored in the dict here
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            # convert labels to numpy
            sem_seg_gt = sem_seg_gt.numpy()
            # creates list of instances and stores in "fields" such as boxes, masks, etc.
            instances = Instances(image_shape)

            # in my case, this will return something like [0, 1, 2] if transforms have been carried out correctly
            classes = np.unique(sem_seg_gt)
            # remove ignored region -> you can set this when defining mapping and it will igore at evaluration
            classes = classes[classes != self.ignore_label]

            # setting instances to the unique classes for easy access
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            # creates one mask for each class (multiple binary masks)
            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            # apply a 0 mask to image without any masks
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            
            # BitMasks stores segmentation masks for all objects in one image, in bitmaps
            # stack a copy of all the masks inside a tensor
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            # creates keys for each class name, and sets their values to 0
            num_class_obj = {}
            for name in self.class_names:
                num_class_obj[name] = 0

            task = "The task is semantic"
            text = self._get_texts(instances.gt_classes, num_class_obj)

            dataset_dict["instances"] = instances
            dataset_dict["orig_shape"] = image_shape
            dataset_dict["task"] = task
            dataset_dict["text"] = text
        
        return dataset_dict
