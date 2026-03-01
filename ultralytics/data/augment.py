import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0


# TODO: 可能需要一个 BaseTransform 来使所有这些增强都与分类和语义兼容
class BaseTransform:
    """
    用于图像转换的基类.

    这是一个泛型转换类，可以针对特定的图像处理需求进行扩展.
    该类旨在与分类和语义分割任务兼容.

    Methods:
        __init__: 初始化 BaseTransform 对象.
        apply_image: 将图像转换应用于标签.
        apply_instances: 将转换应用于标签中的对象实例.
        apply_semantic: 将语义分割应用于图像.
        __call__: 将所有标签转换应用于图像、实例和语义掩码.
    """

    def __init__(self) -> None:
        """初始化 BaseTransform 对象."""
        pass

    def apply_image(self, labels):
        """将图像转换应用于标签."""
        pass

    def apply_instances(self, labels):
        """将转换应用于标签中的对象实例."""
        pass

    def apply_semantic(self, labels):
        """将语义分割应用于图像."""
        pass

    def __call__(self, labels):
        """将所有标签转换应用于图像、实例和语义掩码."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    """用于组合多个图像转换的类."""

    def __init__(self, transforms):
        """使用转换列表初始化 Compose 对象."""
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """对输入数据应用一系列转换."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """将新转换追加到现有转换列表."""
        self.transforms.append(transform)

    def insert(self, index, transform):
        """将新转换插入到现有转换列表中."""
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """使用索引检索特定转换或一组转换."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """使用索引检索特定转换或一组转换."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """将转换列表转换为标准 Python 列表."""
        return self.transforms

    def __repr__(self):
        """返回对象的字符串表示形式."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """
    基础混音（混合/马赛克）转换类.

    此实现来自 mmyolo.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """使用数据集、pre_transform和概率初始化 BaseMixTransform 对象."""
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """将预处理变换和混合/镶嵌变换应用于标签数据."""
        if random.uniform(0, 1) > self.p:
            return labels

        # 获取一个或三个其他图像的索引
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # 获取图像信息将用于 Mosaic 或 MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # 更新 cls 和文本
        labels = self._update_label_text(labels)
        # 马赛克或混搭
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels):
        """将 MixUp 或 Mosaic 增强应用于标签字典."""
        raise NotImplementedError

    def get_indexes(self):
        """获取用于镶嵌增强的随机索引列表."""
        raise NotImplementedError

    def _update_label_text(self, labels):
        """更新标签文本."""
        if "texts" not in labels:
            return labels

        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """
    马赛克增强.

    此类通过将多个(4 or 9)个图像组合到单个镶嵌图像中来执行镶嵌增强.
    增强应用于具有给定概率的数据集.

    Attributes:
        dataset: 应用镶嵌增强的数据集.
        imgsz (int, optional): 单个图像的马赛克管线后的图像大小（高度和宽度）。默认值为 640。
        p (float, optional): 应用镶嵌增强的概率。必须在 0-1 范围内。默认值为 1.0。
        n (int, optional): 网格大小,4(对于 2x2)或 9(对于 3x3).
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """使用数据集、图像大小、概率和边框初始化对象."""
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # 宽, 高
        self.n = n

    def get_indexes(self, buffer=True):
        """从数据集中返回随机索引列表."""
        if buffer:  # 从缓冲区中选择图像
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # 选择任何图像
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """将混合转换应用于输入图像和标签."""
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  

    def _mosaic3(self, labels):
        """创建 1x3 图像马赛克."""
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # 载入图片
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # 放置 img 到 img3
            if i == 0:  # 中
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # 具有 3 个图块的基本图像
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) 坐标
            elif i == 1:  # 右
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # 左
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # 分配坐标

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # 高度、宽度为下一次迭代

            # 假定 imgsz*2 马赛克尺寸的标签
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """创建 2x2 图像马赛克."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # 马赛克中心 X、Y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # 加载图片
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # 放置 img 到 img4
            if i == 0:  # 左上角
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 具有 4 个图块的基本图像
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # 右上角
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下角
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # 右下角
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels):
        """Create a 3x3 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # 先前的高度、宽度
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # 加载图片
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # 放置 img 到 img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # 具有 4 个图块的基本图像
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) 坐标
            elif i == 1:  # 上
                c = s, s - h, s + w, s
            elif i == 2:  # 右上角
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # 右
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # 右下角
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # 下
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # 左下角
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # 左
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # 左上角
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # 分配坐标

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # 高度、宽度为下一次迭代

            # 假定 imgsz*2 马赛克尺寸的标签
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """更新标签."""
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """返回剪切了镶嵌边框实例的标签."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """用于将 MixUp 增强应用于数据集的类."""

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """使用数据集、pre_transform和应用 MixUp 的概率初始化 MixUp 对象."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """从数据集中获取随机索引."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """根据 https://arxiv.org/pdf/1710.09412.pdf 应用 MixUp 增强。"""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class RandomPerspective:
    """
    对图像和相应的边界框、线段和关键点实现随机透视和仿射变换.
    这些变换包括旋转、平移、缩放和剪切.
    该类还提供选项以指定的概率有条件地应用这些变换.

    Attributes:
        degrees (float): 随机旋转的度数范围.
        translate (float): 随机平移的总宽度和高度的分数.
        scale (float): 比例因子间隔,例如,0.1 的比例因子允许在 90%-110%.
        shear (float): 剪切强度（角度，单位为度）.
        perspective (float): 透视畸变系数.
        border (tuple): 指定马赛克边框的元组.
        pre_transform (callable): 在开始随机变换之前应用于图像的函数/变换.

    Methods:
        affine_transform(img, border): 对图像应用一系列仿射变换.
        apply_bboxes(bboxes, M): 使用计算的仿射矩阵变换边界框.
        apply_segments(segments, M): 转换线段并生成新的边界框.
        apply_keypoints(keypoints, M): 转换关键点.
        __call__(labels): 将变换应用于图像及其相应注释的主要方法.
        box_candidates(box1, box2): 筛选出在转换后不符合特定条件的边界框.
    """

    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        """使用转换参数初始化 RandomPerspective 对象."""

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # 马赛克边框
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """
        应用一系列以图像中心为中心的仿射变换.

        Args:
            img (ndarray): 输入图像.
            border (tuple): 边框尺寸.

        Returns:
            img (ndarray): 转换后的图像.
            M (ndarray): 变换矩阵.
            s (float): 比例因子.
        """

        # 中心
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x 平移 (像素)
        C[1, 2] = -img.shape[0] / 2  # y 平移 (像素)

        # 透视
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x 透视 (对于 y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y 透视 (对于 x)

        # 旋转和缩放
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # 将 90 度旋转添加到小旋转中
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # 剪切
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # 转变
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x 转变 (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y 转变 (pixels)

        # 组合旋转矩阵
        M = T @ S @ R @ P @ C  # 操作顺序（从右到左）很重要
        # 仿射图像
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 图像已更改
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # 仿射
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        """
       仅将仿射应用于 bbox.

        Args:
            bboxes (ndarray): BBOX列表,XYXY格式,形状(num_bboxes,4).
            M (ndarray): 仿射矩阵.

        Returns:
            new_bboxes (ndarray): bboxes 仿射后, [num_bboxes, 4].
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # 变换
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # 透视重放或仿射

        # 创建新的盒子
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):
        """
       将仿射应用于线段并从线段生成新的bbox.

        Args:
            segments (ndarray):细分列表, [num_samples, 500, 2].
            M (ndarray): 仿射矩阵.

        Returns:
            new_segments (ndarray): 仿射后的段列表, [num_samples, 500, 2].
            new_bboxes (ndarray): bboxes 仿射后, [N, 4].
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        将仿射应用于关键点.

        Args:
            keypoints (ndarray): 关键点, [N, 17, 3].
            M (ndarray):仿射矩阵.

        Returns:
            new_keypoints (ndarray): 关键点 仿射后, [N, 17, 3].
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
       仿射图像和目标.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # 不需要比率垫

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # 确保坐标格式正确
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M 是仿射矩阵
        # 对 func:`box_candidates` 缩放
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # 更新 bboxes 如果有区段.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # 筛选实例
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # 使 bbox 具有与 new_bboxes 相同的比例
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """
        基于一组阈值的计算框候选项。此方法比较盒子的特征在增强之前和之后，以决定盒子是否适合进一步处理.

        Args:
            box1 (numpy.ndarray): 扩充前的 4,n 边界框，表示为 [x1, y1, x2, y2].
            box2 (numpy.ndarray): 扩充后的 4,n 边界框，表示为 [x1, y1, x2, y2].
            wh_thr (float, optional): 宽度和高度阈值（以像素为单位）. 默认为 2.
            ar_thr (float, optional):纵横比阈值. 默认为 100.
            area_thr (float, optional): 面积比阈值. 默认为 0.1.
            eps (float, optional):小的 epsilon 值，以防止除以零. 默认为 1e-16.

        Returns:
            (numpy.ndarray): 一个布尔数组，根据给定的阈值指示哪些框是候选框.
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV:
    """
    This class is responsible for performing random adjustments to the Hue, Saturation, and Value (HSV) channels of an
    image.

    The adjustments are random but within limits set by hgain, sgain, and vgain.
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        Initialize RandomHSV class with gains for each HSV channel.

        Args:
            hgain (float, optional): Maximum variation for hue. Default is 0.5.
            sgain (float, optional): Maximum variation for saturation. Default is 0.5.
            vgain (float, optional): Maximum variation for value. Default is 0.5.
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """
        Applies random HSV augmentation to an image within the predefined limits.

        The modified image replaces the original image in the input 'labels' dict.
        """
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip:
    """
    以给定的概率对图像应用随机水平或垂直翻转.

    还会相应地更新任何实例（边界框、关键点等）.
    """

    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        """
        使用概率和方向初始化 RandomFlip 类.

        Args:
            p (float, optional): 应用翻转的概率。必须介于 0 和 1 之间.默认值为 0.5.
            direction (str, optional): 应用翻转的方向。必须是“水平”或“垂直”.默认值为水平.
            flip_idx (array-like, optional): 用于翻转关键点的索引映射（如果有）.
        """
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """
        将随机翻转应用于图像，并相应地更新任何实例，如边界框或关键点.

        Args:
            labels (dict): 包含键“img”和“instances”的字典。“img”是要翻转的图像。
                           “instances”是一个包含边界框和可选关键点的对象.

        Returns:
            (dict): 与翻转图像和“img”和“instances”键下的更新实例相同的字典.
        """
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # 上下翻转
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # 对于关键点
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """调整图像大小和填充以进行检测、实例分割、姿势."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """使用特定参数初始化 LetterBox 对象."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # 将图像放在中间或左上角

    def __call__(self, labels=None, image=None):
        """返回已添加边框的更新标签和图像."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # 当前形状 [高度、宽度]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 比例比例（新/旧）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # 只缩减，不纵向扩展（为了更好的 val mAP）
            r = min(r, 1.0)

        # 计算填充
        ratio = r, r  # 宽、高比
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # 最小矩形
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # 伸展
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height 比率

        if self.center:
            dw /= 2  # 将填充分为 2 个侧面将填充分为 2 个侧面
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste:
    """
    实现复制粘贴扩充，如论文中所述 https://arxiv.org/abs/2012.07177. 
    这个类是负责对图像及其相应实例应用复制粘贴增强.
    """

    def __init__(self, p=0.5) -> None:
        """
        以给定的概率初始化 CopyPaste 类.

        Args:
            p (float, optional): 应用复制粘贴扩充的概率。必须介于 0 和 1 之间.
                                 默认值为 0.5.
        """
        self.p = p

    def __call__(self, labels):
        """
        将复制粘贴增强应用于给定的图像和实例.

        Args:
            labels (dict): A dictionary containing:
                           - 'img': 要增强的图像.
                           - 'cls': 与实例关联的类标签.
                           - 'instances': 包含边界框以及可选的关键点和线段的对象.

        Returns:
            (dict): 在“img”、“cls”和“instances”键下使用增强图像和更新实例的字典.

        Notes:
            1. 实例应将“段”作为其属性之一，以便此增强起作用.
            2. 此方法就地修改输入字典“标签”.
        """
        im = labels["img"]
        cls = labels["cls"]
        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # 高度、宽度、通道
            im_new = np.zeros(im.shape, np.uint8)

            # 先计算 ioa，然后随机选择索引
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # 面积交叉点, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # 增强片段（左右翻转）
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]

        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        return labels


class Albumentations:
    """
    Albumentations 转换.

    可选）卸载要禁用的软件包。应用模糊、中位模糊、转换为灰度、对比度限制自适应直方图均衡、亮度和对比度的随机变化、RandomGamma 和图像质量降低压缩.
    """

    def __init__(self, p=1.0):
        """初始化 YOLO bbox 格式参数的 transform 对象."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """生成对象检测并返回包含检测结果的字典."""
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            # TODO: 添加对段和关键点的支持
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new["class_labels"]) > 0:  # 如果新 IM 中没有 bbox，则跳过更新
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes=bboxes)
        return labels


class Format:
    """
    格式化图像注释，用于对象检测、实例分割和姿态估计任务. 
    该类标准化了 PyTorch DataLoader 中的“collate_fn”要使用的图像和实例注释.

    Attributes:
        bbox_format (str): 边界框的格式. Default is 'xywh'.
        normalize (bool): 是否规范化边界框. Default is True.
        return_mask (bool): 返回用于分段的实例掩码. Default is False.
        return_keypoint (bool): 返回用于姿态估计的关键点. Default is False.
        mask_ratio (int): 掩模的下采样率. Default is 4.
        mask_overlap (bool): 是否重叠掩码. Default is True.
        batch_idx (bool): 保留批处理索引. Default is True.
        bgr (float): 返回 BGR 图像的概率. Default is 0.0.
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        """使用给定参数初始化 Format 类."""
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # 仅在训练检测时设置 False
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # 保留批处理索引
        self.bgr = bgr

    def __call__(self, labels):
        """返回格式化的图像、类、边界框和关键点,供“collate_fn”使用."""
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        # NOTE: 需要以 xywhr 格式规范化 OBB 以实现宽度-高度一致性
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # 然后我们可以使用collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """将 YOLO 的图像从 Numpy 数组格式化为 PyTorch 张量."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """将多边形点转换为位图."""
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls


class RandomLoadText:
    """
    随机抽取正面文本和负面文本，并根据样本数量更新类索引.

    Attributes:
        prompt_format (str): 提示格式. Default is '{}'.
        neg_samples (tuple[int]): 随机抽取负面文本的护林员, Default is (80, 80).
        max_samples (int): 一个图像中不同文本样本的最大数量, Default is 80.
        padding (bool): 是否将文本填充到max_samples. Default is False.
        padding_value (str): 填充文本. Default is "".
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: Tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: str = "",
    ) -> None:
        """使用给定参数初始化 RandomLoadText 类."""
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: dict) -> dict:
        """返回更新的类和文本."""
        assert "texts" in labels, "No texts found in labels."
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()

        if len(pos_labels) > self.max_samples:
            pos_labels = set(random.sample(pos_labels, k=self.max_samples))

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = []
        for i in range(num_classes):
            if i not in pos_labels:
                neg_labels.append(i)
        neg_labels = random.sample(neg_labels, k=neg_samples)

        sampled_labels = pos_labels + neg_labels
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(new_cls)

        # 当有多个提示时，随机选择一个提示
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)

        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        labels["texts"] = texts
        return labels


def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """将图像转换为适合 YOLOv8 训练的大小."""
    pre_transform = Compose(
        [
            Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            CopyPaste(p=hyp.copy_paste),
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    flip_idx = dataset.data.get("flip_idx", [])  # 用于关键点增强
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


# 分类增强 -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation=Image.BILINEAR,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    用于评估/推理的分类转换.

    Args:
        size (int): 图像尺寸
        mean (tuple): RGB通道的平均值
        std (tuple): RGB通道的标准值
        interpolation (T.InterpolationMode): 插值模式。默认值为 T.InterpolationMode.BILINEAR.
        crop_fraction (float): 要裁剪的图像比例. default is 1.0.

    Returns:
        (T.Compose): TorchVision 转变
    """
    import torchvision.transforms as T  # 加快 'import ultralytics' 的范围

    if isinstance(size, (tuple, list)):
        assert len(size) == 2
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # 保留纵横比，裁剪在图像中居中，不添加边框，图像丢失
    if scale_size[0] == scale_size[1]:
        # 简单情况，使用torchvision内置的“调整大小”，使用最短边模式（标量大小参数）
        tfl = [T.Resize(scale_size[0], interpolation=interpolation)]
    else:
        # 调整最短边的大小，使其与非方形目标的目标调光相匹配
        tfl = [T.Resize(scale_size)]
    tfl += [T.CenterCrop(size)]

    tfl += [
        T.ToTensor(),
        T.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std),
        ),
    ]

    return T.Compose(tfl)


# 分类训练增强 --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # 图像 HSV-Hue 增强（分数）
    hsv_s=0.4,  # 图像 HSV-饱和度增强（分数）
    hsv_v=0.4,  # 图像 HSV-值增强（分数）
    force_color_jitter=False,
    erasing=0.0,
    interpolation=Image.BILINEAR,
):
    """
    分类通过训练的增强进行转换.

    Args:
        size (int): 图像尺寸
        scale (tuple): 图像的缩放范围. default is (0.08, 1.0)
        ratio (tuple): 图像的纵横比范围. default is (3./4., 4./3.)
        mean (tuple): RGB通道的平均值
        std (tuple): RGB通道的标准值
        hflip (float): 水平翻转的概率
        vflip (float): 垂直翻转的概率
        auto_augment (str): 自动增强策略. can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): 图像 HSV-Hue 增强（分数）
        hsv_s (float): 图像 HSV-饱和度增强（分数）
        hsv_v (float): 图像 HSV-值增强（分数）
        force_color_jitter (bool): 强制应用颜色抖动，即使启用了自动增强
        erasing (float): 随机擦除的概率
        interpolation (T.InterpolationMode): 插值模式. default is T.InterpolationMode.BILINEAR.

    Returns:
        (T.Compose): TorchVision 转变
    """
    # 在未安装 Albumentations 时应用的转换
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # 默认 ImageNet 缩放范围
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # 默认 ImageNet 比率范围
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl += [T.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [T.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # 如果 AA/RA 开启，颜色抖动通常会被禁用，
        # 这允许在不破坏旧 HPARM的 CFG 的情况下进行覆盖
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl += [T.RandAugment(interpolation=interpolation)]
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl += [T.AugMix(interpolation=interpolation)]
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl += [T.AutoAugment(interpolation=interpolation)]
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl += [T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h)]

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


# NOTE: keep this class for backward compatibility
class ClassifyLetterBox:
    """
    YOLOv8 LetterBox class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([LetterBox(size), ToTensor()]).

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically solves for short side using stride.
        stride (int): The stride value, used when 'auto' is True.
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox class with a target size, auto-flag, and stride.

        Args:
            size (Union[int, Tuple[int, int]]): The target dimensions (height, width) for the letterbox.
            auto (bool): If True, automatically calculates the short side based on stride.
            stride (int): The stride value, used when 'auto' is True.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes the image and pads it with a letterbox method.

        Args:
            im (numpy.ndarray): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The letterboxed and resized image as a numpy array.
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # Calculate padding dimensions
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # Create padded image
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


# NOTE: keep this class for backward compatibility
class CenterCrop:
    """YOLOv8 CenterCrop class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([CenterCrop(size), ToTensor()]).
    """

    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Resizes and crops the center of the image using a letterbox method.

        Args:
            im (numpy.ndarray): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The center-cropped and resized image as a numpy array.
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


# NOTE: keep this class for backward compatibility
class ToTensor:
    """YOLOv8 ToTensor class for image preprocessing, i.e., T.Compose([LetterBox(size), ToTensor()])."""

    def __init__(self, half=False):
        """Initialize YOLOv8 ToTensor object with optional half-precision support."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Transforms an image from a numpy array to a PyTorch tensor, applying optional half-precision and normalization.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized to [0, 1].
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
