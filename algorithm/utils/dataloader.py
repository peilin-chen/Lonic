import numpy as np
import torch
from torchvision import transforms, datasets
import tonic
import logging
from .cifar10_dvs import CIFAR10DVS
from .augmentation import ToPILImage, Resize, Padding, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip
import random
import math
import PIL
import warnings

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
from . import datasets as sjds
from torchvision.datasets.utils import extract_archive
import torch
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import torch.utils.data as data

class DVS128Gesture(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            padding_frame: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        '''
        :param root: root path of the dataset
        :type root: str
        :param train: whether use the train set
        :type train: bool
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number`
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param padding_frame: whether padding the frames number to the maximum number of frames
        :type padding_frame: bool
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable

        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``.

        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`cal_fixed_frames_number_segment_index` for
            more details.

        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration. If ``padding_frame`` is ``True``, each sample
            will be padded to the same frames number (length), which is the maximum frames number of all frames.

        '''
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, padding_frame, transform, target_transform)
        self.transform = transform
        self.target_transform = target_transform
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract [{fpath}] to [{extract_root}].')
        extract_archive(fpath, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        return sjds.load_aedat_v3(file_name)

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = DVS128Gesture.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 11

        # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
        # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
        # are replaced by new codes.


        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np.savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1

        # old codes:

        # index = 0
        # index_l = 0
        # index_r = 0
        # for i in range(csv_data.shape[0]):
        #     # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
        #     label = csv_data[i][0] - 1
        #     t_start = csv_data[i][1]
        #     t_end = csv_data[i][2]
        #
        #     while True:
        #         t = events['t'][index]
        #         if t < t_start:
        #             index += 1
        #         else:
        #             index_l = index
        #             break
        #     while True:
        #         t = events['t'][index]
        #         if t < t_end:
        #             index += 1
        #         else:
        #             index_r = index
        #             break
        #
        #     file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
        #     np.savez(file_name,
        #         t=events['t'][index_l:index_r],
        #         x=events['x'][index_l:index_r],
        #         y=events['y'][index_l:index_r],
        #         p=events['p'][index_l:index_r]
        #     )
        #     print(f'[{file_name}] saved.')
        #     label_file_num[label] += 1

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedat_dir = os.path.join(extract_root, 'DvsGesture')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 64)) as tpe:
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), train_dir)

                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file,
                                   os.path.join(aedat_dir, fname + '_labels.csv'), test_dir)

            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = torch.from_numpy(sample)
            T, C, H, W = sample.size()
            transformed_sample = self.transform([sample[i] for i in range(T)])
            sample = torch.stack(transformed_sample, 0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def _is_numpy_image(img):
    return img.ndim in {2, 3}

def cutout(img, i, j, h, w, v, inplace=False):
    """ Erase the CV Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        CV Image: Cutout image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.copy()

    img[i:i + h, j:j + w, :] = v
    return img


class Cutout(object):
    """Random erase the given CV Image.

    It has been proposed in
    `Improved Regularization of Convolutional Neural Networks with Cutout`.
    `https://arxiv.org/pdf/1708.04552.pdf`


    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        pixel_level (bool): filling one number or not. Default value is False
    """
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.pixel_level = pixel_level
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio):
        if type(img) == np.ndarray:
            img_h, img_w, img_c = img.shape
        else:
            img_h, img_w = img.size
            img_c = len(img.getbands())

        s = random.uniform(*scale)
        # if you img_h != img_w you may need this.
        # r_1 = max(r_1, (img_h*s)/img_w)
        # r_2 = min(r_2, img_h / (img_w*s))
        r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if random.random() < self.p:
            left, top, h, w, ch = self.get_params(img, self.scale, self.ratio)

            if self.pixel_level:
                c = np.random.randint(*self.value, size=(h, w, ch), dtype='uint8')
            else:
                c = random.randint(*self.value)

            if type(img) == np.ndarray:
                return cutout(img, top, left, h, w, c, self.inplace)
            else:
                if self.pixel_level:
                    c = PIL.Image.fromarray(c)
                img.paste(c, (left, top, left + w, top + h))
                return img
        return img

def dataloader(args, dataset='DVSGesture', evaluate=False, distributed=False, batch_size=16, val_batch_size=16, workers=4):
    data_path = args.data_path
    if dataset == 'DVSGesture':
        train_loader, val_loader, trainset_len, testset_len = dataloader_gesture(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 11
        args.n_steps = 20
        args.n_inputs = 2
        args.dt = 75e-3
        args.classif = True
        args.delay_targets = 5
        args.skip_test = False
    elif dataset == "CIFAR10DVS":  # Dim: (2, 34, 34)
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar10dvs(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 10
        args.n_inputs = 2
        args.dt = 10e-3
        args.classif = True
        args.delay_targets = 7
        args.skip_test = False
    elif dataset == 'CIFAR10':
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar10(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 6
        args.n_inputs = 32
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    elif dataset == 'CIFAR100':
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar100(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 100
        args.n_steps = 6
        args.n_inputs = 32
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    elif dataset == 'TINYIMAGENET':
        train_loader, val_loader, trainset_len, valset_len = dataloader_tiny_imagenet(
            batch_size, val_batch_size, workers, data_path
        )
        args.full_train_len = trainset_len
        args.full_test_len = valset_len
        args.n_classes = 200
        args.n_steps = 6
        args.n_inputs = 64
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    else:
        logging.info("ERROR: {0} is not supported".format(dataset))
        raise NameError("{0} is not supported".format(dataset))

    return train_loader, val_loader


#def dataloader_gesture(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", reproducibility=False):
#    labels = 11
#    sensor_size = tonic.datasets.DVSGesture.sensor_size
#    trainset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=True)
#    testset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=False)
#
#    slicing_time_window = 1575000
#    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)
#
#    frame_transform = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
#        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=75000),
#        torch.tensor, transforms.Resize(32)
#    ])
#    frame_transform_test = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
#        tonic.transforms.ToFrame(sensor_size=sensor_size,
#                                 time_window=75000),
#        torch.tensor,
#        transforms.Resize(32, antialias=True)
#    ])
#
#    trainset_ori_sl = tonic.SlicedDataset(trainset_ori, slicer=slicer,
#                                          metadata_path=data_path + '/metadata/online_dvsg_train',
#                                          transform=frame_transform)
#    # testset_ori_sl = tonic.SlicedDataset(testset_ori, slicer=slicer,
#    #                                      metadata_path=data_path + '/metadata/online_dvsg_test',
#    #                                      transform=frame_transform_test)
#
#    print(
#        f"Went from {len(trainset_ori)} samples in the original dataset to {len(trainset_ori_sl)} in the sliced version.")
#    print(
#        f"Went from {len(testset_ori)} samples in the original dataset to {len(testset_ori)} in the sliced version.")
#
#    frame_transform2 = tonic.transforms.Compose([  # tonic.transforms.DropEvent(p=0.1),
#        torch.tensor,
#        transforms.RandomCrop(32, padding=4)
#    ])
#
#    trainset = tonic.CachedDataset(trainset_ori_sl,
#                                   cache_path=data_path + '/cache/online_fast_dataloading_train',
#                                   transform=frame_transform2)
#    # if evaluate:
#    testset = tonic.CachedDataset(testset_ori,
#                                  cache_path=data_path + '/cache/online_fast_dataloading_test',
#                                  transform=frame_transform_test)
#
#    if reproducibility:
#        import numpy as np
#        import random
#        def seed_worker(worker_id):
#            worker_seed = torch.initial_seed() % 2 ** 32
#            np.random.seed(worker_seed)
#            random.seed(worker_seed)
#
#        g = torch.Generator()
#        g.manual_seed(0)
#        train_loader = torch.utils.data.DataLoader(
#            trainset, batch_size=batch_size, shuffle=True,
#            num_workers=workers, pin_memory=True,
#            collate_fn=tonic.collation.PadTensors(batch_first=True), worker_init_fn=seed_worker, generator=g, )
#        val_loader = torch.utils.data.DataLoader(
#            testset,
#            batch_size=val_batch_size, shuffle=False,
#            num_workers=workers, pin_memory=True,
#            collate_fn=tonic.collation.PadTensors(batch_first=True), worker_init_fn=seed_worker, generator=g, )
#    else:
#        train_loader = torch.utils.data.DataLoader(
#            trainset, batch_size=batch_size, shuffle=True,
#            num_workers=workers, pin_memory=True,
#            collate_fn=tonic.collation.PadTensors(batch_first=True))
#        val_loader = torch.utils.data.DataLoader(
#            testset,
#            batch_size=val_batch_size, shuffle=False,
#            num_workers=workers, pin_memory=True,
#            collate_fn=tonic.collation.PadTensors(batch_first=True))
#
#    return train_loader, val_loader, len(trainset_ori_sl), len(testset_ori)

def dataloader_gesture(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", reproducibility=False):
    
    trainset = DVS128Gesture(root="/sfs/gpfs/tardis/home/cts3td/SNN/SLTT-main/data", train=True,  data_type='frame', frames_number=10, split_by='number')
    testset  = DVS128Gesture(root="/sfs/gpfs/tardis/home/cts3td/SNN/SLTT-main/data", train=False, data_type='frame', frames_number=10, split_by='number')

    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)
                                       
    test_loader  = data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers, drop_last=False, pin_memory=True)
                                       
    return train_loader, test_loader, len(trainset), len(testset)

def str_to_num(x):
    labels_dict = {'cup': 0, 'ibis': 1, 'crocodile': 2, 'wild_cat': 3, 'Leopards': 4, 'watch': 5, 'pagoda': 6, 'soccer_ball': 7, 'accordion': 8, 'sunflower': 9, 'rooster': 10, 'ewer': 11, 'stegosaurus': 12, 'ketch': 13, 'rhino': 14, 'cellphone': 15, 'brontosaurus': 16, 'buddha': 17, 'chandelier': 18, 'crayfish': 19, 'strawberry': 20, 'stapler': 21, 'nautilus': 22, 'stop_sign': 23, 'BACKGROUND_Google': 24, 'lamp': 25, 'platypus': 26, 'gerenuk': 27, 'starfish': 28, 'octopus': 29, 'flamingo_head': 30, 'butterfly': 31, 'revolver': 32, 'umbrella': 33, 'garfield': 34, 'sea_horse': 35, 'yin_yang': 36, 'beaver': 37, 'metronome': 38, 'tick': 39, 'trilobite': 40, 'airplanes': 41, 'hawksbill': 42, 'chair': 43, 'pizza': 44, 'anchor': 45, 'euphonium': 46, 'lotus': 47, 'minaret': 48, 'cannon': 49, 'bonsai': 50, 'windsor_chair': 51, 'wrench': 52, 'headphone': 53, 'Motorbikes': 54, 'scorpion': 55, 'cougar_face': 56, 'crocodile_head': 57, 'mandolin': 58, 'barrel': 59, 'inline_skate': 60, 'ferry': 61, 'laptop': 62, 'bass': 63, 'okapi': 64, 'saxophone': 65, 'hedgehog': 66, 'cougar_body': 67, 'scissors': 68, 'crab': 69, 'dalmatian': 70, 'dolphin': 71, 'mayfly': 72, 'pigeon': 73, 'emu': 74, 'electric_guitar': 75, 'panda': 76, 'helicopter': 77, 'schooner': 78, 'camera': 79, 'ant': 80, 'water_lilly': 81, 'elephant': 82, 'llama': 83, 'car_side': 84, 'binocular': 85, 'ceiling_fan': 86, 'menorah': 87, 'dragonfly': 88, 'brain': 89, 'joshua_tree': 90, 'lobster': 91, 'grand_piano': 92, 'flamingo': 93, 'wheelchair': 94, 'dollar_bill': 95, 'kangaroo': 96, 'gramophone': 97, 'Faces_easy': 98, 'snoopy': 99, 'pyramid': 100}
    return torch.tensor(labels_dict[x])


def dataloader_cifar10(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    import torch.utils.data as data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR10


    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    val_loader = data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, len(trainset), len(testset)


def dataloader_cifar100(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    import torch.utils.data as data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100


    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    val_loader = data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, len(trainset), len(testset)

def dataloader_tiny_imagenet(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    import os
    import torch
    import torch.utils.data as data
    from datasets import load_dataset
    from torchvision import transforms

    data_path = os.path.expanduser(data_path)

    # Tiny-ImageNet commonly used normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2302, 0.2265, 0.2262)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2302, 0.2265, 0.2262)),
    ])

    hf_train = load_dataset(
        "slegroux/tiny-imagenet-200-clean",
        split="train",
        cache_dir=data_path
    )

    hf_val = load_dataset(
        "slegroux/tiny-imagenet-200-clean",
        split="validation",
        cache_dir=data_path
    )

    class TinyImageNetDataset(data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf_dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, index):
            sample = self.hf_dataset[index]
            image = sample["image"].convert("RGB")
            label = int(sample["label"])

            if self.transform is not None:
                image = self.transform(image)

            return image, label

    trainset = TinyImageNetDataset(hf_train, transform=transform_train)
    valset = TinyImageNetDataset(hf_val, transform=transform_val)

    train_loader = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    val_loader = data.DataLoader(
        valset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers
    )

    return train_loader, val_loader, len(trainset), len(valset)

def dataloader_cifar10dvs(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", img_size=48):
    transform_train = transforms.Compose([
        ToPILImage(),
        Resize(48),
        Padding(4),
        RandomCrop(size=48, consistent=True),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])
    num_classes = 10

    trainset = CIFAR10DVS(data_path, train=True, use_frame=True, frames_num=10, split_by='number',
                          normalization=None, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = CIFAR10DVS(data_path, train=False, use_frame=True, frames_num=10, split_by='number',
                         normalization=None, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, len(trainset), len(testset)
