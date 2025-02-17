import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

def load_dataset(data_path, labels=None, min_length=3, max_length=None):#接受数据路径、标签文件名、最小长度和最大长度
    sizes = []#样本长度
    offsets = []#样本偏移量
    emo_labels = []#情感标签
    npy_data = np.load(os.path.join(data_path , "train.npy"))

    #npy_data = np.load(data_path + "test.npy")#通过np.load加载.npy文件，这通常包括特征数据
    #print(npy_data)

    offset = 0 #用于追踪当前处理的样本在数据中的偏移位置
    skipped = 0 #计数器，用于统计被跳过的样本数量

    #if not os.path.exists(data_path + f".{labels}"):#如果存在，从对应的标签文件中读取标签
    if not os.path.exists(os.path.join(data_path , "train.emo")):
        labels = None
    train_data = os.path.join(data_path , "train.lengths")
    #print(train_data)

    # with open(train_data , "r") as len_f, open(
    #     data_path + f".{labels}", "r"
    # ) if labels is not None else contextlib.ExitStack() as lbl_f:
    #     for line in len_f:
    #         length = int(line.rstrip())
    #         lbl = None if labels is None else next(lbl_f).rstrip().split()[
    #             1]  # only emo is needed
    #         if length >= min_length and (
    #             max_length is None or length <= max_length
    #         ):
    #             sizes.append(length) #如果当前样本满足长度要求，则将其长度添加到sizes列表中。
    #             offsets.append(offset)#同时，将当前样本的偏移量添加到offsets列表中
    #             if lbl is not None:
    #                 emo_labels.append(lbl)
    #         offset += length

    with open(train_data , "r") as len_f, open(
        os.path.join(data_path , "train.emo"), "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length) #如果当前样本满足长度要求，则将其长度添加到sizes列表中。
                offsets.append(offset)#同时，将当前样本的偏移量添加到offsets列表中
                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
#这段代码的目的是从一个给定的数据路径中加载语音数据集，同时读取每个样本的长度和（如果提供）情感标签，然后根据长度要求过滤样本，最后返回处理后的数据集信息。
    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")

    return npy_data, sizes, offsets, emo_labels

class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        labels=None,
        shuffle=True,#是否打乱数据
        sort_by_length=True,#是否按长度排序等参数
    ):
        super().__init__()
        
        self.feats = feats#特征数据
        self.sizes = sizes  # length of each sample 样本大小
        self.offsets = offsets  # offset of each sample 偏移量

        self.labels = labels

        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        res = {"id": index, "feats": feats}
        if self.labels is not None:
            res["target"] = self.labels[index]
        print(res)

        return res
#根据索引获取单个样本的方法。它返回一个包含样本ID、特征数据和（如果有的话）目标标签的字典。
    def __len__(self):
        return len(self.sizes)
#返回数据集的样本数量
    def collator(self, samples):
        if len(samples) == 0:
            return {}
#自定义的批处理函数，用于将多个样本组合成一个批次。它处理特征数据的填充，并生成一个包含输入网络所需信息的字典。
        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None

        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(
            len(feats), target_size, feats[0].size(-1)
        )

        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

def load_ssl_features(feature_path, label_dict, max_speech_seq_len=None):

    data, sizes, offsets, labels = load_dataset(feature_path, labels='emo', min_length=1, max_length=max_speech_seq_len)
    labels = [ label_dict[elem] for elem in labels ]
    
    num = len(labels)
    iemocap_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "num": num
    } 

    return iemocap_data

def train_valid_test_iemocap_dataloader(
        data, 
        batch_size,
        test_start, 
        test_end,
        eval_is_test=False,
    ):
    feats = data['feats']
    sizes, offsets = data['sizes'], data['offsets']
    labels = data['labels']
    print(feats)

    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_labels = labels[test_start:test_end]

    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start
    
    test_dataset = SpeechDataset(
        feats=test_feats,
        sizes=test_sizes, 
        offsets=test_offsets,
        labels=test_labels,
    )
    np.save('test_feats.npy', test_feats)
    np.save('test_sizes.npy', test_sizes)
    np.save('test_offsets.npy', test_offsets)
    np.save('test_labels.npy', test_labels)

    train_val_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    train_val_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_sizes)[:-1]], dtype=np.int64)
    train_val_labels = [item for item in labels[:test_start] + labels[test_end:]]
    train_val_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)
    print("train_val_sizes")
    print(train_val_sizes,train_val_offsets,train_val_labels,train_val_feats)

    print(eval_is_test)

    if eval_is_test:
        train_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            labels=train_val_labels,
        )
        print("train_dataset_collator")
        print(train_dataset.collator)
        val_dataset = test_dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator,
                               num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator,
                               num_workers=4, pin_memory=True, shuffle=False)
    
    else:
        train_val_nums = data['num'] - (test_end - test_start)
        train_nums = int(0.8 * train_val_nums)
        val_nums = train_val_nums - train_nums

        train_val_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            labels=train_val_labels,
        )
        print(f"Size of the dataset before split: {len(train_val_dataset)}")

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])
        print(f"Train nums: {train_nums}, Val nums: {val_nums}")
        print("train_val_dataset.collator")
        print(train_val_dataset.collator)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader

#数据加载、预处理、批处理和数据集分割的功能
if __name__ == "__main__":
    None
    # npy_data, sizes, offsets, emo_labels = load_dataset(r'C:\Users\ROG\Desktop\project\emotion2vec\data')
    # print("emo_labels")
    # print(emo_labels)
    # feats = npy_data
    # print(feats)
    # print(feats[0:255835])
    # if isinstance(feats, np.ndarray) and feats.ndim == 2:
    #     print("feats 是一个二维 NumPy 数组。")
    # else:
    #     print("feats 不是一个二维 NumPy 数组。")
    # speech_dataset = SpeechDataset(
    #     feats=npy_data,
    #     sizes=sizes,
    #     offsets=offsets,
    #     labels=emo_labels,
    #     shuffle=True,  # 在初始化时打乱数据
    #     sort_by_length=True  # 按样本长度排序
    # )
    # print(speech_dataset)
    # data, sizes, offsets, labels = load_dataset(r'C:\Users\ROG\Desktop\project\emotion2vec\data', labels='emo', min_length=1)
    # print(labels)

