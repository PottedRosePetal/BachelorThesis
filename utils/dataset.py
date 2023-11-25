import os, random, h5py
from copy import copy
from collections import Counter

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

synsetid_to_cate = {
    '02691156': 'airplane', 
    '02773838': 'bag', 
    '02801938': 'basket',
    '02808440': 'bathtub', 
    '02818832': 'bed', 
    '02828884': 'bench',
    '02876657': 'bottle', 
    '02880940': 'bowl', 
    '02924116': 'bus',
    '02933112': 'cabinet', 
    '02747177': 'can', 
    '02942699': 'camera',
    '02954340': 'cap', 
    '02958343': 'car', 
    '03001627': 'chair',
    '03046257': 'clock', 
    '03207941': 'dishwasher', 
    '03211117': 'monitor',
    '04379243': 'table', 
    '04401088': 'telephone', 
    '02946921': 'tin_can',
    '04460130': 'tower', 
    '04468005': 'train', 
    '03085013': 'keyboard',
    '03261776': 'earphone', 
    '03325088': 'faucet', 
    '03337140': 'file',
    '03467517': 'guitar', 
    '03513137': 'helmet', 
    '03593526': 'jar',
    '03624134': 'knife', 
    '03636649': 'lamp', 
    '03642806': 'laptop',
    '03691459': 'speaker', 
    '03710193': 'mailbox', 
    '03759954': 'microphone',
    '03761084': 'microwave', 
    '03790512': 'motorcycle', 
    '03797390': 'mug',
    '03928116': 'piano', 
    '03938244': 'pillow', 
    '03948459': 'pistol',
    '03991062': 'pot', 
    '04004475': 'printer', 
    '04074963': 'remote_control',
    '04090263': 'rifle', 
    '04099429': 'rocket', 
    '04225987': 'skateboard',
    '04256520': 'sofa', 
    '04330267': 'stove', 
    '04530566': 'vessel',
    '04554684': 'washer', 
    '02992529': 'cellphone',
    '02843684': 'birdhouse', 
    '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}

cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}
cate_to_list = [v for k, v in synsetid_to_cate.items()]
cate_to_index = lambda cate: cate_to_list.index(cate)

def cate_ind_generator(batch_size):
    for _ in range(batch_size):
        yield random.sample(list(range(len(cate_to_list))),1)


class ShapeNetCore(Dataset):
    GRAVITATIONAL_AXIS = 1

    def __init__(self, path, cates, split, scale_mode, seed, transform=None, even_out=False, logger = None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.seed = seed
        self.even_out = even_out
        self.logger = logger

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):
        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):
        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name
        
        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):
                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale,
                    'angle': torch.zeros(pc.shape[1])
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(self.seed).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_sampler(self):
        total_items = len([pc["cate"] for pc in self.pointclouds])
        counts_dict = dict(Counter([pc["cate"] for pc in self.pointclouds]))
        weight_per_sample = [1/counts_dict[pc["cate"]] for pc in self.pointclouds]
        return WeightedRandomSampler(weights=weight_per_sample, num_samples=total_items)

# Code below was used to generate evened out code, but deprectated/not maintained due to it being a bad method

# class ShapeNetCore(Dataset):

#     GRAVITATIONAL_AXIS = 1
    
#     def __init__(self, path, cates, split, scale_mode, seed, transform=None, even_out=False, logger = None):
        # super().__init__()
        # assert isinstance(cates, list), '`cates` must be a list of cate names.'
        # assert split in ('train', 'val', 'test')
        # assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        # self.path = path
        # if 'all' in cates:
        #     cates = cate_to_synsetid.keys()
        # self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        # self.cate_synsetids.sort()
        # self.split = split
        # self.scale_mode = scale_mode
        # self.transform = transform
        # self.seed = seed
        # self.even_out = even_out
        # self.logger = logger

        # self.pointclouds = []
        # self.stats = None

        # self.get_statistics()
        # self.load()

#     def get_statistics(self):

#         basename = os.path.basename(self.path)
#         dsetname = basename[:basename.rfind('.')]
#         stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
#         os.makedirs(stats_dir, exist_ok=True)

#         if len(self.cate_synsetids) == len(cate_to_synsetid):
#             stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
#         else:
#             stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
#         if os.path.exists(stats_save_path):
#             self.stats = torch.load(stats_save_path)
#             return self.stats

#         with h5py.File(self.path, 'r') as f:
#             pointclouds = []
#             for synsetid in self.cate_synsetids:
#                 for split in ('train', 'val', 'test'):
#                     pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

#         all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
#         B, N, _ = all_points.size()
#         mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
#         std = all_points.view(-1).std(dim=0)        # (1, )

#         self.stats = {'mean': mean, 'std': std}
#         torch.save(self.stats, stats_save_path)
#         return self.stats

#     def load(self):

#         def _enumerate_pointclouds(f):
#             for synsetid in self.cate_synsetids:
#                 cate_name = synsetid_to_cate[synsetid]
#                 for j, pc in enumerate(f[synsetid][self.split]):
#                     yield torch.from_numpy(pc), j, cate_name
        
#         with h5py.File(self.path, mode='r') as f:
            
#             for pc, pc_id, cate_name in _enumerate_pointclouds(f):
#                 if self.scale_mode == 'global_unit':
#                     shift = pc.mean(dim=0).reshape(1, 3)
#                     scale = self.stats['std'].reshape(1, 1)
#                 elif self.scale_mode == 'shape_unit':
#                     shift = pc.mean(dim=0).reshape(1, 3)
#                     scale = pc.flatten().std().reshape(1, 1)
#                 elif self.scale_mode == 'shape_half':
#                     shift = pc.mean(dim=0).reshape(1, 3)
#                     scale = pc.flatten().std().reshape(1, 1) / (0.5)
#                 elif self.scale_mode == 'shape_34':
#                     shift = pc.mean(dim=0).reshape(1, 3)
#                     scale = pc.flatten().std().reshape(1, 1) / (0.75)
#                 elif self.scale_mode == 'shape_bbox':
#                     pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
#                     pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
#                     shift = ((pc_min + pc_max) / 2).view(1, 3)
#                     scale = (pc_max - pc_min).max().reshape(1, 1) / 2
#                 else:
#                     shift = torch.zeros([1, 3])
#                     scale = torch.ones([1, 1])

#                 pc = (pc - shift) / scale

#                 self.pointclouds.append({
#                     'pointcloud': pc,
#                     'cate': cate_name,
#                     'id': pc_id,
#                     'shift': shift,
#                     'scale': scale,
#                     'angle': torch.zeros(pc.shape[1])
#                 })


#         if self.even_out:   #shouldnt be needed anymore with weighted random sampler?
#             # ensure all categories are of an even distribution, no category being overrepresented
#             self.pointclouds = self.filter_category_limit(self.pointclouds)

#         # for i in range(len(self.pointclouds)):
#         #     self.pointclouds[i]["cate"] = torch.tensor(int(cate_to_synsetid[self.pointclouds[i]["cate"]])).unsqueeze(dim=0)

#         # Deterministically shuffle the dataset
#         self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
#         random.Random(self.seed).shuffle(self.pointclouds)   #does not use random seed in argument, but default value of seed, changed from orig, intended behavior?

#     def __len__(self):
#         return len(self.pointclouds)

#     def __getitem__(self, idx):
#         data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
#         if self.transform is not None:
#             data = self.transform(data)
#         return data

#     def filter_category_limit(self, list_of_dicts):
#         category_counts = {}
#         new_list_of_dicts = []
#         removed_pcs = {pc["cate"]:0 for pc in list_of_dicts}

#         for item in list_of_dicts:
#             category = item['cate']
#             if category in category_counts:
#                 if category_counts[category] < self.min_categories():
#                     new_list_of_dicts.append(item)
#                     category_counts[category] += 1
#                 else:
#                     removed_pcs[category] += 1
#             else:
#                 category_counts[category] = 1
#                 new_list_of_dicts.append(item)

#         if self.logger:
#             self.logger.info(f"[Dataset] Removed the following ammount of data from each category in {self.split}-split:")
#             for cate, ammount in removed_pcs.items():
#                 self.logger.info(f"[Dataset] {ammount}/{category_counts[cate]+ammount} number of pointclouds removed from category {cate}")
                
#         return new_list_of_dicts

    
#     def min_categories(self):
#         return min(Counter([pc["cate"] for pc in self.pointclouds]).values())

#     def get_sampler(self):
#         total_items = len([pc["cate"] for pc in self.pointclouds])
#         counts_dict = dict(Counter([pc["cate"] for pc in self.pointclouds]))
#         weight_per_sample = [1/counts_dict[pc["cate"]] for pc in self.pointclouds]
#         return WeightedRandomSampler(weights=weight_per_sample, num_samples=total_items)
        
#     def balanced_iter(self):
#         # Create a dictionary to organize point clouds by category and initialize with empty lists
#         cates = {pc["cate"]:[] for pc in self.pointclouds}
        
#         # Populate the dictionary with point cloud IDs grouped by their categories
#         for pc in self.pointclouds:
#             cates[pc["cate"]].append(pc["id"])

#         # Find the maximum number of point cloud IDs under any category
#         max_length = max(len(lst) for lst in cates.values())
        
#         # Extend the point cloud IDs lists under each category to match the maximum length
#         for cate in cates:
#             extend_factor = (max_length // len(cates[cate])) + 1
#             cates[cate] = (cates[cate] * extend_factor)[:max_length]

#         # Iterate over the range of indices up to the maximum length
#         for i in range(max_length):
#             # Iterate over categories
#             for cate in cates:
#                 # Check if the current index is within the valid range of point cloud IDs
#                 if i < len(cates[cate]):
#                     # Yield the data corresponding to the current index and category
#                     yield self.__get_data_by_id_cate(cates[cate][i],cate)

#     def __get_data_by_id_cate(self, idx, cate):
#         # Iterate through the point clouds to find data with the matching ID and category
#         for data in self.pointclouds:
#             if data["id"] == idx and data["cate"] == cate:
#                 # Return the found data
#                 return data
#         # Return None if no matching data is found
#         return None


