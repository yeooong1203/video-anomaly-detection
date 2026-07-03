import torch.utils.data as data
import numpy as np
import torch
import option

args = option.parser.parse_args()

class UCFTestVideoDataset(data.Dataset):
    
    def __init__(self, conall_path, nalist_path):
        self.nalist = np.load(nalist_path)                 # (N,2)
        self.total_T = int(self.nalist[-1, 1])
        
        self.con_all = np.memmap(conall_path, dtype="float32", mode="r",
                                 shape=(self.total_T, 10, args.feature_size))
        '''self.con_all  = np.load(conall_path, mmap_mode="r")

        assert int(self.nalist[-1, 1]) == self.total_T, "nalist end index must equal total_T"
        assert self.con_all.shape == (self.total_T, 10, args.feature_size), (
            f"Test feature shape mismatch: {self.con_all.shape} vs "
            f"{(self.total_T, 10, args.feature_size)}"
        )'''


    def __len__(self):
        return len(self.nalist)

    def __getitem__(self, index):
        a, b = map(int, self.nalist[index])
        x = np.array(self.con_all[a:b], dtype=np.float32)  # (T,10,2048)
        if x.ndim == 3:
            x = x.mean(axis=1)
        return torch.from_numpy(x)                         # CPU float32


class UCFTrainVideoDataset_Stratified(data.Dataset):
    
    def __init__(self, conall_path, pseudo_path, nalist_path, confidence_path,
                 window_size=500, stride=500):
        
        self.nalist = np.load(nalist_path)
        self.num_videos = len(self.nalist)
        self.pseudo_labels = np.load(pseudo_path).astype(np.float32)
        self.confidences = np.load(confidence_path).astype(np.float32)
        self.total_T = len(self.pseudo_labels)
        assert int(self.nalist[-1, 1]) == self.total_T, (
            f"nalist total_T mismatch: nalist_end={int(self.nalist[-1, 1])}, pseudo_T={self.total_T}"
        )
        assert len(self.confidences) == self.total_T, (
            f"confidence length mismatch: confidence_T={len(self.confidences)}, pseudo_T={self.total_T}"
        )
        self.con_all = np.load(conall_path, mmap_mode="r")
        assert self.con_all.shape == (self.total_T, 10, args.feature_size), (
            f"Train feature shape mismatch: {self.con_all.shape} vs "
            f"{(self.total_T, 10, args.feature_size)}"
        )
        
        '''self.con_all = np.memmap(
            conall_path,
            dtype="float32",
            mode="r",
            shape=(self.total_T, 10, args.feature_size)
        )'''
        
        print("loaded feature:", conall_path, self.con_all.shape)

        self.window_size = window_size
        self.stride = stride
        self.windows = []
        
        for vid_idx in range(self.num_videos):
            start, end = map(int, self.nalist[vid_idx])
            video_len = end - start
            
            if video_len <= window_size:
                # Short video: use entire video
                self.windows.append((vid_idx, start, end))
            else:
                # Long video: sliding windows
                for local_start in range(0, video_len - window_size + 1, stride):
                    global_start = start + local_start
                    global_end = global_start + window_size
                    self.windows.append((vid_idx, global_start, global_end))
                
                remainder = (video_len - window_size) % stride
                if remainder > 0:
                    global_start = end - window_size
                    global_end = end
                    if (vid_idx, global_start, global_end) not in self.windows:
                        self.windows.append((vid_idx, global_start, global_end))
        
    
    def __len__(self):
        return len(self.windows)
    

    def __getitem__(self, idx):
        vid_idx, global_start, global_end = self.windows[idx]   
        
        window_features = self.con_all[global_start:global_end].copy() # window 구간의 feature(T, 10, 2048)를 가져와서
        if window_features.ndim == 3:
            window_features = window_features.mean(axis=1) # 10-crop 평균 -> (T, 2048)로 만듦.
        
        window_labels = self.pseudo_labels[global_start:global_end]  # 해당 구간의 pseudo label도 가져온다. (T,) 
        
        window_confidences = self.confidences[global_start:global_end]

        features = torch.from_numpy(window_features.astype(np.float32))
        labels = torch.from_numpy(window_labels)   
        confidences = torch.from_numpy(window_confidences)     

        window_length = len(features)
        
        return features, labels, confidences, window_length


# batch 안의 가장 긴 segment에 맞추어 padding 
def collate_fn_variable_length(batch):

    features_list, labels_list, confidences_list, lengths = zip(*batch)
    
    max_length = max(lengths)
    batch_size = len(batch)
    
    # padding
    features_padded = torch.zeros(batch_size, max_length, args.feature_size)  # (B, max_length, 2048)
    labels_padded = torch.zeros(batch_size, max_length)  # (B, max_length)
    confidences_padded = torch.zeros(batch_size, max_length) 
    masks = torch.zeros(batch_size, max_length)  # (B, max_length)
    
    for i, (feat, label, confidence, length) in enumerate(zip(features_list, labels_list, confidences_list, lengths)):
        features_padded[i, :length] = feat
        labels_padded[i, :length] = label
        confidences_padded[i, :length] = confidence
        masks[i, :length] = 1 
    
    lengths = torch.tensor(lengths, dtype=torch.long)  # (B,)
    
    return features_padded, labels_padded, confidences_padded, masks, lengths