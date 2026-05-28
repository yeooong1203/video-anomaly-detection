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
                                 shape=(self.total_T, 10, 2048))

    def __len__(self):
        return len(self.nalist)

    def __getitem__(self, index):
        a, b = map(int, self.nalist[index])
        x = np.array(self.con_all[a:b], dtype=np.float32)  # (T,10,2048)
        x = x.mean(axis=1) 
        return torch.from_numpy(x)                         # CPU float32


class UCFTrainVideoDataset_Stratified(data.Dataset):
    
    def __init__(self, conall_path, pseudo_path, nalist_path,
                 window_size=2000, stride=2000):
        
        self.nalist = np.load(nalist_path)
        self.num_videos = len(self.nalist)
        
        self.pseudo_labels = np.load(pseudo_path).astype(np.float32)
        self.total_T = len(self.pseudo_labels)

        
        self.con_all = np.memmap(
            conall_path,
            dtype="float32",
            mode="r",
            shape=(self.total_T, 10, 2048)
        )
        
        print("loaded feature:", conall_path, self.con_all.shape)

        assert self.con_all.shape[0] == self.total_T, (
            f"feature T mismatch: feature={self.con_all.shape[0]}, nalist={self.total_T}"
        )

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
        window_features = window_features.mean(axis=1)  # 10-crop 평균 -> (T, 2048)로 만듦.
        
        window_labels = self.pseudo_labels[global_start:global_end]  # 해당 구간의 pseudo label도 가져온다. (T,) 
        
        features = torch.from_numpy(window_features.astype(np.float32))
        labels = torch.from_numpy(window_labels)        

        window_length = len(features)
        
        return features, labels, window_length


# batch 안의 가장 긴 segment에 맞추어 padding 
def collate_fn_variable_length(batch):

    features_list, labels_list, lengths = zip(*batch)
    
    max_length = max(lengths)
    batch_size = len(batch)
    
    # padding
    features_padded = torch.zeros(batch_size, max_length, args.feature_size)  # (B, max_length, 2048)
    labels_padded = torch.zeros(batch_size, max_length)  # (B, max_length)
    masks = torch.zeros(batch_size, max_length)  # (B, max_length)
    
    for i, (feat, label, length) in enumerate(zip(features_list, labels_list, lengths)):
        features_padded[i, :length] = feat
        labels_padded[i, :length] = label
        masks[i, :length] = 1 
    
    lengths = torch.tensor(lengths, dtype=torch.long)  # (B,)
    
    return features_padded, labels_padded, masks, lengths