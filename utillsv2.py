import numpy as np
import torch
import option


args = option.parser.parse_args()


def Concat_list_all_crop_feedback(Test=False, create='False'):  # UCF
    
    if Test is True:
        con_test = np.memmap(args.test_conall_path, dtype='float32', mode='r', shape=(290, 32, 10, 2048)).copy()
        print('Testset size:', con_test.shape)
        
        return con_test
    

    if create == 'True':
        print('loading Pseudo Labels......',args.pseudofile )

        label_all = np.load(args.pseudofile)
        print('[*] concatenated labels shape:',label_all.shape)

        return len(label_all), torch.tensor(label_all)

