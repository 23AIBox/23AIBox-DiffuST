from torch.utils.data import DataLoader,SubsetRandomSampler,Dataset
import torch

class myDataset(Dataset):
    def __init__(self, feature,condition):


        condition = torch.tensor(condition)
        self.model_input = feature.reshape((feature.shape[0],1,feature.shape[1]))
        self.cond_input = condition.reshape((condition.shape[0],1,condition.shape[1]))


            
    def __getitem__(self, index):
        xA = self.model_input[index]
        xB = self.cond_input[index]
        return xA,xB
    
    def __len__(self):
        return len(self.model_input)