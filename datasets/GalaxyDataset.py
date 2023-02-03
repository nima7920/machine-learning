from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GalaxyDataSet(Dataset):
    def __init__(self,base_dataset,transform):
      self.base_dataset=base_dataset
      self.transform=transform

    def __len__(self):
      return len(self.base_dataset)

    def __getitem__(self,index):
      img=plt.imread('efigi-1.6/png/'+self.base_dataset[index,0]+'.png')
      label=int(self.base_dataset[index,2])
      return self.transform(img),label