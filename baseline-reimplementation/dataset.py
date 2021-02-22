import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, triples):
        self.e1 = triples[:,0].tolist()
        self.r = triples[:,1].tolist()
        self.e2 = triples[:,2].tolist()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.e1)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.e1[index],self.r[index],self.e2[index]