import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

class ab_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2=None, op_token3=None):
        self.data = self.generate_data(p, eq_token, op_token1)
    
    def generate_data(self, p, eq_token, op_token1):
        """
        (a*b) % p for 0 <= a < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        result = (a * b) % p
        return torch.stack([a, op1, b, eq, result])

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]

class a_plus_b_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2=None, op_token3=None):
        self.data = self.generate_data(p, eq_token, op_token1)
    
    def generate_data(self, p, eq_token, op_token1):
        """
        (a*b) % p for 0 <= a < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        result = (a + b) % p
        return torch.stack([a, op1, b, eq, result])

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    
class a_minus_b_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2=None, op_token3=None):
        self.data = self.generate_data(p, eq_token, op_token1)
    
    def generate_data(self, p, eq_token, op_token1):
        """
        (a*b) % p for 0 <= a < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        result = (a - b) % p
        return torch.stack([a, op1, b, eq, result])

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]

class aa_sub_b_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2, op_token3=None):
        self.data = self.generate_data(p, eq_token, op_token1, op_token2)
    
    def generate_data(self, p, eq_token, op_token1, op_token2):
        """
        (a-b+c) % p for 0 <= a, c < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1,p)
        a, b = torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        op2 = torch.ones_like(a) * op_token2
        result = (a * a - b) % p
        return torch.stack([a, op1, a, op2, b, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    
class ac_plus_bd_sub_e_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2, op_token3):
        self.data = self.generate_data(p, eq_token, op_token1, op_token2, op_token3)
    
    def generate_data(self, p, eq_token, op_token1, op_token2, op_token3):
        """
        (a*c+b*d-e) % p for 0 <= a, c < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(p)
        c = torch.arange(1, p)
        d = torch.arange(1, p)
        e = torch.arange(p)
        a, b, c, d, e = torch.cartesian_prod(a, b, c, d, e).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        op2 = torch.ones_like(a) * op_token2
        op3 = torch.ones_like(a) * op_token3
        result = (a * c + b * d - e) % p
        return torch.stack([a, op1, b, op2, c, op1, d, op3, e, eq, result])

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    
TASK2DATASET = {
        "a+b": a_plus_b_mod_p_data,
        "a-b": a_minus_b_mod_p_data,
        "axb": ab_mod_p_data,
        "axa-b": aa_sub_b_mod_p_data,
        "axc+bxd-e": ac_plus_bd_sub_e_mod_p_data,
        }

def get_dataset(task_type:str = "ac+bd-e",
                p:int = 7,):
    assert task_type in TASK2DATASET.keys(), f"NotImplementedError: {task_type}"
    eq_token = p # token id
    op_token1 = p + 1
    op_token2 = p + 2
    op_token3 = p + 3
    
    seq_len = len(task_type) + 2
    dataset = TASK2DATASET[task_type](p=p,
                                      eq_token=eq_token,
                                      op_token1=op_token1,
                                      op_token2=op_token2,
                                      op_token3=op_token3)
    return dataset, seq_len

def get_dataloader(dataset,
                   p_train=0.5, 
                   p_outer=0., 
                   batch_size=512):
    assert p_outer < p_train
    train_size = int(p_train * len(dataset))
    test_size = len(dataset) - train_size
    outer_size = int(p_outer * len(dataset))
    inner_size = train_size - outer_size
    
    inner_data, outer_data, test_data = torch.utils.data.random_split(dataset, [inner_size, outer_size, test_size])

    inner_loader = DataLoader(inner_data,
                              batch_size=batch_size,
                              shuffle=True)
    outer_loader = DataLoader(outer_data,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_data,
                              batch_size=batch_size,
                              shuffle=False)
    return inner_loader, outer_loader, test_loader

if __name__ == "__main__":
    dataset, seq_len = get_dataset("a+b", p=97)
    inner_loader, outer_loader, test_loader = get_dataloader(dataset, p_train=0.5, p_outer=0.01)
    import pdb
    pdb.set_trace()