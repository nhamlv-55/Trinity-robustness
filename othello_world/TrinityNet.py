from torch import nn
import torch

class TrinityNet(nn.Module):
    def __init__(self, probe: nn.Module|None, head: nn.Module|None, 
                 n_embd:int = 512, vocab_size:int = 61, mid_dim:int = 256,
                 load:bool = True):
        super().__init__()
        #the probe head
        self.probe = nn.Sequential(
            nn.Linear(n_embd, mid_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(mid_dim, 64*3, bias=True),
        )
        #the logits head
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        #set weight
        if not load:
            with torch.no_grad():
                self.probe[0].weight = nn.Parameter(probe.proj[0].weight)
                self.probe[0].bias = nn.Parameter(probe.proj[0].bias)
                
                self.probe[2].weight = nn.Parameter(probe.proj[2].weight)
                self.probe[2].bias = nn.Parameter(probe.proj[2].bias)
            with torch.no_grad():
                self.head.weight = nn.Parameter(head.weight)
        
    def forward(self, h):
        logits = self.head(h)
        probe = self.probe(h)
        return torch.cat([logits, probe], dim = -1)