import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c
    def forward(self, x):
        """x: (B,F,T,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)
        return x_c.permute(0,3,2,1).contiguous()

class FC(nn.Module):
    """feature catcher (FC) block"""
    def __init__(self, in_channels, out_channels, kernel_size=(1,1), stride=(1,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.soft = nn.Softmax(dim=-2)
        self.sig = nn.Sigmoid()
    
    def forward(self, f, m):
        """
        f : (B,C,T,F)"""
        # global-dependency
        f1 = self.conv(f)
        f2 = self.conv(f)
        f1 = self.bn(f1)
        f2 = self.bn(f2)
        k = self.prelu(f1)
        v = self.prelu(f2)
        x = torch.mul(k, v)
        x = self.soft(x)
        g = torch.mul(v, x)

        # local-dependency
        m = torch.mul(m, f)
        m = self.conv(m)
        m = self.bn(m)
        q = self.prelu(m)
        y = torch.mul(q, g)
        y = self.soft(y)
        l = torch.mul(y, g)
        return l


class TNB(nn.Module):
    """Target Negative Block"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = FC(input_size, output_size)
    
    def forward(self, f, m):
        """
        f: (B,C,T,F)"""
        mtn = torch.sub(1.0, m)
        ftn = self.fc(f, mtn)
        return ftn


class TPB(nn.Module):
    """Target Positive Block"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = FC(input_size, output_size)
    
    def forward(self, f, m):
        """
        f: (B,C,T,F)"""
        f = self.fc(f, m)
       
        return f


class IB(nn.Module):
    """Interactive Block"""
    def __init__(self, input_size, hidden_size = 1):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(input_size, input_size, batch_first=True, bidirectional=False)
        self.soft = nn.Softmax(dim=1)
    
    def forward(self, ftn, ftp):
        """
        ftn: (B,C,T,F)"""
        f = self.mean(ftn + ftp)
        ff = rearrange(f, 'b c t f -> b t (c f)')
        f1 = self.gru(ff)[0]
        f2 = self.gru(ff)[0]
        wtp, wtn = self.soft(torch.cat([f1, f2]))
        wtp = wtp.unsqueeze(-1).unsqueeze(-1)
        wtn = wtn.unsqueeze(-1).unsqueeze(-1)
        fout = torch.add(wtp*ftp, wtn*ftn)

        return fout
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4,3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
    def forward(self, x):
        """x: (B,C,T,F)"""
        y = self.elu(self.bn(self.conv(self.pad(x))))
        return y + x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)
    def forward(self, x):
        return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))


class CM(nn.Module):
    """Collaboration Module"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,7), stride=(1,1)):
        super().__init__()
        self.pad = nn.ZeroPad2d([3,3,1,1])
        self.tnb = TNB(in_channels, out_channels)
        self.tpb = TPB(in_channels, out_channels)
        self.ib = IB(out_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.sig = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.fe = FE()
        self.en = EncoderBlock(2, 64)
        
    def forward(self, fin):
        """fin: (B,F,T,2)"""
        fin = self.fe(fin)
        fin = self.en(fin)
        fin1 = self.conv(self.pad(fin))
        fin1 = self.prelu(fin1)
        fin1 = self.conv1(self.pad(fin1))
        mtp = self.sig(fin1)

        ftp = self.tpb(fin, mtp)
        ftn = self.tnb(fin, mtp)

        fout = self.ib(ftn, ftp)
        
        return fout



if __name__ == "__main__":
    device = torch.device('cpu')
    x = torch.randn(1, 160000)
    x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    model = CM(64, 32).eval()
    y = model(x)

    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 257, 100, 2)
    b = torch.randn(1, 257, 100, 2)
    y1 = model(a)
    y2 = model(b)
    print((y1[:,:,:100,:] - y2[:,:,:100,:]).abs().max())
    print((y1[:,:,100:,:] - y2[:,:,100:,:]).abs().max())
        