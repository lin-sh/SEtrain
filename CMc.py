import torch
import torch.nn as nn
import numpy as np
from einops import rearrange




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
        self.fc = FC(input_size, input_size)
    
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
        self.fc = FC(input_size, input_size)
    
    def forward(self, f, m):
        """
        f: (B,C,T,F)"""
        f = self.fc(f, m)
       
        return f


class IB(nn.Module):
    """Interactive Block"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.soft = nn.Softmax(dim=-1)
    
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


class CM(nn.Module):
    """Collaboration Module"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,7), stride=(1,1)):
        super().__init__()
        self.pad = nn.ZeroPad2d([3,3,1,1])
        self.tnb = TNB(in_channels, out_channels)
        self.tpb = TPB(in_channels, out_channels)
        self.ib = IB(in_channels, 1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.sig = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.fe = FE()
        self.en = EncoderBlock(2, 64)
        
    def forward(self, fin):
        """fin: (B,F,T,2)"""
        fin1 = self.conv(self.pad(fin))
        fin1 = self.prelu(fin1)
        fin1 = self.conv1(self.pad(fin1))
        mtp = self.sig(fin1)

        ftp = self.tpb(fin, mtp)
        ftn = self.tnb(fin, mtp)

        fout = self.ib(ftn, ftp)
        
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


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        """x : (B,C,T,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y = self.gru(y)[0]
        y = self.fc(y)
        y = rearrange(y, 'b t (c f) -> b c t f', c=x.shape[1])
        return y
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size)
        
    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last
    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y
    

class CCM(nn.Module):
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        self.v = torch.tensor([1, -1/2 + 1j*np.sqrt(3)/2, -1/2 - 1j*np.sqrt(3)/2])
        self.unfold = nn.Sequential(nn.ZeroPad2d([1,1,2,0]),
                                    nn.Unfold(kernel_size=(3,3)))
    
    def forward(self, m, x):
        """
        m: (B,27,T,F)
        x: (B,F,T,2)"""
        m = rearrange(m, 'b (r c) t f -> b r c t f', r=3)
        H = torch.sum(self.v.to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F), complex
        M = rearrange(H, 'b (m n) t f -> b m n t f', m=3)  # (B,m,n,T,F), complex
        
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F), real
        x = torch.complex(x[:,0], x[:,1])    # (B,T,F), complex
        x_unfold = self.unfold(x[:,None])
        x_unfold = rearrange(x_unfold, 'b (m n) (t f) -> b m n t f', m=3,f=x.shape[-1])
        
        x_enh = torch.sum(M * x_unfold, dim=(1,2))  # (B,T,F), complex
        x_enh = torch.stack([x_enh.real, x_enh.imag], dim=3).transpose(1,2).contiguous()
        return x_enh

class AlignBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, delay=100):
        super().__init__()
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.unfold = nn.Sequential(nn.ZeroPad2d([0,0,delay-1,0]),
                                    nn.Unfold((delay, 1)))
        self.conv = nn.Sequential(nn.ZeroPad2d([1,1,4,0]),
                                  nn.Conv2d(hidden_channels, 1, (5,3)))
        
        
    def forward(self, x_mic, x_ref):
        """
        x_mic: (B,C,T,F)
        x_ref: (B,C,T,F)
        """
        Q = self.pconv_mic(x_mic)  # (B,H,T,F)
        K = self.pconv_ref(x_ref)  # (B,H,T,F)
        Ku = self.unfold(K)        # (B, H*D, T*F)
        Ku = Ku.view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
            .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)      # (B,H,T,D)
        V = self.conv(V)           # (B,1,T,D)
        A = torch.softmax(V, dim=-1)[..., None]  # (B,1,T,D,1)
        
        y = self.unfold(x_ref).view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
                .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        y = torch.sum(y * A, dim=-2)
        return y

class DeepCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FE()
        
        self.align = AlignBlock(64, 64)
        self.enblockRef1 = EncoderBlock(2, 32)
        self.enblockRef2 = EncoderBlock(32, 64)

        self.enblock1 = EncoderBlock(2, 32)
        self.enblock2 = EncoderBlock(32, 64)
        self.enblock3 = EncoderBlock(128, 128)
        
        self.bottle = CM(128, 64)
        
        self.deblock3 = DecoderBlock(128, 64)
        self.deblock2 = DecoderBlock(64, 32)
        self.deblock1 = DecoderBlock(32, 27)
        self.ccm = CCM()
        
    def forward(self, x, y):
        en_y0 = self.fe(y)            
        en_y1 = self.enblockRef1(en_y0)  
        en_y2 = self.enblockRef2(en_y1)  
        
        """x: (B,F,T,2)"""
        en_x0 = self.fe(x)            # ; print(en_x0.shape)
        en_x1 = self.enblock1(en_x0)  # ; print(en_x1.shape)
        en_x2 = self.enblock2(en_x1)  # ; print(en_x2.shape)

        en_xy0 = self.align(en_x2, en_y2)  # ; print(en_x2.shape)
        # concat en_xy0 and en_x2
        en_xy1 = torch.cat([en_x2, en_xy0], dim=1)

        en_x3 = self.enblock3(en_xy1)  # ; print(en_x3.shape)

        en_xr = self.bottle(en_x3)    # ; print(en_xr.shape)
        
        de_x3 = self.deblock3(en_xr, en_x3)[..., :en_x2.shape[-1]]  # ; print(de_x3.shape)
        de_x2 = self.deblock2(de_x3, en_x2)[..., :en_x1.shape[-1]]  # ; print(de_x2.shape)
        de_x1 = self.deblock1(de_x2, en_x1)[..., :en_x0.shape[-1]]  # ; print(de_x1.shape)
        
        x_enh = self.ccm(de_x1, x)  # (B,F,T,2)
        
        return x_enh




if __name__ == "__main__":
    device = torch.device('cpu')
    x = torch.randn(1, 160000)
    x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x1 = torch.randn(1, 160000)
    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    model = DeepCM().eval()
    y = model(x, x1)

    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 626, 2), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print(flops, params)
        