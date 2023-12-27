import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
import torchvision
from SpatialTransformer import SpatialTransformer


def defout(cong, x):
    if cong is not None:
        return cong
    else:
        return x


def norm_layer(channels):
    return nn.GroupNorm(32, channels)


class Block(nn.Module):
    @abstractmethod
    def forward(self, x, t):
        """"""


class EmbedBlock(nn.Sequential, Block):
    """"""

    def forward(self, x, t=None, context=None):
        for layer in self:
            if isinstance(layer, Block):
                x = layer(x, t)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upasmple(nn.Module):
    def __init__(self, channels, out_channels=None) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = defout(out_channels, channels)
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.size(1) == self.channels
        x=nn.functional.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = defout(out_channels, channels)
        stride = 2 
        self.conv = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=3, padding=1,stride=stride
        )

    def forward(self, x):
        assert x.size(1) == self.channels
        return self.conv(x)


class ResBlock(EmbedBlock):
    def __init__(self, channels, time_channels, dropout, out_channels=None):
        super().__init__()
        self.channels = channels
        self.time_channels = time_channels
        self.out_channels = defout(out_channels, channels)
        self.dropout = dropout
        self.time_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(time_channels, self.out_channels)
        )
        self.conv = nn.Sequential(
            norm_layer(channels),
            nn.SiLU(),
            nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            norm_layer(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        x = self.conv(x)
        t = self.time_layers(t)
        while len(t.size()) < len(x.size()):
            t = t[..., None]
        h = self.out_conv(x + t)
        return h + x

class gTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int,model_channle:int):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(model_channle, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim , embedding_dim ),
        )

    def forward(self, x: torch.Tensor):
        return self.time_emb(x)


def ge(
    timesteps, embedding_dim: int, downscale_freq_shift: float, max_period: int = 1000
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class UnetModel(nn.Module):
    def __init__(
        self,
        channels,
        model_channels,
        res_block,
        context=None,
        dropout=0.0,
        channel_mutl=(1, 2, 4, 8),
        head=1,
        updown=False,
        out_channels=256,
        device="cuda:0"
     
    ):
        super().__init__()
        self.channels = channels
        self.head = head
        self.time_channels = model_channels*4
        self.time_layers = nn.Sequential(gTimestepEmbedding(self.time_channels,model_channels))

        self.input_layers_one =EmbedBlock(nn.Conv2d(channels, res_block, kernel_size=3, padding=1))
        
        ch = res_block
        input_block_chans = [model_channels]
        self.input_layers=nn.ModuleList()
        for layer, mutl in enumerate(channel_mutl):
            for _ in range(2):
                layers = [
                    ResBlock(ch, self.time_channels, dropout, model_channels * mutl),
                    SpatialTransformer(
                        model_channels * mutl, model_channels * mutl, self.head,context,device=device
                    ),
                ]
                ch = mutl * model_channels
                self.input_layers.append(EmbedBlock(*layers))
                input_block_chans.append(ch)
            if layer != len(channel_mutl)-1 :
                out_ch = ch
                self.input_layers.append(
                    EmbedBlock(Downsample(ch, ch)
                ))
                input_block_chans.append(ch)
            
        self.middle_block=EmbedBlock(
            ResBlock(
                ch,self.time_channels,dropout
            ),
            SpatialTransformer(ch,ch,self.head,context,device=device),
            ResBlock(
                ch,self.time_channels,dropout
            )
        )
        self.output_blocks=nn.ModuleList()
        for layer,mutl in list(enumerate(channel_mutl))[::-1]:
            for i in range(3):
                data=input_block_chans.pop()+ch 
                layers=[
                    ResBlock(
                    data,
                    self.time_channels,
                    dropout,
                    model_channels*mutl
                ),
                SpatialTransformer(
                    model_channels * mutl, model_channels * mutl, self.head, context,device=device
                )
            ]
                ch=model_channels*mutl
                if layer and i == 2:
                    layers.append(
                        Upasmple(
                            ch,ch
                        )
                    )
                self.output_blocks.append(EmbedBlock(*layers))
        self.out=nn.Sequential(
            EmbedBlock(nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1))
        )
    def forward(self,x,timesteps,context=None):
        time_emd=self.time_layers(timesteps)
        hs=[]
        h=self.input_layers_one(x)
        hs.append(h)
        for model in self.input_layers:
            h=model(h,time_emd,context)
            hs.append(h)
        h=self.middle_block(h,time_emd,context)

        for model in self.output_blocks:
            data=hs.pop()
            h=torch.cat([h,data],dim=1)
            h=model(h,time_emd,context)
        return self.out(h)

if __name__=='__main__':
    model = UnetModel(3, 128, 128,512,head=8)