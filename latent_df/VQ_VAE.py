import torch 
import torch.nn as nn
from taming.modules.vqvae.quantize import VectorQuantizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms,datasets
from PIL import Image
import os

torch.backends.cudnn.benchmark = True
def norm_layer(channels):
    return nn.GroupNorm(num_groups=32,num_channels=channels, eps=1e-6, affine=True)

def defout(cong, x):
    if cong is not None:
        return cong
    else:
        return x
class ReanetBlock(nn.Module):
    def __init__(self,*,in_channels,out_channels=None,dropout) -> None:
        super().__init__()
        self.in_channels=in_channels 
        out_channels=in_channels if out_channels is None else out_channels
        self.out_channels=out_channels
        self.norm_layer1=norm_layer(in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.droput=nn.Dropout(dropout)
        self.norm_layer2=norm_layer(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv_shortcut=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        h=x
        h=self.norm_layer1(h)
        h=h*torch.sigmoid(h)
        h=self.conv1(h)
        h=self.norm_layer2(h)
        h=h*torch.sigmoid(h)
        h=self.conv2(h)
        x=self.conv_shortcut(x)
        
        return h+x
class AttnBlock(nn.Module):
    def __init__(self, q_dim,device,head=1,dropout=0.1):
        super().__init__()
        self.head=head
        self.socale=torch.sqrt(torch.FloatTensor([q_dim // head])).to(device)
        self.norm=norm_layer(q_dim)
        self.softmax = nn.Softmax(dim=1)
        self.Q=nn.Linear(q_dim,q_dim,bias=False)
    
        self.K=nn.Linear(q_dim,q_dim,bias=False)
        self.V=nn.Linear(q_dim,q_dim,bias=False)
        self.proj=nn.Sequential(
            nn.Linear(q_dim,q_dim),
            nn.Dropout(dropout)
        )
    def forward(self,x,context=None,mask=None):
        b,c,h,w=x.size()
        x=self.norm(x)
        x_=x.view(b,c,h*w).transpose(1,2)
       
        head=self.head
        Q=self.Q(x_)
        K=self.K(x_)
        V=self.V(x_)
        Q=Q.view(b,h*w,head,-1)
        K=K.view(b,h*w,head,-1)
        V=V.view(b,h*w,head,-1)
        dots = torch.einsum('bhid,bhjd->bhij', Q, K) * self.socale
        if mask is not None:
            dots=dots.masked_fill(mask==0,float("-inf"))
        dots=self.softmax(dots)
        out=torch.matmul(dots,V).reshape(b,-1,c)
        out=self.proj(out)
        return out.transpose(1,2).reshape(b,c,h,w)+x
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
class Encoder(nn.Module):
    def __init__(self,*, ch,out_ch,num_res_blocks,ch_mult=(1,2,4,8),in_channels,dropout,device) :
        super().__init__()
        self.ch=ch
        self.num_resolutions=len(ch_mult)
        self.num_res_blocks=num_res_blocks
        self.in_channels=in_channels
        in_ch_mult = (1,)+tuple(ch_mult)
        self.conv_in=nn.Conv2d(self.in_channels,self.ch,kernel_size=3,stride=1,padding=1)
        self.down=nn.ModuleList()
        for i in range(self.num_resolutions):
            block=nn.ModuleList()
            attn=nn.ModuleList()
            block_in=ch*in_ch_mult[i]
            block_out=ch*ch_mult[i]
            for i_block in range(self.num_res_blocks):
                block.append(ReanetBlock(in_channels=block_in,out_channels=block_out,dropout=dropout)
                )
                block_in=block_out
                attn.append(AttnBlock(block_in,device))
            down=nn.Module()
            down.block=block
            down.attn=attn
            if i !=self.num_resolutions-1:
                down.downsample=Downsample(block_in)
            self.down.append(down)
    def forward(self,x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        return hs[-1]
class Decoder(nn.Module):
    def __init__(self,*, ch,out_ch,num_res_blocks,ch_mult=(1,2,4,8),in_channels,dropout,z_channels,device) :
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels


    
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.up=nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ReanetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                
                attn.append(AttnBlock(block_in,device))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 1:
                up.upsample = Upasmple(block_in)
               
            self.up.insert(0, up)
            
        self.norm_out = norm_layer(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    def forward(self,z):
        h=self.conv_in(z)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 1:
                h = self.up[i_level].upsample(h)
        h=self.norm_out(h)
        h=self.conv_out(h)
        return h
class VQ_VAE(nn.Module):
    def __init__(self,*, ch,out_ch,num_res_blocks,ch_mult=(1,2,4,8),in_channels,dropout,z_channels,device) -> None:
        super().__init__()
        self.encoder=Encoder(ch=ch,out_ch=out_ch,num_res_blocks=num_res_blocks,ch_mult=(1,2,4,8),in_channels=in_channels,dropout=dropout,device=device)
        self.decoder=Decoder(ch=ch,out_ch=out_ch,num_res_blocks=num_res_blocks,in_channels=in_channels,dropout=dropout,z_channels=z_channels,device=device)
        self.quantize=VectorQuantizer(z_channels,32,beta=0.25)
        self.quant_conv=nn.Conv2d(z_channels,z_channels,1)
        self.post_quant_conv =nn.Conv2d(z_channels,z_channels,1)
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff
class Celeb(Dataset):
    def __init__(self,root,img_shape=(512,512))->None:
        super().__init__()
        self.root=root
        self.img_shape=img_shape
        self.filenames=sorted(os.listdir(root))
    def __len__(self) -> int:
        return len(self.filenames)
    def __getitem__(self,index:int):
        path=os.path.join(self.root,self.filenames[index])
        img=Image.open(path).convert('RGB')
        pipeline =transforms.Compose([
            
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)
def get_dataloader(root='/home/ubuntu/images/train2017',**kwargs):
    dataset=Celeb(root,**kwargs)

    return DataLoader(dataset=dataset,batch_size=1,num_workers=2)

def loss_vq(x,y,diff):
    recos=torch.nn.functional.mse_loss(y,x)
    return recos+diff
if __name__=='__main__':
    model=VQ_VAE(ch=32,out_ch=3,num_res_blocks=2,in_channels=3,dropout=1.,z_channels=256,device="cpu")