import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange

def defout(cong,x):
    if cong is not None:
        return cong
    else :
        return x
class GEGLU(nn.Module): #GeGLU(x,W,V,b,c)=GELU(xW+b)*(xV+c)
    # https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/attention.py#L49
    def __init__(self, dim_in,dim_out) :
        super().__init__()
        self.proj=nn.Linear(dim_in,dim_out*2)

    def forward(self,x):
        x,gatl=self.proj(x).chunk(2,dim=-1)
        return x*nn.functional.gelu(gatl)
    
class FeedForward(nn.Module):
    def __init__(self,channels,mult=4,dropout=0.,glu=False) -> None:
        super().__init__()
        self.hidden_size=channels*mult
        self.proj=nn.Sequential(
            nn.Linear(channels,self.hidden_size),
            nn.GELU()
        )if glu else GEGLU(channels,self.hidden_size)

        self.net=nn.Sequential(
            GEGLU(channels,self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size,channels)
        )
    def forward(self,x):
        return self.net(x)
def norm_layer(channels):
    return nn.GroupNorm(32,channels)
class SelfAttention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.norm=nn.Softmax2d(channels)
        self.q=nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k=nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v=nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj=nn.Conv2d(
            channels,
            3,
            kernel_size=1,
            stride=1,
            padding=0

        )
    def forward(self,x):
        b,c,h,w=x.size()
        input=x
        input=self.norm(input)
        Q=self.q(input)
        K=self.k(input)
        V=self.v(input)
        Q=Q.view(b,c,h*w).transpose(1,2)
        K=K.view(b,c,h*w)
        V=V.view(b,c,h*w)
        attn=torch.matmul(Q,K)
        output=torch.matmul(V,attn.transpose(1,2))
        output=output.view(b,c,h,w)
        output=self.proj(output)
        return output+x

class CrossAttention(nn.Module):
    def __init__(self, q_dim,context=None,head=16,dropout=0.,device="cuda:0"):
        super().__init__()
        self.head=head
        self.device=device
        self.socale=torch.sqrt(torch.FloatTensor([q_dim // head])).to(device)
        context=defout(context,q_dim)
        self.softmax = nn.Softmax(dim=1)
        self.Q=nn.Linear(q_dim,q_dim,bias=False)
    
        self.K=nn.Linear(context,q_dim,bias=False)
        self.V=nn.Linear(context,q_dim,bias=False)
        self.proj=nn.Sequential(
            nn.Linear(q_dim,q_dim),
            nn.Dropout(dropout)
        )
    def forward(self,x,context=None,mask=None):
        
        h=self.head
        Q=self.Q(x)
        
        context=defout(context,x)
      
        b,h_w,c=x.size()

        bt,seq,hidim=context.size()
        K=self.K(context)
        V=self.V(context)
        Q=Q.view(b,h_w,h,-1).transpose(1, 2)
        K=K.view(bt,seq,h,-1).transpose(1, 2)
        V=V.view(bt,seq,h,-1).transpose(1, 2)
        dots = torch.einsum('bhid,bhjd->bhij', Q, K) * self.socale
        if mask is not None:
            dots=dots.masked_fill(mask==0,float("-inf"))
        dots=self.softmax(dots)

        out=torch.matmul(dots,V).reshape(b,-1,c)
        out=self.proj(out)
        return out
class BasicTransformer(nn.Module):
    def __init__(self,dim,heads,context,dropout=0.,glu=False,device="cuda:0"):
        super().__init__()
        self.attn1=CrossAttention(dim,context,dropout=dropout,head=heads,device=device)
        self.attn2=CrossAttention(dim,context,dropout=dropout,head=heads,device=device)
        self.feed=FeedForward(dim,dropout=dropout,glu=glu)
        self.norm0=nn.LayerNorm(dim)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
    def forward(self,x,context):
        x=self.attn1(self.norm0(x),context)+x
        x=self.attn2(self.norm0(x),context)+x
        x=self.feed(self.norm2(x))+x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self,dim,out_dim,head,context=None,glu=False,dropout=0.,depath=16,device="cuda:0"):
        super().__init__()
        context=defout(context,dim)
        self.norm=norm_layer(dim)
        self.proj=nn.Conv2d(
                dim,
                out_dim,
                kernel_size=1,
                stride=1,
                padding=0
        )
        self.transformer_blok=nn.ModuleList(
            [
                BasicTransformer(out_dim,heads=head,dropout=dropout,context=context,glu=glu,device=device)
             for _ in range(depath)]
        )
        self.proj_out=nn.Conv2d(
            out_dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
    def forward(self,x,context=None):
        in_x=x
    
        b,c,h,w=x.size()
        x=self.norm(x)
        x=self.proj(x)
        x=rearrange(x,'b c h w->b (h w) c')
        context=defout(context,x)
        for i in self.transformer_blok:
            x=i(x,context)
        x=rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
        x=self.proj_out(x)
        return x+in_x
if __name__=='__main__':
    model=SpatialTransformer(128,512,8,context=512,device="cpu")
