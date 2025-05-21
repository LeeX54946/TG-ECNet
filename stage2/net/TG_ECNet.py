import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from net.TaskMoE import TaskMoE
from einops import rearrange
from einops.layers.torch import Rearrange
import time



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res

class Taskprompt(nn.Module):
    def __init__(self, in_dim, atom_num=32, atom_dim=256):
        super(Taskprompt, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)

        return out

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt



class TG_ECNet(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = True,
    ):

        super(TG_ECNet, self).__init__()
        atom_dim = 256
        atom_num = 32
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)     
        self.decoder = decoder      
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)             
        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])       
        self.down1_2 = Downsample(dim) 
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])        
        self.down2_3 = Downsample(int(dim*2**1)) 
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2)) 
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])       
        self.up4_3 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])       
        self.up2_1 = Upsample(int(dim*2**1))  
        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])     
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])             
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.task_prompt = Taskprompt(in_dim=inp_channels, atom_num=atom_num, atom_dim=atom_dim)
        self.degradmoe1 = TaskMoE(atom_dim=atom_dim, dim=dim,ffn_expansion_factor=ffn_expansion_factor,num_experts = 11, use_experts = 6)
        self.degradmoe2 = TaskMoE(atom_dim=atom_dim, dim=int(dim * 2 ** 1),ffn_expansion_factor=ffn_expansion_factor,num_experts = 11, use_experts = 6)
        self.degradmoe3 = TaskMoE(atom_dim=atom_dim, dim=int(dim * 2 ** 2),ffn_expansion_factor=ffn_expansion_factor,num_experts = 11, use_experts = 6)

        self.param_beta = nn.Parameter(torch.ones(int(dim*2**1), 1, 1))
        self.param_alpha = nn.Parameter(torch.ones(int(dim*2**1), 1, 1))
        self.param_theta = nn.Parameter(torch.ones(int(out_channels), 1, 1))
        self.param_gamma = nn.Parameter(torch.ones(int(out_channels), 1, 1))
        self.Refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.Output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.task_promptfusion = Taskprompt(in_dim=int(dim*2**1), atom_num=atom_num, atom_dim=atom_dim)
        self.fusionmoe1 = TaskMoE(atom_dim=atom_dim, dim=int(dim*2**1),ffn_expansion_factor=ffn_expansion_factor,num_experts = 11, use_experts = 6)
        self.Decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

    def forward(self, inp_img, ir_img):

        task_prompt = self.task_prompt(inp_img)
        task_promptir = self.task_prompt(ir_img)
                
        inp_enc_level1 = self.patch_embed(inp_img)
        ir_enc_level1 = self.patch_embed(ir_img)


        task_harmonization_output1, loss_tmp = self.degradmoe1(inp_enc_level1 , task_prompt)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1) 
        task_harmonization_output2, loss_tmp = self.degradmoe2(inp_enc_level2 , task_prompt)
        out_enc_level2 = self.encoder_level2(task_harmonization_output2.half())
        inp_enc_level3 = self.down2_3(out_enc_level2)
        task_harmonization_output3, loss_tmp = self.degradmoe3(inp_enc_level3 , task_prompt)
        out_enc_level3 = self.encoder_level3(task_harmonization_output3.half())
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        
        task_harmonization_output1, loss_tmp = self.degradmoe1(ir_enc_level1 , task_promptir)
        out_ir_level1 = self.encoder_level1(ir_enc_level1)
        ir_enc_level2 = self.down1_2(out_ir_level1) 
        task_harmonization_output2, loss_tmp = self.degradmoe2(ir_enc_level2 , task_promptir)
        out_ir_level2 = self.encoder_level2(task_harmonization_output2.half())
        ir_enc_level3 = self.down2_3(out_ir_level2)
        task_harmonization_output3, loss_tmp = self.degradmoe3(ir_enc_level3 , task_promptir)
        out_ir_level3 = self.encoder_level3(task_harmonization_output3.half())
        ir_enc_level4 = self.down3_4(out_ir_level3)
        latentir = self.latent(ir_enc_level4)  
        
               
        if self.decoder:
            rgb3_param = self.prompt3(latent)

            latent = torch.cat([latent, rgb3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)     
        if self.decoder:
            ir3_param = self.prompt3(latentir)

            latentir = torch.cat([latentir, ir3_param], 1)
            latentir = self.noise_level3(latentir)
            latentir = self.reduce_noise_level3(latentir)      
        rgbi3 = self.up4_3(latent)
        rgbi3 = torch.cat([rgbi3, out_enc_level3], 1)
        rgbi3 = self.reduce_chan_level3(rgbi3)
        rgbo3 = self.decoder_level3(rgbi3)
        
        iri3 = self.up4_3(latentir)
        iri3 = torch.cat([iri3, out_ir_level3], 1)
        iri3 = self.reduce_chan_level3(iri3)
        iro3 = self.decoder_level3(iri3)

        if self.decoder:
            rgb2_param = self.prompt2(rgbo3)
            rgbo3 = torch.cat([rgbo3, rgb2_param], 1)
            rgbo3 = self.noise_level2(rgbo3)
            rgbo3 = self.reduce_noise_level2(rgbo3)
            
        if self.decoder:
            ir2_param = self.prompt2(iro3)
            iro3 = torch.cat([iro3, ir2_param], 1)
            iro3 = self.noise_level2(iro3)
            iro3 = self.reduce_noise_level2(iro3)

        rgbi2 = self.up3_2(rgbo3)
        rgbi2 = torch.cat([rgbi2, out_enc_level2], 1)
        rgbi2 = self.reduce_chan_level2(rgbi2)
        rgbo2 = self.decoder_level2(rgbi2)        
        
        iri2 = self.up3_2(iro3)
        iri2 = torch.cat([iri2, out_ir_level2], 1)
        iri2 = self.reduce_chan_level2(iri2)
        iro2 = self.decoder_level2(iri2)       
        
        if self.decoder:           
            rgb1_param = self.prompt1(rgbo2)
            rgbo2 = torch.cat([rgbo2, rgb1_param], 1)
            rgbo2 = self.noise_level1(rgbo2)
            rgbo2 = self.reduce_noise_level1(rgbo2)        
        if self.decoder:           
            ir1_param = self.prompt1(iro2)
            iro2 = torch.cat([iro2, ir1_param], 1)
            iro2 = self.noise_level1(iro2)
            iro2 = self.reduce_noise_level1(iro2) 
            
            
        rgbi1 = self.up2_1(rgbo2)
        rgbi1 = torch.cat([rgbi1, out_enc_level1], 1)
        rgbo1 = self.decoder_level1(rgbi1)
        
        iri1 = self.up2_1(iro2)
        iri1 = torch.cat([iri1, out_ir_level1], 1)
        iro1 = self.decoder_level1(iri1)
        

        out_fu_level1 = self.param_alpha * rgbo1 + self.param_beta * iro1
        task_promptfuse = self.task_promptfusion(out_fu_level1)
        task_harmonization_output3, loss_tmp = self.fusionmoe1(out_fu_level1 , task_promptfuse)
        out_fu_level1 = self.Decoder_level1(task_harmonization_output3)        
        
        rgbo1 = self.refinement(rgbo1)
        rgbo1 = self.output(rgbo1)
        re  = rgbo1 + inp_img

        iro1 = self.refinement(iro1)
        iro1 = self.output(iro1)
        irir  = iro1 + ir_img
        
        out_dec_level1 = self.Refinement(out_fu_level1)
        out_dec_level1 = self.Output(out_dec_level1) + self.param_theta * re + self.param_gamma * irir




        if self.training:
            return out_dec_level1,loss_tmp
        else:
            return out_dec_level1