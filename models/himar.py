from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
from timm.layers import PatchEmbed, Mlp, DropPath, RmsNorm
from timm.models.vision_transformer import LayerScale, Attention
from models.diffloss import DiffLoss, GlobalDiffLoss
import scipy.stats as stats


def modulate3(x, scale):
    return x * (1 + scale)

class MaskRatioGenerator:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
    def rvs(self, i):
        r = torch.rand(i) * (self.max_val - self.min_val) + self.min_val
        mask_rate = torch.cos(r * math.pi * 0.5)
        return mask_rate

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

def mask_by_order_step(mask_len, order, bsz, seq_len, cond_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    masking[:,cond_len:] = 1
    return masking

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            num_tasks: int = 2,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adaLN_scale = nn.Parameter(torch.zeros((num_tasks, 1, 1, 4*dim)), requires_grad=True)
        self.adaLN_shift = nn.Parameter(torch.cat([torch.zeros((num_tasks, 1, 1, dim)),torch.ones((num_tasks, 1, 1, dim)),torch.zeros((num_tasks, 1, 1, dim)),torch.ones((num_tasks, 1, 1, dim))],dim=-1), requires_grad=True)
    def forward(self, x: torch.Tensor, c: torch.Tensor, task: int) -> torch.Tensor:
        c = c * self.adaLN_scale[task] + self.adaLN_shift[task]
        scale_msa, gate_msa, scale_mlp, gate_mlp = c.chunk(4, dim=-1)
        x = x + gate_msa * self.drop_path1(self.ls1(self.attn(modulate3(self.norm1(x), scale_msa))))
        x = x + gate_mlp * self.drop_path2(self.ls2(self.mlp(modulate3(self.norm2(x), scale_mlp))))
        return x

class HiMAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 cond_drop_prob=0.5,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 diff_seqlen=False,
                 step_warmup=100,
                 step_stage2_rate=0.5,
                 cond_scale=4,
                 cond_dim=16,
                 two_diffloss=False,
                 seq_len=None,
                 global_dm=False,
                 gdm_w=768,
                 gdm_d=3,
                 head=8,
                 ratio=4,
                 cos=True,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.two_diffloss = two_diffloss
        self.global_dm = global_dm
        self.cond_scale = cond_scale
        self.cond_dim = cond_dim
        self.step_warmup = step_warmup
        self.step_stage2_rate = step_stage2_rate
        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.diff_seqlen = diff_seqlen
        self.seq_len = self.seq_h * self.seq_w if seq_len is None else seq_len
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.decoder_depth=decoder_depth
        self.epoch = 0
        self.gdm_w = gdm_w
        self.gdm_d = gdm_d
        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.cond_drop_prob = cond_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        if cos:
            self.mask_ratio_generator2 = MaskRatioGenerator()
        else:
            self.mask_ratio_generator2 = stats.truncnorm((0.38 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        
        # token --> hidden state
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        # cond part
        # conditioned on hidden state
        self.cond_z_proj = nn.Linear(decoder_embed_dim, encoder_embed_dim, bias=True)
        self.buffer_size = buffer_size
        # conditioned on fake latent
        self.cond_tokens = nn.Parameter(torch.zeros(1, self.cond_scale*self.cond_scale, self.cond_dim))
        self.cond_proj = nn.Linear(self.cond_dim, encoder_embed_dim, bias=True)
        
        self.seq_len = self.seq_len + self.cond_scale*self.cond_scale
        self.small_seqlen = self.seq_len
        self.mask_token = nn.Parameter(torch.zeros(1, self.small_seqlen+self.seq_len, encoder_embed_dim))

        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.small_seqlen + 2*self.buffer_size, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
                Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                    proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_task_emb = nn.Embedding(2, encoder_embed_dim)
        self.encoder_blocks_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(encoder_embed_dim, 4 * encoder_embed_dim, bias=True)
        ) 
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.small_seqlen + 2*self.buffer_size, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                    proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_task_emb = nn.Embedding(2, decoder_embed_dim)
        self.decoder_blocks_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim, bias=True)
        ) 
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.cond_diffloss = GlobalDiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=gdm_w,
            depth=gdm_d,
            num_sampling_steps=num_sampling_steps,
            head=head,
            ratio=ratio,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.cond_tokens, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]
    
    def unpatchify_small(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.cond_dim
        h_, w_ = self.cond_scale, self.cond_scale

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]
    
    def sample_orders_step(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for i in range(bsz):
            order = list(range(self.cond_scale*self.cond_scale))
            np.random.shuffle(order)
            order1 = list(range(self.cond_scale*self.cond_scale,self.seq_len))
            np.random.shuffle(order1)
            order = np.array(order + order1)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders
    
    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        r = torch.rand(1)
        # mask_rate = torch.cos(r * math.pi * 0.5)[0]
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask
    
    def random_masking_step(self, device, seq_len, bsz, orders, mode):
        # generate token mask
        # bsz, seq_len, embed_dim = x.shape
        mask = torch.zeros(bsz, seq_len, device=device)
        # mask_rate = self.mask_ratio_generator.rvs(1)[0]
            

        if mode:
            r = torch.rand(1)
            # mask_rate = torch.cos(r * math.pi * 0.5)[0]
            mask_rate = self.mask_ratio_generator.rvs(1)[0]
            # 前16个token按照mask_rate做mask，后面的全mask
            num_masked_tokens = int(np.ceil(self.cond_scale*self.cond_scale * mask_rate))
            mask = torch.zeros(bsz, seq_len, device=device)
            mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                                src=torch.ones(bsz, seq_len, device=device))
            mask[:, self.cond_scale*self.cond_scale:] = 1
        else:
            r = torch.rand(1)
            # mask_rate = torch.cos(r * math.pi * 0.5)[0]
            mask_rate = self.mask_ratio_generator2.rvs(1)[0]
            # 前16个token不mask，后面的按照mask_rate做mask
            num_masked_tokens = int(np.ceil((seq_len-self.cond_scale*self.cond_scale) * mask_rate))
            mask = torch.zeros(bsz, seq_len, device=device)
            mask = torch.scatter(mask, dim=-1, index=orders[:, self.cond_scale*self.cond_scale:self.cond_scale*self.cond_scale+num_masked_tokens],
                                src=torch.ones(bsz, seq_len, device=device))   

        return mask
    
    def forward_mae_encoder_stage1(self, x, mask, class_embedding, task, cond):

        if self.training:
            # mode -> True: train small scale, so -> small_cond_mask is False
            small_x = self.cond_proj(cond)
            x = torch.cat([small_x,self.z_proj(x)],dim=1)
        else:
            x = torch.cat([self.cond_proj(cond),self.z_proj(x)],dim=1)
            
        bsz, seq_len, embed_dim = x.shape
        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        # encoder position embedding
        x = x + self.encoder_pos_embed_learned[:,:self.small_seqlen+self.buffer_size,:]
        
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        # apply Transformer blocks
        task_c = self.encoder_task_emb(torch.tensor([[task]]).cuda())
        ada = self.encoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.encoder_blocks:
                x = block(x, ada, task)
        x = self.encoder_norm(x)
        return x

    def forward_mae_encoder_stage2(self, x, mask, class_embedding, task, cond=None,small_z=None,small_mask=None):

        if self.training:
            # mode -> True: train small scale, so -> small_cond_mask is False
            small_cond_mask = (torch.rand(x.shape[0]) <= 1.0-self.step_stage2_rate*min(1,self.epoch/self.step_warmup))
            small_cond_mask = small_cond_mask.unsqueeze(-1).repeat(1,self.cond_scale*self.cond_scale).cuda().to(x.dtype).unsqueeze(-1)
            
            if small_z is None:
                small_x = small_cond_mask * self.cond_proj(self.cond_tokens).repeat(x.shape[0],1,1) + (1 - small_cond_mask) * self.cond_proj(cond)
            else:
                small_x1 = small_mask.unsqueeze(-1) * self.cond_proj(self.cond_tokens).repeat(x.shape[0],1,1) + (1 - small_mask.unsqueeze(-1)) * self.cond_z_proj(small_z)
                small_x = small_cond_mask * self.cond_proj(self.cond_tokens).repeat(x.shape[0],1,1) + (1 - small_cond_mask) * small_x1
                
            x = torch.cat([small_x,self.z_proj(x)],dim=1)
        else:
            if small_z is None:
                x = torch.cat([self.cond_proj(cond),self.z_proj(x)],dim=1)
            else:
                x = torch.cat([self.cond_z_proj(small_z),self.z_proj(x)],dim=1)
                
        
        bsz, seq_len, embed_dim = x.shape
        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        # encoder position embedding
        x = x + self.encoder_pos_embed_learned[:,self.small_seqlen+self.buffer_size:,:]
        
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        # apply Transformer blocks
        task_c = self.encoder_task_emb(torch.tensor([[task]]).cuda())
        ada = self.encoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.encoder_blocks:
                x = block(x, ada, task)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder_stage1(self, x, mask, task):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = torch.cat([torch.zeros(mask_with_buffer.shape[0], self.buffer_size, self.mask_token.shape[2],dtype=x.dtype,device=x.device),self.mask_token[:,:self.small_seqlen,:].repeat(mask_with_buffer.shape[0], 1, 1).to(x.dtype)],dim=1)

        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned[:,:self.small_seqlen+self.buffer_size,:]
        # apply Transformer blocks
        task_c = self.decoder_task_emb(torch.tensor([[task]]).cuda())
        ada = self.decoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.decoder_blocks:
                
                x = block(x, ada, task)
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_mae_decoder_stage2(self, x, mask, task):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = torch.cat([torch.zeros(mask_with_buffer.shape[0], self.buffer_size, self.mask_token.shape[2],dtype=x.dtype,device=x.device),self.mask_token[:,self.small_seqlen:,:].repeat(mask_with_buffer.shape[0], 1, 1).to(x.dtype)],dim=1)

        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned[:,self.small_seqlen+self.buffer_size:,:]

        # apply Transformer blocks
        task_c = self.decoder_task_emb(torch.tensor([[task]]).cuda())
        ada = self.decoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.decoder_blocks:
                
                x = block(x, ada, task)
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x
    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss
    
    def global_forward_loss(self, z, target, mask, c):
        bsz, _, h, w = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1, 1, 1)
        z = z.reshape(bsz, h, w, -1).permute(0,3,1,2).repeat(self.diffusion_batch_mul, 1, 1, 1)
        mask = mask.reshape(bsz, h, w).repeat(self.diffusion_batch_mul,1,1)
        c = c.repeat(self.diffusion_batch_mul)
        loss = self.cond_diffloss(z=z, target=target, mask=mask,c=c)
        return loss

    def forward(self, imgs, labels, cond=None):
        # small scale part
        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        # patchify and mask small tokens
        cond = self.patchify(cond)
        
        gt_latents = x.clone().detach()
        cond_gt_latents = cond.clone().detach()
        # order: small scale (rand), big scale (rand)
        orders = self.sample_orders_step(bsz=x.size(0))
        #only mask the small scale
        mask = self.random_masking_step(x.device,x.shape[1]+cond.shape[1],x.shape[0], orders, mode=True)   
            
        # mae encoder   
        x_small_c = self.forward_mae_encoder_stage1(x, mask, class_embedding, task=0, cond=cond)

        # mae decoder
        z_small_c = self.forward_mae_decoder_stage1(x_small_c, mask, task=0)

        # diffloss
        #get the small scale 
        z_small_c = z_small_c[:,:self.cond_scale*self.cond_scale,:]
        gt_latents_small_c = cond_gt_latents
        mask = mask[:,:self.cond_scale*self.cond_scale]
        loss1 = self.forward_loss(z=z_small_c, target=gt_latents_small_c, mask=mask)
        small_mask = 1.0 - mask.clone()
        small_z = z_small_c.clone().detach()
        
        # big scale part
        # only mask big scale part
        big_mask = self.random_masking_step(x.device,x.shape[1]+cond.shape[1],x.shape[0], orders, mode=False)

        x_big_c = self.forward_mae_encoder_stage2(x, big_mask, class_embedding, task=1, cond=cond,small_z=small_z,small_mask=small_mask)
        z_big_c = self.forward_mae_decoder_stage2(x_big_c, big_mask, task=1)
        z_big_c = z_big_c[:,self.cond_scale*self.cond_scale:,:]
        gt_latents_big_c = gt_latents
        big_mask = big_mask[:,self.cond_scale*self.cond_scale:]
        loss2 = self.global_forward_loss(z=z_big_c, target=imgs, mask=big_mask, c=labels)
        if torch.any(torch.isnan(loss1)) or torch.any(torch.isnan(loss2)):
            print("nan")
        return 2.0*loss1, 0.75*loss2


    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, re_cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, vq_model=None,cond=None, stage=1,resmall_tokens=None):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len-self.cond_scale*self.cond_scale, self.token_embed_dim).cuda()
        cond_tokens = torch.zeros(bsz, self.cond_scale*self.cond_scale, self.cond_dim).cuda()
        orders = self.sample_orders_step(bsz)
        indices = list(range(num_iter))
        # only small scale tokens infer
        small_z = torch.zeros(bsz*2, self.cond_scale*self.cond_scale, self.decoder_embed_dim).cuda()
        
        small_num_iter = self.cond_scale*4
        small_indices = list(range(small_num_iter))
        for step in small_indices:
            cur_cond_tokens = cond_tokens.clone()
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            
            if not re_cfg == 1.0:
                cond_tokens = torch.cat([cond_tokens, cond_tokens], dim=0)
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            # mae encoder
            x = self.forward_mae_encoder_stage1(tokens, mask, class_embedding, task=0, cond=cond_tokens)
            # mae decoder                
            z = self.forward_mae_decoder_stage1(x, mask, task=0)
            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / small_num_iter)
            mask_len = torch.Tensor([np.floor(self.cond_scale*self.cond_scale * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                    torch.minimum(torch.sum(mask[:,:self.cond_scale*self.cond_scale], dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order_step(mask_len[0], orders, bsz, self.seq_len, self.cond_scale*self.cond_scale)
            if step >= small_num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
                mask_to_pred[:,self.cond_scale*self.cond_scale:] = False
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            if not re_cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            small_z[mask_to_pred[:,:self.cond_scale*self.cond_scale].nonzero(as_tuple=True)] = z[mask_to_pred[:,:self.cond_scale*self.cond_scale].nonzero(as_tuple=True)] 
            mask = mask_next
            

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (re_cfg - 1) * (self.cond_scale*self.cond_scale - mask_len[0]) / (self.cond_scale*self.cond_scale)
            elif cfg_schedule == "constant":
                cfg_iter = re_cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not re_cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            cur_cond_tokens[mask_to_pred[:,:self.cond_scale*self.cond_scale].nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
            cond_tokens = cur_cond_tokens.clone()


        small_tokens = self.unpatchify_small(cond_tokens)
        mask[:,:self.cond_scale*self.cond_scale] = False  
        
        # cond small, big scale tokens infer
        if not cfg == 1.0:
            small_z = torch.cat([small_z[:bsz],small_z[:bsz]], dim=0)
        else:
            small_z = small_z[:bsz]
        for step in indices:            
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                cond_tokens = torch.cat([cond_tokens, cond_tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            # mae encoder
            x = self.forward_mae_encoder_stage2(tokens, mask, class_embedding, task=1, cond=cond_tokens,small_z=small_z)

            # mae decoder
            z = self.forward_mae_decoder_stage2(x, mask, task=1)
            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor((self.seq_len-self.cond_scale*self.cond_scale) * mask_ratio)]).cuda()
            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                    torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders[:,self.cond_scale*self.cond_scale:], bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            # mask_to_pred = torch.ones_like(mask_to_pred)
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
                
            orig_x = None
            position=None
            
            this_mask = mask[:,self.cond_scale*self.cond_scale:].clone().bool().reshape(mask.shape[0],self.seq_h,self.seq_w)
            this_token = tokens.clone().reshape(tokens.shape[0],self.seq_h,self.seq_w,-1).permute(0,3,1,2)
            last_mask = mask.clone().bool()
            mask = mask_next
            
            # sample token latents for this step
            z = z[:,self.cond_scale*self.cond_scale:]
            z = z.reshape(z.shape[0],self.seq_h,self.seq_w,-1).permute(0,3,1,2)
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len-self.cond_scale*self.cond_scale - mask_len[0]) / (self.seq_len-self.cond_scale*self.cond_scale)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            # cfg_iter = 2.5
            if not cfg == 1.0:
                cfg_labels = torch.cat([labels,1000*torch.ones_like(labels)],dim=0)
            else:
                cfg_labels = labels
            sampled_token_latent = self.cond_diffloss.sample(z, cfg_labels, temperature, cfg_iter,this_mask,this_token)
            sampled_token_latent = sampled_token_latent.reshape(sampled_token_latent.shape[0],sampled_token_latent.shape[1],-1).permute(0,2,1)[last_mask[:,self.cond_scale*self.cond_scale:].nonzero(as_tuple=True)]
            
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                last_mask, _ = last_mask.chunk(2, dim=0)
                cond_tokens, _ = cond_tokens.chunk(2, dim=0)
                
            # cur_tokens[mask_to_pred[:,self.cond_scale*self.cond_scale:].nonzero(as_tuple=True)] = sampled_token_latent
            cur_tokens[last_mask[:,self.cond_scale*self.cond_scale:].nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
        
        big_cond_tokens = self.unpatchify(tokens)
        return small_tokens, big_cond_tokens
    
        
def himar_base(**kwargs):
    model = HiMAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def himar_large(**kwargs):
    model = HiMAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def himar_huge(**kwargs):
    model = HiMAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
