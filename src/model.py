import inspect
import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class Config:
    n_embd: int=1
    n_head: int=1
    n_layer: int=1
    block_size: int=1
    dropout: float=0
    vocab_size: int=50000
    bias: bool=True

class LayerNorm(nn.Module):
    def __init__(self, ndim,bias):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(ndim))
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.size(),self.weight, self.bias,1e-5)#eps ensures denominator non-zero

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(name='bias', tensor=torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1, 1, config.block_size, config.block_size))#view should be included by the bracket

    def forward(self,x):
        B, T, C = x.size() #batch_size, num of token, n_embd
        q,k,v = self.c_attn(x).split(self.config.n_embd,dim=2)
        q = q.view(B,T,self.config.n_head, C//self.config.n_head).transpose(1, 2)
        k = k.view(B,T,self.config.n_head, C//self.config.n_head).transpose(1, 2)
        v = v.view(B,T,self.config.n_head, C//self.config.n_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1/math.sqrt(C//self.config.n_head))#the denominator should be C//n_head instead of C
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class block(nn.Module):
    def __init__(self,config):
        super().__init__()
        # self.ln = LayerNorm(config.n_embd, bias=config.bias)
        # self.c_attn = MultiHeadAttention(config)
        # self.mlp = MLP(config)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))#don't forget here is residual
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.transformer=nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([block(config) for _ in range(config.n_layer)]),
            layernorm = LayerNorm(config.n_embd,config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0,std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.para_count() / 1e6,))

    def init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def para_count(self,non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self,idx,targets = None, srf=None):
        b,t = idx.size()

        assert t<=self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx) #attention! here is idx instead of pos
        pos_emd = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb+pos_emd) #broadcasting
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.layernorm(x)

        if targets is not None:
            logists = self.lm_head(x)
            if srf is not None:
                logists = logists[:, :-1, :].contiguous()
                targets = targets[:, 1:].contiguous()
                loss = F.cross_entropy(logists.view(-1, logists.size(-1)),targets.view(-1), ignore_index=-100)
            else:
                loss = F.cross_entropy(logists.view(-1, logists.size(-1)),targets.view(-1), ignore_index=-100)
            #loss is scalar
        else:
            loss = None
            logists = self.lm_head(x[:,[-1],:])

        return logists,loss

    @torch.no_grad()
    def generate(self,idx,temp = 1.0, top_k=None, max_new_tokens=100, eos_token_id = None):
        for _ in range(max_new_tokens):
            idx_ = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_)
            logits = logits[:, -1, :] / temp
            if not top_k is None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = float('-inf')
            p = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(p, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and (idx_next==eos_token_id).all():
                break
        return idx

    def config_optimizer(self, decay_weight, lr, betas, device_type):
        para = {np:p for np, p in self.named_parameters()}
        para = {np:p for np, p in para.items() if p.requires_grad}
        decay_para = [p for np,p in para.items() if p.dim()>=2]
        nodecay_para = [p for np,p in para.items() if p.dim()<2]
        op_group = [
            {'params': decay_para, 'weight_decay': decay_weight},
            {'params': nodecay_para, 'weight_decay': 0.0},
        ]
        fuse = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fuse and device_type == 'cuda'
        a = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(op_group, lr=lr, betas=betas, **a)
        print(f"using fused AdamW: {use_fused}")

        return optimizer