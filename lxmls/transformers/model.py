"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import re
import torch
import torch.nn as nn
from torch.nn import functional as F

from lxmls.transformers.utils import CfgNode as CN
from lxmls.transformers.bpe import BPETokenizer

# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()

        # Initialize layers and parameters
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head

        # Create the linear projections for query, key, and value tensors
        # Note: the input and output size of all these projections is n_embd
        self.query_proj = nn.Linear(config.n_embd, config.n_embd)
        self.key_proj = nn.Linear(config.n_embd, config.n_embd)
        self.value_proj = nn.Linear(config.n_embd, config.n_embd)

        self.output_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Create the projections for query, key, and value tensors
        # Note: In self-attention these are all over the same tensor x
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape and transpose tensors for multi-head computation.
        # We reshape the output from (B, T, C) to (B, T, num_heads, hidden_size/num_heads)
        # And transpose the result to (B, num_heads, T, hidden_size/num_heads)
        # So the multi-head computation can be implemented as a single matrix multiplication.
        query = query.view(B, T, self.num_heads,
                           self.hidden_size // self.num_heads).transpose(1, 2)
        key = key.view(B, T, self.num_heads,
                       self.hidden_size // self.num_heads).transpose(1, 2)
        value = value.view(B, T, self.num_heads,
                           self.hidden_size // self.num_heads).transpose(1, 2)

        # Compute attention scores. The shape of scores should be (B, num_heads, T, T)
        # Hint: You can use tensor.transpose() to adapt the order of the axes.
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Normalize the scores by dividing by the square root of the hidden size
        # Take into account that you are using multi-head attention!
        scores = scores / math.sqrt(self.hidden_size // self.num_heads)

        # Apply causal mask to restrict attention to the left in the input sequence
        mask = self.bias[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax activation to get attention weights
        # Check the correct axis for the softmax function! What should be the shape of the weights?
        weights = F.softmax(scores, dim=-1)

        # Apply dropout to the attention weights
        weights = self.attn_dropout(weights)

        # Multiply attention weights with values to get attended values
        attended_values = torch.matmul(weights, value)

        # Transpose and reshape attended values to restore original shape
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            B, T, C)

        # Apply output projection and dropout
        output = self.resid_dropout(self.output_proj(attended_values))

        return output


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))
                                        )  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([
            config.n_layer is not None, config.n_head is not None,
            config.n_embd is not None
        ])
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':
                dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':
                dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':
                dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                'gpt2-large':
                dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                'gpt2-xl':
                dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                # Gophers
                'gopher-44m':
                dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':
                dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':
                dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':
                dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config)
                                 for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,
                                      mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6, ))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf
                if not k.endswith('attn.masked_bias')]  # ignore these
        keys = [
            k for k in keys
            if not re.match("transformer\.h\.\d+\.attn\.bias", k)
        ]  # ignore these
        sd_keys = [
            k for k in sd if not re.match("transformer\.h\.\d+\.attn\.bias", k)
        ]  # ignore these

        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them

        # This assert might fail for some transformers library versions. Please comment out if that is the case
        assert len(keys) == len(sd_keys)

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(
                        m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(
                        m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params
        ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=train_config.learning_rate,
                                      betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long,
                           device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx,
                 max_new_tokens,
                 temperature=1.0,
                 do_sample=False,
                 top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def gen_batch(self, idx, max_new_tokens, temperature=1.0, batch=10):
        """
        A dummy function for "fixed" test generation
        We take a conditioning sequence of indices idx (LongTensor of shape (b,t)),
        take the top "batch" predictions for first token and then complete 10 generations as normal
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        out = []
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(
            1) <= self.block_size else idx[:, -self.block_size:]

        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # Get the top "batch" predictions for the word
        _, idx_list = torch.topk(probs, k=batch, dim=-1)

        for idx_next in idx_list[0, :]:
            idx_tmp = torch.cat((idx, idx_next.reshape(-1, 1)), dim=1)

            idx_tmp = self.generate(idx_tmp, max_new_tokens - 1)

            out.append(idx_tmp)

        return (out)

    def prompt(self, p_text="", tokens=20, num_samples=1, do_sample=True):
        """
        Human-usable promting function, for the most part just run with prompt and tokens
        """

        if not hasattr(self, 'tok'):
            self.tok = BPETokenizer()

        if p_text == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[self.tok.encoder.encoder['<|endoftext|>']]],
                             dtype=torch.long)
        else:
            device = next(self.parameters()).device
            x = self.tok(p_text).to(device)

        # we'll process all desired num_samples in a batch, so expand out the batch dim
        x = x.expand(num_samples, -1)

        # forward the model `steps` times to get samples, in a batch
        y = self.generate(x,
                          max_new_tokens=tokens,
                          do_sample=do_sample,
                          top_k=100)

        for i in range(num_samples):
            out = self.tok.decode(y[i].cpu().squeeze())
            print('-' * 80)
            print(out)

    def prompt_topK(self, p_text="", tokens=20, num_samples=5):
        """
        Human-usable prompting function. Deterministic, cah use for evaluation

        """

        if not hasattr(self, 'tok'):
            self.tok = BPETokenizer()

        if p_text == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[self.tok.encoder.encoder['<|endoftext|>']]],
                             dtype=torch.long)
        else:
            device = next(self.parameters()).device
            x = self.tok(p_text).to(device)

        y = self.gen_batch(x, max_new_tokens=tokens, batch=num_samples)

        for y_tmp in y:
            out = self.tok.decode(y_tmp.cpu().squeeze())
            print('-' * 80)
            print(out)
