#imports

import numpy as np
import torch
import torch.nn as nn


def patchify(images, n_patches):
    n, c, h, w = images.shape #num images, channels, height, width
    device = images.device

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2, device=device)
    #(num_images, patches, pixels per patch)
    #(N, 25, 16)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d): #(26, 8), d = embedding_dim, 26 since we do it after class token
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length): # 0 to 25 sequence
        for j in range(d): # 0 to 7 sequence
            if j % 2 == 0: #j is even
                result[i][j] = np.sin(i / (10000 ** (j / d)))
            else: #j is odd
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))      
    return result

class MHSA(nn.Module):
    def __init__(self, embedding_dim, n_heads=2):
        super(MHSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        head_dim = int(embedding_dim / n_heads)

        self.q = nn.ModuleList([nn.Linear(head_dim,head_dim) for _ in range(self.n_heads)])
        self.k = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(self.n_heads)])
        self.v= nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(self.n_heads)])
        #each one is a list of n_heads nn.Linear(4, 4)

        self.proj = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(n_heads)])
        # nn.Linear(4, 4) for each head

        self.final_linear = nn.Linear(embedding_dim, embedding_dim)

        self.head_dim = head_dim
        self.softmax = nn.Softmax(dim=-1) #softmax along last dim

    def forward(self, sequences):

        # (N, 26, 8) -> (N, 26, 2, 4) -> (N, 26, 8)

        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads): # for each of the (2) heads
                qm = self.q[head] #get q,k,v matricies for each head
                km = self.k[head]
                vm = self.v[head]

                projm = self.proj[head]

                seq = sequence[:, head * self.head_dim: (head + 1) * self.head_dim]
                # get the subsequence for each head of attention

                # pass the sequence through each layer
                q, k, v = qm(seq), km(seq), vm(seq)

                #the important attention formula
                attention = self.softmax(q @ k.T / (self.head_dim ** 0.5)) @ v

                projected = projm(attention)  # (26, 4)

                seq_result.append(projected)
            
            res1 = torch.hstack(seq_result) #stack across all heads
            # (26, 4) + ... + (26, 4) = (26, 8)
            result.append(res1)
            #list corresponding to result of batches

        # stacks each item in the batch -> (N, 26, 8)
        output = torch.stack(result, dim=0)

        return self.final_linear(output) #final linear (N, 26, 8) -> (N, 26, 8)

class TE_Block(nn.Module):
    def __init__(self, embedding_dim, n_heads, ratio=4):
        super(TE_Block, self).__init__()
        self.hidden_dim = embedding_dim
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mhsa = MHSA(embedding_dim, n_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ratio * embedding_dim),
            nn.GELU(),
            nn.Linear(ratio * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class ViT(nn.Module):
  def __init__(self, chw=(1,20,20), n_patches=5, hidden_dim=8, n_heads=2, n_blocks=2, out_dim=3):
    # Super constructor
    super(ViT, self).__init__()

    # Attributes
    self.chw = chw # (Channels, height, width)
    self.n_patches = n_patches
    self.hidden_dim = hidden_dim
    self.n_heads = n_heads
    self.n_blocks = n_blocks
    self.out_dim = out_dim
    self.patch_size = (chw[1] // n_patches, chw[2] // n_patches) #(4, 4)

    # Linear Embedding -> nn.Linear (16 -> 16)
    self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1]) #(1 * 4 * 4)
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim) #(16, 8)

    # Classification token
    self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

    # Positional Embedding
    self.pos_embed = nn.Parameter(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_dim).clone().detach()) #(26, 8)
    self.pos_embed.requires_grad = False # this parameter doesn't need to be trained

    # Transformer encoder blocks
    self.blocks = nn.ModuleList([TE_Block(hidden_dim, n_heads) for _ in range(n_blocks)])
    # create n_blocks transformer-encoder blocks

    # Classification MLP
    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_dim, out_dim), # (8 -> 3)
        nn.Softmax(dim=-1)
    )


  def forward(self, images):
    n, c, h, w = images.shape #(N, 1, 20, 20)

    patches = patchify(images, self.n_patches) #images to patches (N, 1, 20, 20) -> (N, 25, 16)
    tokens = self.linear_mapper(patches) #patches to embedded tokens (N, 25, 16) -> (N, 25, 8)

    new_tokens = []
    for i in range(len(tokens)):
        combined = torch.vstack((self.class_token.to(tokens.device), tokens[i]))
        #add class token to start (1, 25, 1) -> (1, 26, 1)
        new_tokens.append(combined)

    # Stack back into a single tensor of shape (N, 25 + 1, 8)
    tokens = torch.stack(new_tokens) #now at (N, 26, 8)

    pos_embed = self.pos_embed.to(tokens.device).repeat(n, 1, 1) #repeat along batch dim, once against seq and embed
    #we are encoding position in the image, only need to get one sequence per image
    out = tokens + pos_embed #(N, 25, 8)

    #pass through each transformer-encoder block
    for block in self.blocks:
        out = block(out)

    #get first token only (classification token)
    out = out[:, 0]

    #classify the token
    out = self.mlp(out)

    return out

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
