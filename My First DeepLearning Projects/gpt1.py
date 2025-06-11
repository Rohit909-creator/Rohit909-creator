import torch
import torch.nn as nn
from torch.nn import functional as F




batch_size = 4
block_size = 8
n_emb = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)



with open("C:\\Users\\ROHIT FRANCIS\\OneDrive\\Documents\\input.txt", 'r') as f:
  data = f.read()

# print(len(data))

text = data[:1000]
chars = sorted(set(text))
# print(chars)
vocab_size = len(chars)
# print("Vocab_size:",vocab_size)
# print("".join(chars))

itos = {i:w for i,w in enumerate(chars)}
# print("index t string",itos)
stoi = {w:i for i,w in enumerate(chars)}
# print("string to index",stoi)

encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: "".join([itos[i] for i in l])
# print(encoder("hii there"))
# print(decoder(encoder("hii there")))


data = torch.tensor(encoder(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[0:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]


class Head(nn.Module):

    def __init__(self, head_dim):
        super().__init__()

        self.queries = nn.Linear(n_emb, head_dim, bias = False)
        self.keys = nn.Linear(n_emb, head_dim, bias=False)
        self.values = nn.Linear(n_emb, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # self.ln = nn.LayerNorm(head_dim)
        # self.ln2 = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
    def forward(self, X):
        
        B,T,C = X.shape 
        q = self.queries(X)
        k = self.keys(X)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        v = self.values(X)

        out = wei@v

        return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_dim=head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim =-1)

        


class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_sized):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
    self.position_embedding_table = nn.Embedding(block_size, n_emb)
    # self.sa_head = Head(n_emb, n_emb)
    self.sa_head = MultiHeadedAttention(4, n_emb//4)
    self.lm_head = nn.Linear(n_emb, vocab_size)
    
  def forward(self, idx, targets = None):
    B,T = idx.shape
    # print(B,T)
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.sa_head(x)
    logits = self.lm_head(x)
    if targets == None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss



  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):


      idx_cond = idx[:,-block_size:]

      logits, loss = self(idx_cond)

      logits = logits[:,-1,:]

      probs = F.softmax(logits, dim=-1)

      idx_next = torch.multinomial(probs, num_samples=1)

      idx = torch.cat((idx, idx_next), dim = 1)
      # print(idx.shape)
    return idx




batch_size = 4
block_size = 8

def get_batch(split):
  data  = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y


xb, yb = get_batch("train")
# print("inputs")
# print(xb.shape)
# print(xb)
# print("targets")
# print(yb.shape)
# print(yb)


m = BigramLanguageModel(vocab_size)
# out, loss = m(xb, yb)
# print(out.shape, loss)

m = m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), 1e-3)


batch_size = 32

for steps in range(5000):

  xb, yb = get_batch('train')
  
  xb = xb.to(device)
  yb = yb.to(device)



  logits, loss = m(xb, yb)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  torch.cuda.empty_cache()

print(loss.item())


idxx = torch.zeros((1,1), dtype = torch.long).to(device=device)

print(decoder(m.generate(idxx, 1000)[0].tolist()))