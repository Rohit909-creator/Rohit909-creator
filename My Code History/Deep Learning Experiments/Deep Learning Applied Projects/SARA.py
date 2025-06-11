import torch
import torch.nn as nn
import numpy as np

#Plan A 
#Self Attention equipped FeedForwardNetwork

class NN(nn.Module):

  def __init__(self,Self_Attention,inp_size,embed_size,hidd_size,out_size,action_dim):
    super().__init__()

    self.attention = Self_Attention(inp_size,embed_size)
    self.fc1 = nn.Linear(120,hidd_size)
    self.fc2 = nn.Linear(hidd_size,hidd_size)
    self.fc3 = nn.Linear(hidd_size,out_size)
    self.action = nn.Linear(hidd_size,action_dim)
    self.relu = nn.ReLU()

  def forward(self,X):
    bs,seq,embeddings = X.shape
    #print(f'bs:{bs}')
    out = self.attention(X)
    out = out.reshape(bs,-1)
    out = self.relu(self.fc1(out))
    out = self.relu(self.fc2(out))
    output = self.fc3(out)
    action = self.action(out)

    return output,action

class Self_Attention(nn.Module):


  def __init__(self,dim_size,embed_size):
    super().__init__()
    self.dim_size = dim_size
    self.queries = nn.Linear(dim_size,embed_size)
    self.keys = nn.Linear(dim_size,embed_size)
    self.values = nn.Linear(dim_size,embed_size)
    self.softmax = nn.Softmax(dim = 0)
  

  def forward(self,Query,Keys):

    Q = self.queries(Queries)
    K = self.keys(Keys)
    V = self.values(X)
    K = K.t()
    attn_scores = torch.matmul(Q,K)
    softmaxed_attn_scores = self.softmax(attn_scores)
    out = torch.matmul(softmaxed_attn_scores,V)

    return out

    



class Multi_headed_attention(nn.Module):

    def __init__(self,embeddings,head_dim,num_heads = 2):
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.queries = nn.Linear(embeddings,num_heads*head_dim)
        self.keys = nn.Linear(embeddings,num_heads*head_dim)
        self.values = nn.Linear(embeddings,num_heads*head_dim)
        self.softmax = nn.Softmax(dim = -1)
        self.unify = nn.Linear(num_heads*head_dim,head_dim)

    def forward(self,X):
        bs,seq,embed_dim = X.shape
        Q = self.queries(X).view(bs,seq,self.num_heads,self.head_dim).transpose(1,2)
        K = self.keys(X).view(bs,seq,self.num_heads,self.head_dim).transpose(1,2)
        V = self.values(X).view(bs,seq,self.num_heads,self.head_dim).transpose(1,2)
        K = K.transpose(-1,-2)
        attn_scores = torch.matmul(Q,K)/(self.head_dim**(1/float(2)))
        softmaxed_attn_scores = self.softmax(attn_scores)
        out = torch.matmul(softmaxed_attn_scores,V)
        out = out.transpose(1,2).contiguous().view(bs,seq,self.num_heads*self.head_dim)
        out = self.unify(out)
        return out




def padding(token,max_tokens):
    if len(token) < max_tokens:
        for i in range(max_tokens - len(token)):
            token[i] = np.zeros((1,len(all_words)))

    else:
        pass

def tokenizer(all_words,string):
    all_words_bool = all_words == string
    return all_words_bool






sentences = ['hello',
             'open chapter 1 from voice notes',
             'take me to chapter 1 in voice notes',
             'open chapter 1 from assignments',
             'take me to chapter 1 in pdf']

all_words = []
for s in sentences:
    w_s = s.split(' ')
    all_words.extend(w_s)


max_tokens = 8

print(len(all_words))
        
#cleaning all_words

for w in all_words:
    if all_words.count(w) > 1:
        all_words.pop(all_words.index(w))
print(len(all_words))
        
    



all_words = np.array(all_words)



data = []
for i,s in enumerate(sentences):
    words = s.split(' ')
    d = [[0]*len(all_words)]*max_tokens
    for j,w in enumerate(words):
        token = tokenizer(all_words,w)
        d[j] = token
    data.append(d)


data = torch.tensor([data],dtype = torch.float32)
data = data.reshape(5,8,15)
print(data.shape)


'''
['hello', 'open chapter 1 from voice notes', 'take me to chapter 1 in voice notes', 'open chapter 1 from assignments', 'take me to chapter 1 in pdf']

sentences = ['hello',
             'open chapter 1 from voice notes',
             'take me to chapter 1 in voice notes',
             'open chapter 1 from assignments',
             'take me to chapter 1 in pdf',
             'open pdf notes , general notes',
             'open recent',
             'go back',
             'how many left to complete in pdf',
             'how many left New in pdf',
             'how many have I completed in pdf',
             'sort everything in date wise order']





'''

intents = {'intents':[{'tag':'greetings','patterns':['hello','hi','hola','namaskar'],'responses':['hello','hi','how can i help']},
                      {'tag':'actions','patterns':['get me to pdf','take me to pdf','open pdf',
                      'go to assignments','take me to assignments','open assignments','take me to voice notes'
                      ,'open voice notes','take me to voice notes','open voice notes','go back','open recent'],'responses':['ok','here you go','how can i help']},
                      {'tag':'thanks','patterns':['thank you','thanks'],'responses':['you are welcome','happy to help']},
                      {'tag':'read','patterns':['how many left to complete in pdf','how many left New in pdf','how many have I completed in pdf'],'responses':['read']},
                      {'tag':'bye','patterns':['bye bye','ta ta','see ya','shutdown'],'responses':['bye bye']},
                      {'tag':'wishing','patterns':['how are you','you good','how are you feeling'],'responses':['I am good']},
                      
                      
                      ]}

'''
actions = ['none','open','back','bookmark','download','recent']

action_data = [0,1,1,1,1]
action_data = torch.tensor(action_data)


tags = []
for intent in intents['intents']:
  tags.append(intent['tag'])

target_data = []
for i in range(len(tags)):
  target_data.append(i)

target_data = torch.tensor([0,1,1,1,1])
print(target_data)



loss = nn.CrossEntropyLoss()
model = NN(Multi_headed_attention,len(all_words),len(all_words),256,5,len(actions))
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

model.action.requires_grad_(False)

#training model:
num_epochs = 1000


for epoch in range(num_epochs):
  out,action = model(data)
  #print(out.shape)
  optimizer.zero_grad()
  l = loss(out,target_data)
  if not epoch%100:
    print(f'At Epoch:{epoch} Loss:{loss}')
  if l.item() < 0.07:
    print("got there")
    print(f'At Epoch:{epoch} Loss:{loss}')
    break
  l.backward()
  optimizer.step()


print(model(data))

#freezing the synaptic weights in all other layers except action_layer 
model.action.requires_grad_(True)
model.fc1.requires_grad_(False)
model.fc2.requires_grad_(False)
model.fc3.requires_grad_(False)
model.attention.requires_grad_(False)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

#training Action prediction layer:

for epoch in range(1000):
  out,action = model(data)
  optimizer.zero_grad()
  l = loss(action,action_data)
  if l.item() < 0.085:
    print(f"At Epoch:{epoch} Loss:{l}")
    break
    
  l.backward()
  optimizer.step()

out,actions = model(data)
print(actions)




def vectorize(sentence):
  d = [[[0]*len(all_words)]*8]
  words = sentence.split(' ')
  for i,w in enumerate(words):
    token = tokenizer(all_words,w)
    d[0][i] = token
  return d

test_data = vectorize('hello take me to assignments')
test_data = torch.tensor(test_data,dtype = torch.float32)

print(test_data.shape)


out,action = model(test_data)
out = torch.argmax(out).item()
action = torch.argmax(action)

print(intents['intents'][out]['tag'])
print(action)

'''



