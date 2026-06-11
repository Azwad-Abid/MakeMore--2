#-----Introduction to Make More 2------#
#Make More 2 is different from part 1 mainly because
#we add another hidden layer (tanh) and it is morre of a word level language model instead of
#a character level one
#What we're doing:
#- Part 1 looked at 1 previous character
#- Part 2 looks at 3 previous characters (context window)
#- Part 1 had 0 hidden layers
#- Part 2 has 1+ hidden layers
#Goal: Show that depth + context = much better learning
#Expected NLL: ~1.8-2.0 (way better than part 1's 2.558)

#Libraries
#Modules

import torch
import torch.nn as nn
import torch.nn.functional as f
import urllib.request
import os

if not os.path.exists('names.txt'):
    print("Downloading names.txt...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt',
        'names.txt'
    )
    print("Downloaded!\n")

#STEP -1 : lOAD DATA
print("Loading Data....")
words = open('names.txt','r').read().splitlines()
print(f"Loaded {len(words)} names \n")
#character lookup
chars = ['.'] + sorted(list(set(''.join(words))))
#enumerate(Index Mapping) -- A function that assigns an index to a letter (basically)
stoi = {s:i for i,s in enumerate(chars)}
itos = { i : s for i,s in enumerate(chars)}
print(f"Vocab size: {len(chars)} characters\n")


#STEP - 2 : PREPARE TRAINING DATA WITH CONTEXT WINDOW
print("Preparing training data with context window")

context_length = 3
xs ,ys = [], [] #xs is inputs ; # ys is outputs

for w in words :
  context = [0]* context_length

  for ch in w + '.':
    ix = stoi[ch]
    xs.append(context)
    ys.append(ix)
    context= context[1:]+ [ix]

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"Total training examples : {len(xs)}")
print(f"xs shape: {xs.shape}")
print(f"ys shape: {ys.shape}\n")

# STEP - 3 --- BUILDING NEURAL NETWORK --- #
class MLP(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_dim , context_length):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size , embedding_dim)
    self.hidden = nn.Linear(context_length * embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, vocab_size)
  def forward(self,x):
    x= self.embedding(x)
    x = x.view(x.shape[0],-1)
    x= torch.tanh(self.hidden(x))
    logits = self.output(x)
    return logits

vocab_size = len(chars)
embedding_dim = 10
hidden_dim = 200
context_length = 3

model = MLP(vocab_size, embedding_dim,hidden_dim , context_length)

print(f"Model Created:")
print(f"Total parameters : {sum(p.numel() for p in model.parameters())}\n")

#---STEP -4 : TRAINING---#
print("Training The Neural Network....\n")
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 10
batch_size = 32
#Shuffling the data so that the model learn to catch random patterns
for epoch in range(num_epochs):
  indices = torch.randperm(len(xs))
  xs_shuffled = xs[indices]
  ys_shuffled = ys[indices]

  total_loss = 0
  num_batches = 0

  for i in range(0, len(xs),batch_size):
    xs_batch = xs_shuffled[i : i + batch_size]
    ys_batch = ys_shuffled[i : i + batch_size]

    logits = model(xs_batch)
    loss = loss_fn(logits, ys_batch)

    optimizer.zero_grad()  # Set all gradients to 0
    loss.backward()        # Compute NEW gradients
    optimizer.step()       # Use NEW gradients

    total_loss += loss.item()
    num_batches += 1

  avg_loss = total_loss / num_batches
  print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f}")

# --- STEP -5 :Final Evaluation --- #
with torch.no_grad():
    logits = model(xs)
    loss = loss_fn(logits, ys)
    print(f"Final training NLL: {loss.item():.4f}")

#-- Part 6 Generating names --- #
print("Generating names from the trained model:\n")

g = torch.Generator().manual_seed(42)

for _ in range(10):
    out = []
    context = [0] * context_length
    
    while True:
        x_tensor = torch.tensor([context])
        logits = model(x_tensor)
        probs = f.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        out.append(itos[ix])
        context = context[1:] + [ix]
        
        if ix == 0:
            break
    
    print(f"  {''.join(out)}")

print("\n✓ Done!")

#I DID THIS IN GOOGLE COLAB SO YEAH : D





