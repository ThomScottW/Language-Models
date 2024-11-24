"""Bigram Language Model Code from https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2349s"""


from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # context length
max_iterations = 10000
evaluation_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluation_iterations = 200
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
with open('shakespeare.txt', 'r') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# These two just map the characters to their position in the vocabulary
_ENCODER_MAPPING = {
    character: position for position, character in enumerate(chars)
}

_DECODER_MAPPING = {
    position: character for position, character in enumerate(chars)
}


def encode(input_text: str) -> List[int]:
  """Encode a string of text to an integer vector."""
  return [_ENCODER_MAPPING[character] for character in input_text]


def decode(input_vector: List[int]) -> str:
  """Decode an integer vector into a string of text."""
  return ''.join([_DECODER_MAPPING[value] for value in input_vector])


data = torch.tensor(encode(text), dtype=torch.long)
_TRAINING_SPLIT_FRACTION = 0.9
_TRAINING_SPLIT_INDEX = int(_TRAINING_SPLIT_FRACTION * len(data))
training_data = data[:_TRAINING_SPLIT_INDEX]
validation_data = data[_TRAINING_SPLIT_INDEX:]


def get_batch(split: str) -> Tuple[torch.tensor, torch.tensor]:
  """Get a batch of data containing training and target sequences.

  Choose batch_size pairs of randomly chosen input and target blocks.
  
  Args:
    split: A string specifying whether the returned batch should be from
      the training set or the validation set. "train" is checked for, while
      anything else will be considered validation.
  Returns:
    inputs, targets: Two batch_size x block_size tensors, where the i'th row
      in each tensor corresponds to the input and prediction targets.
  """
  data = training_data if split == 'train' else validation_data

  # Choose batch_size number of random offsets between 0 and the length
  # of the data - the block size.
  randomly_chosen_indices = torch.randint(len(data) - block_size, (batch_size,))
  inputs = torch.stack(
      [data[i:i + block_size] for i in randomly_chosen_indices]
  )
  targets = torch.stack(
      [data[i + 1:i + block_size + 1] for i in randomly_chosen_indices]
  )
  return inputs.to(device), targets.to(device)


# This torch.no_grad() decorator stops Pytorch from storing all the intermediate
# variables in memory, because it signals that we'll never be calling .backward
# on anything. This leads to greater memory efficiency.
@torch.no_grad()
def estimate_loss() -> Dict[str, int]:
  """Get the average loss for both splits over a large number of iterations."""
  out = {}
  bigram_language_model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(evaluation_iterations)
    for k in range(evaluation_iterations):
      inputs, targets = get_batch(split)
      logits, loss = bigram_language_model(inputs, targets)
      losses[k] = loss.item()
    out[split] = losses.mean()
  bigram_language_model.train()
  return out


class BigramLanguageModel(nn.Module):
  def __init__(self, vocabulary_size):
    super().__init__()

    # The embedding table is basically just a tensor of vocab_size x vocab_size
    self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
  
  def forward(self, index, targets=None):
    # index and targets are both (B, T) tensors of integers
    # B = batch_size
    # T = time = block_size
    # C = channels = vocab_size
    # logits are the scores for the next token
    logits = self.token_embedding_table(index)  # (B,T,C)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
    # Pytorch's cross_entropy function expects a tensor of B, C, so here
    # the logits and targets tensors are reshaped to accomodate this
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, index, max_new_tokens):
    """Generate predictions on the characters given by index.
    
    Args:
      index: Current context of some characters in a batch. Shape BxT. This
        function's job is to extend index to be Bx(T+1), Bx(T+2), etc for
        max_new_tokens.
      max_new_tokens: How many additional tokens to predict from the given
        context.
    """
    # index is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      
      # Get the predictions for the current context
      logits, loss = self(index)

      # Focus only on the last time step
      logits = logits[:, -1, :]  # becomes (B, C)

      # Apply softmax to get the probabilities
      # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
      probs = F.softmax(logits, dim=-1)  #(B, C)

      # Sample from the distribution
      index_next = torch.multinomial(probs, num_samples=1) # (B, 1)

      # Append the sampled index to the running sequence
      index = torch.cat((index, index_next), dim=1) # (B, T+1)
      
    return index


bigram_language_model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(bigram_language_model.parameters(), lr=1e-3)

for iter in range(max_iterations):

  # Evaluate loss on both training and validation sets
  if iter % evaluation_interval == 0:
    losses = estimate_loss()
    print(f'Step {iter}: Training Loss: {losses['train']:.4f}, Validation Loss: {losses['val']:.4f}')

  input_data, target_data = get_batch('train')

  logits, loss = bigram_language_model(input_data, target_data)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


# Feeding in a 1x1 tensor with a zero in it is equivalent to passing a newline
# as the first token, or whatever the first character in the vocabulary is.
index = torch.zeros((1, 1), dtype=torch.long, device=device)

# This will be a Bx(T+1) output, and since the prior line of code passed in
# 1 batch, B is 1.
generations = bigram_language_model.generate(index, max_new_tokens=500)
generations = generations[0].tolist()

print(decode(generations))