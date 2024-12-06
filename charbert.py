
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class CharTokenizer:
    def __init__(self):
        chars = "".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        self.vocab = {char: idx for idx, char in enumerate(chars, start=2)}  # Start indexing from 2
        self.vocab["_"] = 1  # Add MASK token
        self.vocab["[PAD]"] = 0  # Add PAD token
        
        self.inv_vocab = {idx: char for char, idx in self.vocab.items()}  # Reverse lookup

    def tokenize(self, word, mask_prob=0.1):
        out = []
        original = []
        for char in word:
            if random.random() < mask_prob:
                out.append(self.vocab["_"])
                original.append(self.vocab.get(char, 0))
            else:
                
                out.append(self.vocab.get(char, 0))
                original.append(self.vocab.get(char, 0))
        return out, original

    def batch_tokenize(self, words, mask_prob):
        out = []
        original = []
        for word in words:
            dummy = self.tokenize(word, mask_prob)
            out.append(dummy[0])
            original.append(dummy[1])

        # add padding to make the sequences the same length, assuming padding index is 0 
        max_len = max(len(tokens) for tokens in out)
        out = [tokens + [0] * (max_len - len(tokens)) for tokens in out]
        original = [tokens + [0] * (max_len - len(tokens)) for tokens in original]
        return out, original  

    def decode(self, tokens): # remove the padding before returning
        return ''.join([self.inv_vocab[token] for token in tokens if token != 0])    
    def batch_decode(self, tokens_list):
        return [self.decode(tokens) for tokens in tokens_list]





# STEP 2.
# 1. Create a model that will take tokenized masked words and predict the original words.
# 2. The model should have an embedding layer,Convolutional layers, norm and relu, and a linear layer.
# 3. The model should have a method that will take a batch of tokenized masked words and return the predicted words.

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader



class CNNBERT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        """
        CNN-based BERT-like model.
        """
        super(CNNBERT, self).__init__()
        
        num_classes = vocab_size  # Number of output classes is the same as the vocabulary size
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1 ml moudle using Convolutional layers, norm and relu
        out = 128
        self.cnn1 = nn.Conv1d(in_channels=embedding_dim, out_channels=out, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.LayerNorm(out, elementwise_affine=False)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        # 2 ml moudle using Convolutional layers, norm and relu
        self.cnn2 = nn.Conv1d(in_channels=out, out_channels=out, kernel_size=5, padding=2, bias=False)
        self.norm2 = nn.LayerNorm(out, elementwise_affine=False)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        # 3 ml moudle using Convolutional layers, norm and relu
        self.cnn3 = nn.Conv1d(in_channels=out, out_channels=out, kernel_size=5, padding=2, bias=False)
        self.norm3 = nn.LayerNorm(out, elementwise_affine=False)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        

        # 4 ml moudle using Convolutional layers, norm and relu
        self.cnn4 = nn.Conv1d(in_channels=out, out_channels=out, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm4 = nn.LayerNorm(out, elementwise_affine=False)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.1)
        
        outt = 256
        # 5 ml moudle using Convolutional layers, norm and relu
        self.cnn5 = nn.Conv1d(in_channels=out, out_channels=outt, kernel_size=1, padding=0, bias=False)
        self.norm5 = nn.LayerNorm(outt, elementwise_affine=False)
        self.relu5 = nn.ReLU()
        
        
        # Linear layer
        self.linear = nn.Linear(outt, num_classes)
        
        
    def forward(self, x):
        # x is the batch of input tokens
        # Embedding layer
        x = self.embedding(x).permute(0, 2, 1) # Permute to (batch, channels, sequence)
        
        # Convolutional layers
        x = self.cnn1(x).permute(0, 2, 1) # Permute to (batch, sequence, channels)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.cnn2(x.permute(0, 2, 1)).permute(0, 2, 1) # Permute to (batch, sequence, channels)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.cnn3(x.permute(0, 2, 1)).permute(0, 2, 1) # Permute to (batch, sequence, channels)
        x = self.norm3(x)
        x = self.relu3(x)
        
        x = self.cnn4(x.permute(0, 2, 1)).permute(0, 2, 1) # Permute to (batch, sequence, channels)
        x = self.norm4(x)
        x = self.relu4(x)
        
        x = self.cnn5(x.permute(0, 2, 1)).permute(0, 2, 1) # Permute to (batch, sequence, channels)
        x = self.norm5(x)
        x = self.relu5(x)
        
        # Linear layer
        x = self.linear(x)
        return x

    @torch.no_grad()
    def predict(self, x, topk):
        # Get the logits
        logits = self.forward(x)
        
        # Get the index of the highest logit
        # return logits.argmax(dim=-1)
        return logits.topk(topk, dim=-1).indices
    
    @torch.no_grad()
    def accuracy(self, x, y, mask=None, topk=1):
        # Get the predictions
        predictions = self.predict(x, topk)
    
        if mask is None:
            # If no mask is provided, calculate accuracy for all tokens
            return (predictions == y).float().mean()

        # Calculate the accuracy only for masked tokens
        # masked_accuracy = ((predictions == y) * mask).float().sum()  # Apply mask
    
        dummy = []
        for j in range(predictions.shape[-1]):
            dummy.append((predictions[:,:,j] == y))

        dummy = torch.stack(dummy, dim=-1)
        masked_accuracy =  ( dummy.any(dim=-1) * mask).float().sum()  # Apply mask
        
        return masked_accuracy



def train(model, tokenizer, data, full_data, epochs=10, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    
    model.train()
    
    mask_p = []
    best_accuracy = 0
    for epoch in range(epochs):
        total_loss = 0
        
        optimizer.zero_grad()
        for i, batch_data in enumerate(data):
            mask_prob = random.random()
            if mask_prob < 0.1:
                mask_prob = 0.1
            if mask_prob > 0.9:
                mask_prob = 0.9
            mask_p.append(mask_prob)
            tokenized_words, original_words = tokenizer.batch_tokenize(batch_data, mask_prob=mask_prob)
            tokenized_words = torch.tensor(tokenized_words).to(device)
            original_words = torch.tensor(original_words).to(device)
            
            masked_indices = (tokenized_words == 1).float()  # Create a mask for non-zero (non-padding) indices
            
            logits = model.forward(tokenized_words)
       
            # Get the cross-entropy loss
            loss = F.cross_entropy(logits.permute(0, 2, 1), original_words, ignore_index=0, reduction='none')  # Calculate loss for all tokens
            # Apply the mask to calculate the loss only for masked tokens
            loss = loss * masked_indices  # This will set loss to 0 for non-masked tokens (padding tokens)
            # Now, we calculate the average loss over only the masked tokens
            loss = loss.sum() / masked_indices.sum()  # Normalize by the number of masked tokens
         
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients
            
            total_loss += loss

            
        total_loss /= len(data)
        scheduler.step()
        print(f"Train -- Epoch {epoch+1} - Loss: {total_loss.item():.2f} -- avg_mask_prob: {sum(mask_p)/len(mask_p):.2f}")
    
        if epoch % 10 == 0:
            with torch.no_grad():
                # full testing
                model.eval()
                avg_accuracy = 0
                for mask_prob in [0.2]:
                    print(f"Mask Prob: {mask_prob}")
                    for topk in range(1, 6):
                        total_accuracy = 0
                        total_loss = 0
                        for i, batch_data in enumerate(full_data):
                            
                            tokenized_words, original_words = tokenizer.batch_tokenize(batch_data, mask_prob=mask_prob)
                            tokenized_words = torch.tensor(tokenized_words).to(device)
                            original_words = torch.tensor(original_words).to(device)
                            
                            masked_indices = (tokenized_words == 1).float()  # Create a mask for non-zero (non-padding) indices
                            
                            logits = model.forward(tokenized_words)
                            loss = F.cross_entropy(logits.permute(0, 2, 1), original_words, ignore_index=0, reduction='mean')
                                        
                            accuracy = model.accuracy(tokenized_words, original_words, masked_indices, topk) 
                            accuracy /= masked_indices.sum()

                            total_accuracy += accuracy
                            total_loss += loss.item()
                        

                        total_accuracy /= len(full_data)        
                        total_loss /= len(full_data)    
                        
                        avg_accuracy += total_accuracy
                        
                        print(f"Topk ---- {topk} ---- Accuracy: {accuracy:.4f}, Loss: {total_loss:.4f}")
            avg_accuracy /= 5    
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                torch.save(model.state_dict(), "model.pth")
                print(f"Model saved, average accuracy: {avg_accuracy:.4f}")



def beam_search(prob_matrix, beam_width=3):
    """
    prob_matrix: 2D numpy array where each row corresponds to the probability distribution
                 at a particular time step (each column is a possible symbol).
    beam_width: The number of top sequences to maintain at each step.
    
    Returns the top `beam_width` sequences with their corresponding probabilities.
    """
    # Number of time steps and number of possible symbols
    T, V = prob_matrix.shape
    
    # Initialize the beams: each beam is a tuple (sequence, cumulative probability)
    beams = [([], 0.0)]  # Starting with an empty sequence and 0 probability
    
    for t in range(T):  # Iterate over time steps
        all_candidates = []
        
        # For each beam at the previous time step, extend it by every possible symbol
        for seq, prob in beams:
            for v in range(V):  # For each possible symbol at this time step
                new_seq = seq + [v]
                new_prob = prob + prob_matrix[t, v]  # Add log of probability to avoid underflow
                all_candidates.append((new_seq, new_prob))
        
        # Sort all candidates by probability (in descending order)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top `beam_width` candidates
        all_candidates = [(seq, prob) for seq, prob in all_candidates if prob > float('-inf')]
        beams = all_candidates[:beam_width]
    
    # Return the top `beam_width` sequences
    return beams


@torch.no_grad()
def Test(model, tokenizer, device, topk, word, guessed_letters=['l','s']):
    model.eval()
    
    tokenized_words, original_words = tokenizer.batch_tokenize([word], mask_prob=0.0)
    assert tokenized_words == original_words
    tokenized_words = torch.tensor(tokenized_words).to(device)
    
    
    logits = model.forward(tokenized_words)[0,:,:] 
    logits[:,:2] = float('-inf')

    try: 
        del tokenizer.inv_vocab[0]
        del tokenizer.inv_vocab[1]
    except:    pass
 
    guessed_tokens = [tokenizer.vocab[cc] for cc in guessed_letters]
    
    
    if "_" not in word:
        prob_matrix = F.softmax(logits, dim=1).cpu().numpy()    # (5,26)
        prob_matrix = np.log(prob_matrix)
        
        log_prob = 0
        for ind, char in enumerate(word):
            char_ind = tokenizer.vocab[char]
            if char_ind in guessed_tokens:
                return float('-inf')
            log_prob += prob_matrix[ind, char_ind]
        return log_prob
    
    
    
    for ind, char in enumerate(word):
        if char != '_':
            char_ind = tokenizer.vocab[char]
            logits[ind] = float('-inf')
            logits[ind, char_ind] = 1.0
        
        else: 
            for cc in word.replace("_",""):
                char_ind = tokenizer.vocab[cc]
                logits[ind,char_ind] = float('-inf')
            
            # get the topk indices of the logits at time step ind
            # topk_indices = list( torch.topk(logits[ind], k=5).indices.cpu().numpy() )

            topk_indices = list( torch.topk(logits[ind], k=logits[ind].shape[0]).indices.cpu().numpy() )
            # topk indices that are not in the guessed letters
            topk_indices = [ind for ind in topk_indices if ind not in guessed_tokens][:5]
            
            for index in range(logits[ind].shape[0]):
                if index not in topk_indices:
                    logits[ind,index] = float('-inf')
    
    
    
    prob_matrix = F.softmax(logits, dim=1).cpu().numpy()    # (5,26)
    prob_matrix = np.log(prob_matrix)
    prob_matrix[prob_matrix == -float('inf')] = 1
    prob_matrix[prob_matrix == 1] = 2*prob_matrix.flatten().min()


    beam_width = min(topk, int(5**word.count('_'))) # Keep top k beams at each step
    top_beams = beam_search(prob_matrix, beam_width)

    # Output the top beams with their cumulative probabilities
    possible_words = []
    for i, (seq, prob) in enumerate(top_beams):
        # print(f"Beam {i+1}: Sequence = {seq}, Cumulative Probability = {prob}")
        try: 
            word = "".join([tokenizer.inv_vocab[token] for token in seq])
            possible_words.append((word, prob))
        except: pass
        
    
    return  possible_words

def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location,"r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary


# Function to calculate the total number of parameters
def get_param_count(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters in multiples of 100k: {param_count/1e5:.2f}")
 
if __name__ == "__main__":
    # Create the tokenizer
    tokenizer = CharTokenizer()

    # Create the model
    vocab_size=len(tokenizer.vocab)
    model = CNNBERT(vocab_size=vocab_size)

    # use cuda if available
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Get the total parameter count of the model
    get_param_count(model)
    
    trainn = True

    if trainn:
        # Load the data
        full_dictionary_location = "words_250000_train.txt"
        full_dictionary = build_dictionary(full_dictionary_location)
        random.shuffle(full_dictionary)
        print(f"Total words in the dictionary: {len(full_dictionary)}")
        
        train_size = int(len(full_dictionary) * 0.9)
        print(f"Train data size: {train_size}")
        print(f"Test data size: {len(full_dictionary) - train_size}")
        
        train_data = DataLoader(full_dictionary[:train_size], batch_size=1024, shuffle=True)
        test_data = DataLoader(full_dictionary[train_size:], batch_size=8192, shuffle=False)
        
        
        # Train the model
        train(model, tokenizer, train_data, test_data, epochs=1000, lr=0.0001)
    
    else: 
        model.load_state_dict(torch.load("model.pth"), strict=True)
    
    # Test the model
    topk = 1
    possible_words = Test(model, tokenizer, device, topk=5, word = "app_e_")
    print(f"Possible words: {possible_words}")