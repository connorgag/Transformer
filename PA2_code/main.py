import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Transformer_Encoder, Classifier, Transformer_Decoder, Head
from utilities import Utilities
import nltk
import math
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 32  # Maximum context length for predictions

def plot_data(title, x_label, y_label, x, y):
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(title + ".png")

    plt.show()
    
def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

    

def train_encoder(model, dataloader, epochs_CLS, learning_rate, test_dataloader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    classifier_accuracy_list = []
    for epoch in range(epochs_CLS):
        total_loss = 0.0
        total_iterations = 0
        for train_batch_samples, targets in dataloader:
            train_batch_samples, targets = train_batch_samples.to(device), targets.to(device)

            optimizer.zero_grad()
            train_batch_results = model(train_batch_samples)
            loss = loss_function(train_batch_results, targets)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            total_iterations = total_iterations + 1
        print("Epoch " + str(epoch) + " Loss: " + str(total_loss / total_iterations))
        
        class_acc =  compute_classifier_accuracy(model, test_dataloader)
        classifier_accuracy_list.append(class_acc)
        print("Classifier Accuracy: " + str(class_acc))

    plot_data("Classifier Accuracy in Training", "Epoch", "Classifier Accuracy", [i for i in range(len(classifier_accuracy_list))], classifier_accuracy_list)



def train_decoder(model, dataloader, max_iters, eval_iters, eval_interval, learning_rate):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    perp_list = []
    loss_list = []
    total_loss = 0.0
    total_iterations = 0
    for train_batch_samples, targets in dataloader:
        if (total_iterations > max_iters):
            break

        train_batch_samples, targets = train_batch_samples.to(device), targets.to(device)

        train_batch_results = model(train_batch_samples)[0] # Plug into model

        # Reshape output and targets
        B, T, C = train_batch_results.shape

        # Compute loss
        loss = loss_function(train_batch_results.view(B*T, C), targets.view(B*T))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        total_loss = total_loss + loss.item()

        total_iterations = total_iterations + 1

        if (total_iterations % eval_interval == 0):    
            perplexity = compute_perplexity(model, dataloader, eval_iters) 
            perp_list.append(perplexity)   
            print(perplexity)

    plot_data("Perplexity During Training: Basic Positional Embeddings", "Iterations", "Perplexity", [i for i in range(eval_interval, (len(perp_list)+1) * eval_interval, eval_interval)], perp_list)


def president_perplexity(decoder, tokenizer, president, block_size, batch_size):
        inputfile = "speechesdataset/test_LM_" + president + ".txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            speech_text = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, speech_text,  block_size)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        print("Perplexity of " + president + " is " + str(compute_perplexity(decoder, test_LM_loader, 500)))


def main():
    # Hyperparameters

    """ Hyperparameters to use for training to roughly match 
    the numbers mentioned in the assignment description """

    batch_size = 16  # Number of independent sequences  we will process in parallel
    block_size = 32  # Maximum context length for predictions
    learning_rate = 1e-3  # Learning rate for the optimizer
    n_embd = 64  # Embedding dimension
    n_head = 2  # Number of attention heads
    n_layer = 4  # Number of transformer layers
    head_size = 2

    eval_interval = 100  # How often to evaluate train and test perplexity during training
    max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
    eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

    n_hidden = 100  # Hidden size for the classifier
    n_output = 3  # Output size for the classifier, we have 3 classes
    epochs_CLS = 15 # epochs for classifier training

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run a Specific Part of the Project')
    parser.add_argument('--part', type=int, required=True, help='Part to run (e.g., 1, 2, 3)')
    args = parser.parse_args()


    # --------------------- Tokenize Input -------------------------
    print("Loading data and creating tokenizer ...")
    # Loads all training texts into a list ['the cat ate the food', 'our nation under liberty', ...]
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data



    # ---------------------  Part 1: Encoder and Classifier -------------------------
    if (args.part == 1):
        # ---------------------  Prepare Encoder Datasets -------------------------
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        # --------------------- Transformer Encoder -------------------------
        # This isn't used by itself, but it could be if needed
        encoder = Transformer_Encoder(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, block_size, embedding_strategy='basic_positional', n_hidden=n_hidden)
        print("Encoder has " + str(sum(p.numel() for p in encoder.parameters())) + " parameters")

        # --------------------- Classification Training -------------------------
        classifier = Classifier(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, n_output, n_hidden, block_size)
        print("Classifier has " + str(sum(p.numel() for p in classifier.parameters())) + " parameters")

        train_encoder(classifier, train_CLS_loader, epochs_CLS, learning_rate, test_CLS_loader)

        # --------------------- Encoder Sanity Check  -------------------------
        # utilities = Utilities(tokenizer, Classifier(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, n_output, n_hidden, block_size).get_transformer()) # pretraining
        utilities = Utilities(tokenizer, classifier.get_transformer()) # posttraining
        sentence = "And this is important , because no development strategy can be based only upon what comes out of the ground , nor can it be sustained while young people"
        utilities.sanity_check(sentence, block_size)
  


    # --------------------- Part 2: Language Modeling Task -------------------------
    elif (args.part == 2):
        # --------------------- Prepare Decoder Datasets -------------------------
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        # --------------------- Decoder  -------------------------
        decoder = Transformer_Decoder(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, block_size, n_hidden)
        print("Decoder has " + str(sum(p.numel() for p in decoder.parameters())) + " parameters")


        # --------------------- Decoder Training -------------------------
        train_decoder(decoder, train_LM_loader, max_iters, eval_iters, eval_interval, learning_rate)


        # --------------------- Decoder Sanity Check  -------------------------
        # utilities = Utilities(tokenizer, Transformer_Decoder(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, block_size, n_hidden)) # pretraining
        utilities = Utilities(tokenizer, decoder) # posttraining
        sentence = "And this is important , because no development strategy can be based only upon what comes out of the ground , nor can it be sustained while young people"
        utilities.sanity_check(sentence, block_size)

        # --------------------- Perplexity Calculations  -------------------------
        president_perplexity(decoder, tokenizer, 'obama', block_size, batch_size)
        president_perplexity(decoder, tokenizer, 'hbush', block_size, batch_size)
        president_perplexity(decoder, tokenizer, 'wbush', block_size, batch_size)

    elif (args.part == 3):
        batch_size = 16  # Number of independent sequences  we will process in parallel
        block_size = 32  # Maximum context length for predictions
        learning_rate = 1e-3  # Learning rate for the optimizer
        n_embd = 64  # Embedding dimension
        n_head = 4  # Number of attention heads
        n_layer = 4  # Number of transformer layers
        head_size = 2
        embedding_strategy='basic_positional'

        # --------------------- Prepare Decoder Datasets -------------------------
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        # --------------------- Decoder  -------------------------
        decoder = Transformer_Decoder(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, block_size, n_hidden, embedding_strategy=embedding_strategy)
        print("Decoder has " + str(sum(p.numel() for p in decoder.parameters())) + " parameters")


        # --------------------- Decoder Training -------------------------
        train_decoder(decoder, train_LM_loader, max_iters, eval_iters, eval_interval, learning_rate)


        # --------------------- Decoder Sanity Check  -------------------------
        # utilities = Utilities(tokenizer, Transformer_Decoder(tokenizer.get_vocab_length(), n_embd, head_size, n_head, n_layer, block_size, n_hidden, part3=True)) # pretraining
        utilities = Utilities(tokenizer, decoder) # posttraining
        sentence = "And this is important , because no development strategy can be based only upon what comes out of the ground , nor can it be sustained while young people"
        utilities.sanity_check(sentence, block_size)

        # --------------------- Perplexity Calculations  -------------------------
        president_perplexity(decoder, tokenizer, 'obama', block_size, batch_size)
        president_perplexity(decoder, tokenizer, 'hbush', block_size, batch_size)
        president_perplexity(decoder, tokenizer, 'wbush', block_size, batch_size)
    



if __name__ == "__main__":
    main()
