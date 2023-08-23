import argparse
import os
import numpy as np
import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)

import torch
import transformers
from tqdm import tqdm 

from bias_bench.dataset import load_sentence_debias_data
from bias_bench.debias import DensRay

from bias_bench.model import models
from bias_bench.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(
    description="Computes the bias subspace for Densray."
)
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertModel",
    choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
    help="Model (e.g., BertModel) to compute the SentenceDebias subspace for. "
    "Typically, these correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased",'bert-base-multilingual-uncased', 'bert-base-multilingual-cased', "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["gender", "religion", "race"],
    required=True,
    help="The type of bias to compute the bias subspace for.",
)
parser.add_argument(
    "--lang_debias",
    action="store",
    type=str,
    default='en',
    choices=['en','nl','de','fr','pl','ru','ca'],
    help="Batch size to use while encoding.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=32,
    help="Batch size to use while encoding.",
)



if __name__ == "__main__":
    args = parser.parse_args()
    print('Densray')
    experiment_id = generate_experiment_id(
        name="densray",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        lang_debias=args.lang_debias,
    )

    print("Computing bias direction densray:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - model: {args.model}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - language debias : {args.lang_debias}")

    # Get the data to compute the SentenceDebias bias subspace.
    data = load_sentence_debias_data(
        persistent_dir=args.persistent_dir, bias_type=args.bias_type, lang_debias=args.lang_debias
    ) #female_example

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    #for name, param in model.named_parameters():
    #    print(name)

    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    #tokenize the examples (data['female_example] en data['male_example'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_embeddings_male = []
    all_embeddings_female = []

    n_batches = len(data) // args.batch_size
    print(n_batches)
    for i in tqdm(range(n_batches), desc="Encoding gender examples"):
        offset = args.batch_size * i

        inputs_male = tokenizer(
            [example["male_example"] for example in data[offset : offset + args.batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        inputs_female = tokenizer(
            [
                example["female_example"]
                for example in data[offset : offset + args.batch_size]
            ],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        male_input_ids = inputs_male["input_ids"].to(device)
        female_input_ids = inputs_female["input_ids"].to(device)

        male_attention_mask = inputs_male["attention_mask"].to(device)
        female_attention_mask = inputs_female["attention_mask"].to(device)
        with torch.no_grad():
            # Compute average representation from last layer.
            # embedding_male.shape == (batch_size, 128, 768).
            embedding_male = model(
                input_ids=male_input_ids, attention_mask=male_attention_mask
            )["last_hidden_state"]
            embedding_male *= male_attention_mask.unsqueeze(-1)
            embedding_male = embedding_male.sum(dim=1)
            embedding_male /= male_attention_mask.sum(dim=1, keepdims=True)

            embedding_female = model(
                input_ids=female_input_ids, attention_mask=female_attention_mask
            )["last_hidden_state"]
            embedding_female *= female_attention_mask.unsqueeze(-1)
            embedding_female = embedding_female.sum(dim=1)
            embedding_female /= female_attention_mask.sum(dim=1, keepdims=True)

        embedding_male /= torch.norm(embedding_male, dim=-1, keepdim=True)
        embedding_female /= torch.norm(embedding_female, dim=-1, keepdim=True)
       
        all_embeddings_male.append(embedding_male.cpu().numpy())
        all_embeddings_female.append(embedding_female.cpu().numpy())

    # all_embeddings_male.shape == (num_examples, dim).
    all_embeddings_male = np.concatenate(all_embeddings_male, axis=0)
    all_embeddings_female = np.concatenate(all_embeddings_female, axis=0)
    #L, R = analogy.get_embeddings_from_cropus(corpus, layer)
    # compute densray
    print('now we will compute the bias dimensions')
    densray = DensRay(torch.from_numpy(all_embeddings_male),torch.from_numpy(all_embeddings_female))
    densray.fit()

    print(
        f"Saving computed eigenvector to: {args.persistent_dir}/results/densray/{experiment_id}.pt."
    )
    os.makedirs(f"{args.persistent_dir}/results/densray", exist_ok=True)
    torch.save(
        densray.eigvecs, f"{args.persistent_dir}/results/densray{experiment_id}.pt"
    )
    torch.save(densray.mean, f"{args.persistent_dir}/results/densray{experiment_id}_mean.pt")
    torch.save(densray.std, f"{args.persistent_dir}/results/densray{experiment_id}_std.pt")
