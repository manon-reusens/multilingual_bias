import argparse
import os
import json
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)


import torch
import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
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
    default="SentenceDebiasBertForMaskedLM",
    choices=[
        "DensrayDebiasBertForMaskedLM",
        "SentenceDebiasBertForMaskedLM",
        "SentenceDebiasAlbertForMaskedLM",
        "SentenceDebiasRobertaForMaskedLM",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPBertForMaskedLM",
        "INLPmBertForMaskedLM",
        "INLPAlbertForMaskedLM",
        "INLPRobertaForMaskedLM",
        "INLPGPT2LMHeadModel",
        "CDABertForMaskedLM",
        "CDAAlbertForMaskedLM",
        "CDARobertaForMaskedLM",
        "CDAGPT2LMHeadModel",
        "DropoutBertForMaskedLM",
        "DropoutAlbertForMaskedLM",
        "DropoutRobertaForMaskedLM",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasmBERTForMaskedLM",
    ],
    help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM). Typically, these "
    "correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased","bert-base-multilingual-cased","bert-base-multilingual-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_direction",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    help="Path to saved ContextDebias, CDA, or Dropout model checkpoint.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default="gender",
    choices=["gender", "race", "religion", "socioeconomic", "sexual-orientation", "age", "nationality", "disability", "physical-appearance"],
    help="Determines which CrowS-Pairs dataset split to evaluate against.",
)
parser.add_argument(
    "--sample",
    action="store",
    type=str,
    default="false",
    choices=["true","false" ],
    help="Determines whether a sample of the dataset should be taken or not.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="Random seed for the experiments",
)
parser.add_argument(
    "--lang_eval",
    action="store",
    type=str,
    default='en',
    choices=['en', 'fr', 'nl', 'de','pl','ru','ca'],
    help="Language to evaluate on.",
)
parser.add_argument(
    "--lang_debias",
    action="store",
    type=str,
    default='en',
    choices=['en', 'fr', 'nl', 'de','pl','ru','ca'],
    help="Language used to debias",
)

if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="crows",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        sample= args.sample,
        seed=args.seed,
        lang_eval=args.lang_eval,
        lang_debias=args.lang_debias,
    )

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_direction: {args.bias_direction}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - load_path: {args.load_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - sample: {args.sample}")
    print(f" - seed: {args.seed}")
    print(f" - lang_eval: {args.lang_eval}")
    print(f" - lang_debias: {args.lang_debias}")
    s=''
    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        if args.model=='DensrayDebiasBertForMaskedLM':
            bias_direction=torch.load(args.bias_direction)
            bias_direction=bias_direction
            print(bias_direction.size())
        else:   
            bias_direction = torch.load(args.bias_direction)
        
        kwargs["bias_direction"] = bias_direction

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix
        k=args.projection_matrix
        s=k.replace('.pt','')
        s=s.replace('/','')

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )

    if _is_self_debias(args.model):
        model._model.eval()
    else:
        model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.lang_eval == 'en':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_en_s{str(args.seed)}.csv"
    elif args.lang_eval == 'fr':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_fr.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_fr_s{str(args.seed)}.csv"
    elif args.lang_eval == 'de':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_de.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_de_s{str(args.seed)}.csv"
    elif args.lang_eval == 'nl':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_nl.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_nl_s{str(args.seed)}.csv"
    elif args.lang_eval == 'pl':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_pl.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_pl_s{str(args.seed)}.csv"
    elif args.lang_eval == 'ru':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_ru.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_ru_s{str(args.seed)}.csv"
    elif args.lang_eval == 'ca':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_ca.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_ca_s{str(args.seed)}.csv"

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file=input,
        bias_type=args.bias_type,
        is_generative=_is_generative(args.model),  # Affects model scoring.
        is_self_debias=_is_self_debias(args.model),
        sample=args.sample,
        seed=args.seed,
    )
    results,df_data_with_mask_probs = runner()
    print(f"Metric: {results}")

    os.makedirs(f"{args.persistent_dir}/results/crows", exist_ok=True)
    df_data_with_mask_probs.to_csv(f"{args.persistent_dir}/results/crows/{experiment_id}{s}.csv")
    with open(f"{args.persistent_dir}/results/crows/{experiment_id}{s}.json", "w") as f:
        json.dump(results, f)
