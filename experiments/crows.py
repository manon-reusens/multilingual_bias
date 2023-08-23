import argparse
import os
import json

import transformers
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)


from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

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
    default="BertForMaskedLM",
    choices=[
        "BertForMaskedLM",
        "AlbertForMaskedLM",
        "RobertaForMaskedLM",
        "GPT2LMHeadModel",
    ],
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased","bert-base-multilingual-uncased","bert-base-multilingual-cased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default=None,
    choices=["gender", "race", "religion","socioeconomic","sexual-orientation", "age", "nationality","disability","physical-appearance" ],
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
    help="Random seed for the experiments.",
)
parser.add_argument(
    "--lang",
    action="store",
    type=str,
    default='en',
    choices=['en', 'fr', 'nl', 'de','pl','ru','ca' ],
    help="Language to evaluate on.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="crows",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        sample=args.sample,
        seed=args.seed,
        lang_eval=args.lang,
    )

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - sample: {args.sample}")
    print(f" - seed: {args.seed}")
    print(f" -language: {args.lang}")

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.lang == 'en':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_en_s{str(args.seed)}.csv"
    elif args.lang == 'fr':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_fr.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_fr_s{str(args.seed)}.csv"
    elif args.lang == 'de':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_de.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_de_s{str(args.seed)}.csv"
    elif args.lang == 'nl':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_nl.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_nl_s{str(args.seed)}.csv"
    elif args.lang == 'pl':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_pl.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_pl_s{str(args.seed)}.csv"
    elif args.lang == 'ru':
        if args.seed==None:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_ru.csv"
        else:
            input=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized_ru_s{str(args.seed)}.csv"
    elif args.lang == 'ca':
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
        sample=args.sample,
        seed=args.seed, 
   )
    results,df_data_with_masks = runner()

    print(f"Metric: {results}")

    os.makedirs(f"{args.persistent_dir}/results/crows", exist_ok=True)
    df_data_with_masks.to_csv(f"{args.persistent_dir}/results/crows/{experiment_id}.csv")
    with open(f"{args.persistent_dir}/results/crows/{experiment_id}.json", "w") as f:
        json.dump(results, f)
