def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
    sample=None,
    lang_eval=None,
    lang_debias=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(lang_eval, str):
        experiment_id += f"_eval-{lang_eval}"
    if isinstance(lang_debias, str):
        experiment_id += f"_debias-{lang_debias}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(sample,str):
        experiment_id+=f"_sample-{sample}"

    return experiment_id
