---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- loss:TripletLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: Tow Motor Operator - 2nd Shift, $18/hour
  sentences:
  - Back End Engineer
  - Tow Motor Operator
  - County Judge
- source_sentence: Judge - County Court (Civil Division)
  sentences:
  - County Judge
  - Credit Administration Manager
  - Credit Administration Manager
- source_sentence: Steamfitter Assistant - Join Our HVAC Team!
  sentences:
  - Steamfitter
  - Loom Technician
  - Loom Technician
- source_sentence: Circuit Judge - County Court
  sentences:
  - Loom Technician
  - Credit Administration Manager
  - County Judge
- source_sentence: Steamfitter Assistant - Join Our HVAC Team!
  sentences:
  - Tow Motor Operator
  - Mortgage Processor
  - Steamfitter
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on sentence-transformers/all-mpnet-base-v2
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: validation
      type: validation
    metrics:
    - type: cosine_accuracy
      value: 1.0
      name: Cosine Accuracy
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on the generator dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision 12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - generator
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Steamfitter Assistant - Join Our HVAC Team!',
    'Steamfitter',
    'Tow Motor Operator',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Dataset: `validation`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| **cosine_accuracy** | **1.0** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### generator

* Dataset: generator
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 0.3
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset


* Size: 225 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 225 samples:
  |         | anchor                                                                            | positive                                                                        | negative                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                          | string                                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 11.69 tokens</li><li>max: 17 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 5.11 tokens</li><li>max: 7 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 5.17 tokens</li><li>max: 7 tokens</li></ul> |
* Samples:
  | anchor                                                                      | positive                                             | negative                                          |
  |:----------------------------------------------------------------------------|:-----------------------------------------------------|:--------------------------------------------------|
  | <code>Critical Care Clinical Nurse Specialist - ICU Leadership Role!</code> | <code>Critical Care Clinical Nurse Specialist</code> | <code>Back End Engineer</code>                    |
  | <code>Critical Care Clinical Nurse Specialist - ICU Leadership Role!</code> | <code>Critical Care Clinical Nurse Specialist</code> | <code>Training and Development Coordinator</code> |
  | <code>Critical Care Clinical Nurse Specialist - ICU Leadership Role!</code> | <code>Critical Care Clinical Nurse Specialist</code> | <code>Mortgage Processor</code>                   |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 0.3
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 5
- `max_steps`: 565
- `warmup_ratio`: 0.1

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 5
- `max_steps`: 565
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss | validation_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:--------------------------:|
| 0.0496 | 28   | 0.0098        | 0.0001          | 1.0                        |
| 0.0991 | 56   | 0.0022        | 0.0             | 1.0                        |
| 0.1487 | 84   | 0.0031        | 0.0             | 1.0                        |
| 0.1982 | 112  | 0.002         | 0.0             | 1.0                        |
| 0.2478 | 140  | 0.0011        | 0.0             | 1.0                        |
| 0.2973 | 168  | 0.0015        | 0.0             | 1.0                        |
| 0.3469 | 196  | 0.002         | 0.0             | 1.0                        |
| 0.3965 | 224  | 0.002         | 0.0             | 1.0                        |
| 0.4460 | 252  | 0.001         | 0.0             | 1.0                        |
| 0.4956 | 280  | 0.0011        | 0.0             | 1.0                        |
| 0.5451 | 308  | 0.0015        | 0.0001          | 1.0                        |
| 0.5947 | 336  | 0.0014        | 0.0001          | 1.0                        |
| 0.6442 | 364  | 0.0009        | 0.0             | 1.0                        |
| 0.6938 | 392  | 0.0009        | 0.0             | 1.0                        |
| 0.7434 | 420  | 0.0007        | 0.0             | 1.0                        |
| 0.7929 | 448  | 0.0004        | 0.0             | 1.0                        |
| 0.8425 | 476  | 0.0002        | 0.0             | 1.0                        |
| 0.8920 | 504  | 0.0003        | 0.0             | 1.0                        |
| 0.9416 | 532  | 0.0008        | 0.0             | 1.0                        |
| 0.9912 | 560  | 0.0004        | 0.0             | 1.0                        |
| 1.0    | 565  | -             | 0.0             | 1.0                        |


### Framework Versions
- Python: 3.10.16
- Sentence Transformers: 3.3.1
- Transformers: 4.49.0
- PyTorch: 2.2.2
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->