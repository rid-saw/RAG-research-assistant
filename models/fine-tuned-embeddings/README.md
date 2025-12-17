---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:220
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What metric, besides the win-lose rate system, is used to calculate
    scores of the answers provided by both RAG and LC when dealing with long open-ended
    responses?
  sentences:
  - 'from linearity and the sequence of operations used by PowerSGD. Hence, it sufﬁces
    to store the sums of the errors

    buffers taken across all GPUs with the same ordinal. When resuming from a checkpoint,
    we can divide the error buffers

    by the total number of machines and broadcast them.

    F. Details for Human Evaluation Experiments

    We start with a list of 1000 captions and generate one sample image per model
    per caption. Captions and sample images

    are then used to create 1000 image comparison tasks per experiment, which we submitted
    to Amazon’s Mechanical Turk.

    Each task was answered by ﬁve distinct workers. Workers were asked to compare
    two images and answer two questions

    about them: (1) which image is most realistic, and (2) which image best matches
    the shared caption. The experimental setup

    provided to workers is shown in Figure 13. One worker’s answers were disqualiﬁed
    due to a high rate of disagreement'
  - 'We use a win-lose rate system to compare LC and

    RAG, as illustrated in Figure 2. The horizontal

    yellow block represents the questions that the LLM

    answers correctly using LC, while the vertical blue

    block represents the questions that the LLM an-

    swers correctly using RAG. Their overlap in the

    top-left corner represents the questions that both

    methods answer correctly. We apply an Exact

    Match (EM) score strictly to all questions to de-

    termine the correctness of the answers. Excluding

    the overlap, the top right block indicates the ques-

    tions that only LCanswers correctly, and similarly,

    the bottom left block indicates the questions that

    only RAGanswers correctly.

    The remaining gray block represents the ques-

    tions that both RAG and LC answer incorrectly, as

    judged by Exact Match. Since many questions in-

    volve long open-ended responses, we calculate the

    F1 scores of the answers provided by both meth-

    ods against the ground truth. If RAG achieves a'
  - 'ComoRAG: A Cognitive-Inspired Memory-Organized RAG

    for Stateful Long Narrative Reasoning

    Juyuan Wang1* Rongchen Zhao1* Wei Wei2 Yufeng Wang1

    Mo Yu4 Jie Zhou4 Jin Xu1,3 Liyan Xu4†

    1School of Future Technology, South China University of Technology

    2Independent Researcher 3Pazhou Lab, Guangzhou

    4WeChat AI, Tencent

    Abstract

    Narrative comprehension on long stories and novels has been

    a challenging domain attributed to their intricate plotlines and

    entangled, often evolving relations among characters and en-

    tities. Given the LLM’s diminished reasoning over extended

    context and its high computational cost, retrieval-based ap-

    proaches remain a pivotal role in practice. However, tradi-

    tional RAG methods could fall short due to their stateless,

    single-step retrieval process, which often overlooks the dy-

    namic nature of capturing interconnected relations within

    long-range context. In this work, we propose ComoRAG,

    holding the principle that narrative reasoning is not a one-'
- source_sentence: What is the number given at the end of the passage directly before
    the SOURCE?
  sentences:
  - 'Input-Input Layer5

    The

    Law

    will

    never

    be

    perfect

    ,

    but

    its

    application

    should

    be

    just

    -

    this

    is

    what

    we

    are

    missing

    ,

    in

    my

    opinion

    .

    <EOS>

    <pad>

    The

    Law

    will

    never

    be

    perfect

    ,

    but

    its

    application

    should

    be

    just

    -

    this

    is

    what

    we

    are

    missing

    ,

    in

    my

    opinion

    .

    <EOS>

    <pad>

    Input-Input Layer5

    The

    Law

    will

    never

    be

    perfect

    ,

    but

    its

    application

    should

    be

    just

    -

    this

    is

    what

    we

    are

    missing

    ,

    in

    my

    opinion

    .

    <EOS>

    <pad>

    The

    Law

    will

    never

    be

    perfect

    ,

    but

    its

    application

    should

    be

    just

    -

    this

    is

    what

    we

    are

    missing

    ,

    in

    my

    opinion

    .

    <EOS>

    <pad>

    Figure 5: Many of the attention heads exhibit behaviour that seems related to
    the structure of the

    sentence. We give two such examples above, from two different heads from the encoder
    self-attention

    at layer 5 of 6. The heads clearly learned to perform different tasks.

    15'
  - 'views at an elevation of 30◦, a simpler setting than ours.

    Moreover, since these metrics do not assess 3D consistency

    across views, please refer to supplementary for additional

    qualitative comparisons and discussions.

    5. Conclusion

    In this paper, we introduced One-2-3-45++, an innovative

    approach for transforming a single image of any object into

    a 3D textured mesh. This method stands out by offer-

    ing more precise control compared to existing text-to-3D

    models, and it is capable of delivering high-quality meshes

    swiftly—typically in under 60 seconds. Additionally, the

    generated meshes exhibit a high fidelity to the original in-

    put image. Looking ahead, there is potential to enhance the

    robustness and detail of the geometry by incorporating addi-

    tional guiding conditions from 2D diffusion models, along-

    side RGB images.

    8'
  - '2 4 6 8

    Aesthetic Score

    0.0

    0.1

    0.2

    0.3Density

    Num. samples = 796,031

    Mean = 4.65

    Std = 1.02

    ObjaverseXL (sketchfab)

    Threshold: 5.5

    0 2 4 6 8

    Aesthetic Score

    0.0

    0.2

    0.4

    0.6

    0.8

    1.0Density

    Num. samples = 5,238,768

    Mean = 4.08

    Std = 0.90

    ObjaverseXL (github)

    Threshold: 5.5

    2 4 6 8

    Aesthetic Score

    0.0

    0.1

    0.2

    0.3

    0.4

    0.5Density

    Num. samples = 7,953

    Mean = 4.50

    Std = 0.86

    ABO

    Threshold: 4.5

    2 4 6 8

    Aesthetic Score

    0.0

    0.1

    0.2

    0.3

    0.4

    0.5Density

    Num. samples = 16,563

    Mean = 4.64

    Std = 0.81

    3D-FUTURE

    Threshold: 4.5

    0 2 4 6 8

    Aesthetic Score

    0.0

    0.1

    0.2

    0.3Density

    Num. samples = 14,099

    Mean = 4.38

    Std = 1.04

    HSSD

    Threshold: 4.5

    Figure 8. Distribution of aesthetic scores in each dataset.

    Score: 2.32 Score: 3.84 Score: 4.91 Score: 5.24

    Score: 5.85 Score: 6.04 Score: 6.29 Score: 7.03

    Figure 9. 3D asset examples from Objaverse-XL with their corre-

    sponding aesthetic scores.

    Table 8. Composition of the training set and evaluation set.

    Source Aesthetic Score Threshold Filtered Size'
- source_sentence: What prompt is added to the MultiLingo+ variant, as described in
    the text, to ensure English responses?
  sentences:
  - 'and MultiLingo+ variants. For the MultiLingo+

    variant, we add "Answer the following question in

    English" in the prompt, to ensure the response is

    provided in English.

    A.5 Generation Examples

    Table 5 exhibits examples generated by the model

    variants on the TruthfulQA and MMLU datasets.'
  - 'We use a win-lose rate system to compare LC and

    RAG, as illustrated in Figure 2. The horizontal

    yellow block represents the questions that the LLM

    answers correctly using LC, while the vertical blue

    block represents the questions that the LLM an-

    swers correctly using RAG. Their overlap in the

    top-left corner represents the questions that both

    methods answer correctly. We apply an Exact

    Match (EM) score strictly to all questions to de-

    termine the correctness of the answers. Excluding

    the overlap, the top right block indicates the ques-

    tions that only LCanswers correctly, and similarly,

    the bottom left block indicates the questions that

    only RAGanswers correctly.

    The remaining gray block represents the ques-

    tions that both RAG and LC answer incorrectly, as

    judged by Exact Match. Since many questions in-

    volve long open-ended responses, we calculate the

    F1 scores of the answers provided by both meth-

    ods against the ground truth. If RAG achieves a'
  - 'score by calculating the average score for each modality and then averaging across
    all modalities.

    Implementation Details For generations, we use multiple LVLMs, including InternVL2.5-8B
    [14],

    Qwen2.5-VL-7B-Instruct [5], and Phi-3.5-Vision-Instruct [1]. Also, we use retrievers
    tailored to each

    5'
- source_sentence: According to the document, what type of models are leveraged to
    process PDF documents in batches within the novel multimodal document chunking
    approach?
  sentences:
  - 'Retrievable;

    Entry points from vec-

    tor similarity.

    "Due to the increasing impor-

    tance of AI, the Nobel Prize is

    awarded to scholars who have

    made tremendous contributions

    to the field of AI."

    O High-Level Overview

    Titles or keywords summariz-

    ing

    high-level elements.

    Non-Retrievable;

    Entry points from accu-

    rate search.

    "AI significance"

    R Relationship

    Connections between entities

    represented as nodes. Acts

    as connector nodes and sec-

    ondary retrievable node.

    Retrievable;

    Non-Entry points

    "Hinton received the Nobel

    Prize."

    N Entity Named entities such as peo-

    ple, places, or concepts.

    Non-Retrievable;

    Entry points from accu-

    rate search..

    "Hinton," "Nobel Prize"

    Table 6: Node Types in the heterograph

    C.2 K-core & Betweenness centrality

    In this subsection, we present the methodology for identifying important entities
    and generating their

    attribute summaries, ensuring alignment with the mathematical framework established
    in the main text.'
  - 'generative modeling through stochastic differential equations.

    arXiv preprint arXiv:2011.13456, 2020. 3, 4, 17

    [77] Matthew Tancik, Ben Mildenhall, Terrance Wang, Divi

    Schmidt, Pratul P Srinivasan, Jonathan T Barron, and Ren Ng.

    Learned initializations for optimizing coordinate-based neural

    representations. In IEEE Conference on Computer Vision and

    Pattern Recognition (CVPR), pages 2846–2855, 2021. 2

    [78] Ayush Tewari, Justus Thies, Ben Mildenhall, Pratul Srinivasan,

    Edgar Tretschk, W Yifan, Christoph Lassner, Vincent

    Sitzmann, Ricardo Martin-Brualla, Stephen Lombardi, et al.

    Advances in neural rendering. InComputer Graphics Forum,

    volume 41, pages 703–735. Wiley Online Library, 2022. 1, 2

    11'
  - 'Vision-Guided Chunking Is All You Need:

    Enhancing RAG with Multimodal Document

    Understanding

    Vishesh Tripathi◦, Tanmay Odapally◦, Indraneel Das◦, Uday Allu†,

    and Biddwan Ahmed†

    AI Research Team, Yellow.ai

    Abstract

    Retrieval-AugmentedGeneration(RAG)systemshaverevolutionizedinformationretrievalandques-

    tion answering, but traditional text-based chunking methods struggle with complex
    document struc-

    tures, multi-page tables, embedded figures, and contextual dependencies across
    page boundaries.

    We present a novel multimodal document chunking approach that leverages Large
    Multimodal

    Models (LMMs) to process PDF documents in batches while maintaining semantic coherence
    and

    structural integrity. Our method processes documents in configurable page batches
    with cross-batch

    context preservation, enabling accurate handling of tables spanning multiple pages,
    embedded visual

    elements, and procedural content. We evaluate our approach on our internal benchmark
    dataset'
- source_sentence: According to the positional encodings described, what range of
    wavelengths are formed by the geometric progression, beginning with 2π?
  sentences:
  - '24'
  - 'Enhancing Retrieval-Augmented Generation: A Study of Best Practices

    Siran Li Linus Stenzel Carsten Eickhoff Seyed Ali Bahrainian

    University of Tübingen

    siran.li@uni-tuebingen.de, stenzel@student.uni-tuebingen.de,

    {carsten.eickhoff, seyed.ali.bahreinian}@uni-tuebingen.de

    Abstract

    Retrieval-Augmented Generation (RAG) sys-

    tems have recently shown remarkable advance-

    ments by integrating retrieval mechanisms into

    language models, enhancing their ability to

    produce more accurate and contextually rel-

    evant responses. However, the influence of

    various components and configurations within

    RAG systems remains underexplored. A com-

    prehensive understanding of these elements is

    essential for tailoring RAG systems to com-

    plex retrieval tasks and ensuring optimal perfor-

    mance across diverse applications. In this pa-

    per, we develop several advanced RAG system

    designs that incorporate query expansion, vari-

    ous novel retrieval strategies, and a novel Con-'
  - 'tokens in the sequence. To this end, we add "positional encodings" to the input
    embeddings at the

    bottoms of the encoder and decoder stacks. The positional encodings have the same
    dimension dmodel

    as the embeddings, so that the two can be summed. There are many choices of positional
    encodings,

    learned and fixed [9].

    In this work, we use sine and cosine functions of different frequencies:

    P E(pos,2i) = sin(pos/100002i/dmodel )

    P E(pos,2i+1) = cos(pos/100002i/dmodel )

    where pos is the position and i is the dimension. That is, each dimension of the
    positional encoding

    corresponds to a sinusoid. The wavelengths form a geometric progression from 2π
    to 10000 · 2π. We

    chose this function because we hypothesized it would allow the model to easily
    learn to attend by

    relative positions, since for any fixed offset k, P Epos+k can be represented
    as a linear function of

    P Epos.

    We also experimented with using learned positional embeddings [9] instead, and
    found that the two'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    'According to the positional encodings described, what range of wavelengths are formed by the geometric progression, beginning with 2π?',
    'tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the\nbottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel\nas the embeddings, so that the two can be summed. There are many choices of positional encodings,\nlearned and fixed [9].\nIn this work, we use sine and cosine functions of different frequencies:\nP E(pos,2i) = sin(pos/100002i/dmodel )\nP E(pos,2i+1) = cos(pos/100002i/dmodel )\nwhere pos is the position and i is the dimension. That is, each dimension of the positional encoding\ncorresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We\nchose this function because we hypothesized it would allow the model to easily learn to attend by\nrelative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of\nP Epos.\nWe also experimented with using learned positional embeddings [9] instead, and found that the two',
    '24',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

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

#### Unnamed Dataset


* Size: 220 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 220 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 15 tokens</li><li>mean: 29.0 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 210.74 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What publication describes "Deepseek-v2" as a mixture-of-experts language model and where was it published according to abs/2405.04434?</code> | <code>uan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng,<br>Chong Ruan, Damai Dai, Daya Guo, Dejian Yang,<br>Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli<br>Luo, Guangbo Hao, Guanting Chen, and Guowei Li<br>et al. 2024. Deepseek-v2: A strong, economical, and<br>efficient mixture-of-experts language model. CoRR,<br>abs/2405.04434.<br>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,<br>Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,<br>Akhil Mathur, Alan Schelten, Amy Yang, and An-<br>gela Fan et al. 2024. The llama 3 herd of models.<br>CoRR, abs/2407.21783.<br>Weizhi Fei, Xueyan Niu, Pingyi Zhou, Lu Hou, Bo Bai,<br>Lei Deng, and Wei Han. 2024. Extending context<br>window of large language models via semantic com-<br>pression. In Findings of the Association for Compu-<br>tational Linguistics, ACL 2024, Bangkok, Thailand<br>and virtual meeting, August 11-16, 2024, pages 5169–<br>5181. Association for Computational Linguistics.<br>Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,<br>Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,</code>   |
  | <code>According to the paper, what type of feature volume is incorporated to include geometry priors in the diffusion-based model?</code>            | <code>Generative Novel View Synthesis with 3D-Aware Diffusion Models<br>Eric R. Chan*†1,2, Koki Nagano*2, Matthew A. Chan*2, Alexander W. Bergman*1, Jeong Joon Park*1,<br>Axel Levy1, Miika Aittala2, Shalini De Mello2, Tero Karras2, and Gordon Wetzstein1<br>1Stanford University 2NVIDIA<br>Abstract<br>We present a diffusion-based model for 3D-aware gen-<br>erative novel view synthesis from as few as a single input<br>image. Our model samples from the distribution of possible<br>renderings consistent with the input and, even in the presence<br>of ambiguity, is capable of rendering diverse and plausible<br>novel views. To achieve this, our method makes use of existing<br>2D diffusion backbones but, crucially, incorporates geom-<br>etry priors in the form of a 3D feature volume. This latent<br>feature ﬁeld captures the distribution over possible scene rep-<br>resentations and improves our method’s ability to generate<br>view-consistent novel renderings. In addition to generating<br>novel views, our method has the ability to autoregressively</code> |
  | <code>According to the passage, what does VIDEO RAG-V retrieve to generate a response similar to the ground truth about baking cookies?</code>       | <code>ies on your car dashboard”. As shown in Table 5,<br>the NAÏVE baseline, relying solely on its parametric<br>knowledge, generates a generic response highlight-<br>ing the impracticality and safety concerns of such a<br>method, failing to provide the step-by-step instruc-<br>tions necessary to address the query. This example<br>indicates the limitation of parametric knowledge<br>that is inadequate, especially when specific and<br>uncommon information is required. In contrast,<br>VIDEO RAG-V retrieves the relevant video that il-<br>lustrates the process of baking cookies on a car<br>dashboard, and, by leveraging this, it successfully<br>generates a response similar to the ground truth.<br>This highlights how VideoRAG utilizes external<br>video content to produce more precise, contextu-<br>ally rich, and actionable answers. We provide an<br>additional example in Table 12 of Appendix D.<br>4 Related Work<br>Retrieval-Augmented Generation RAG is a<br>strategy that combines retrieval and generation pro-</code>                                   |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
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
- `hub_private_repo`: False
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
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.9.6
- Sentence Transformers: 3.3.1
- Transformers: 4.46.3
- PyTorch: 2.8.0
- Accelerate: 1.10.1
- Datasets: 4.4.1
- Tokenizers: 0.20.3

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

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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