## 📊 Data Augmentation Workflow

## 🧭 User Guide

### 🔨 Installation

If you plan to use multiple models such as **GLM3**, **GLM4**, **QWen2**, and **Llama3**,
it is recommended to create **two separate conda environments** to ensure compatibility between old and new versions of the `transformers` library.
Specifically, **GLM3** requires an older version of `transformers` that is **incompatible** with newer models.

If you only plan to use **GLM3** or **other models excluding GLM3**,
you can simply install one environment accordingly.

---

#### 1. Create a Conda Environment

```bash
conda create -n llm python=3.10
conda activate llm
```

---

#### 2. Install Dependencies

* For **GLM3** models:

```bash
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

* For **other models (GLM4, QWen2, Llama3, etc.)**:

```bash
pip install protobuf transformers==4.44 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate tiktoken
```

---

### 🚀 Running the Pipeline

#### Definition

In **LLM-based Data Augmentation (LLM-DA)**, large language models are used to enrich low-resource datasets via:

* **Entity and POS extension labeling**, producing the `fusion` dataset.
* **Entity description generation**, producing the `syn_fusion` dataset.

> 💡 *If you wish to run the pipeline inside a Jupyter notebook, set `cmd_args=False`.*

---

#### Step 1. Entity & POS Extension Labeling

```bash
python syn_pipelines/s1_predict_data.py \
    --n_gpu=0 \
    --skip=-1 \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --model_from_pretrained="<LLM_PATH>" \
    --batch_size=20 \
    --mode=entity
```

**Arguments:**

* `n_gpu`: GPU index to use (default: 0)
* `skip`: Number of processed files to skip (default: -1, process all)
* `file_dir`: Root directory of all datasets (default: `"./data/few_shot"`)
* `file_name`: Dataset name, e.g., `"weibo"`. Ensure `./data/few_shot/weibo/train_1000.jsonl` exists.
* `save_type_name`: Prefix for saving generated files (default: `"GLM4"`)
* `model_from_pretrained`: Path to the pretrained model
* `batch_size`: Batch size for inference (default: 20)
* `mode`: Labeling mode (`entity` or `pos`, default: `"entity"`)

You can usually keep the default values unless special customization is required.

---

#### Step 2. Generate Enriched Entity Descriptions (Synthetic Data)

```bash
python syn_pipelines/s2_continous_generation.py \
    --n_gpu=0 \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --model_from_pretrained="<LLM_PATH>" \
    --batch_size=40
```

This step synthesizes detailed entity explanations using LLMs, producing the `syn_fusion` dataset.

---

#### Step 3. Extend Entity & POS Labels for Synthetic Data

Parameters are the same as in **Step 1**:

```bash
python syn_pipelines/s3_predict_syn_data.py \
    --n_gpu=0 \
    --skip=-1 \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --model_from_pretrained="<LLM_PATH>" \
    --batch_size=20 \
    --mode=entity
```

---

#### Step 4. Split Data Automatically by Ratio

Split the dataset into subsets with `25%`, `50%`, and `100%` training samples:

```bash
python syn_pipelines/s4_split_data.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4
```

---

#### Step 5. Merge Extended Labels into the Original Dataset

Combine entity and POS annotations from **Step 1** and **Step 3** into the original data to produce both `fusion` and `syn_fusion` labeled datasets.

```bash
python syn_pipelines/s5_merge_data.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --is_syn=0 \
    --save_type_name=GLM4 \
    --label_prefix='' \
    --entity_label='./data/fusion_knowledge/entity_label.json' \
    --pos_label='./data/fusion_knowledge/pos_label.json'
```

**Arguments:**

* `is_syn`: Whether the dataset is synthetic (`0` = no, `1` = yes)
* `label_prefix`: Prefix for avoiding label name conflicts (optional)
* `entity_label`: Path to the standardized entity label file
* `pos_label`: Path to the standardized POS label file

---

#### Step 6. Combine Fusion and Synthetic-Fusion Labels

```bash
python syn_pipelines/s6_combine_syn_fusion_label.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4
```

This step merges `fusion` and `syn_fusion` labels for unified training.

---

#### Step 7. Predict Target Labels for Synthetic Data Using a Trained NER Model

Before this step, it’s recommended to train a model on the `fusion` dataset.
Then, use the trained model to predict labels on the synthetic dataset:

```bash
python syn_pipelines/s7_pred_syn_data.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --save_type_name=GLM4 \
    --from_pretrained='<PLM_PATH>' \
    --model_from_pretrained='<SAVE_PLM_PATH>' \
    --batch_size=4
```

**Arguments:**

* `from_pretrained`: Path to the base pretrained model
* `model_from_pretrained`: Path to the fine-tuned model (tokenizer config may not be included in this directory)

---

#### Step 8. Final Merge of Fusion and Synthetic-Fusion Labels

Integrate annotations from **Steps 1**, **3**, and **7**, merging `fusion` and `syn_fusion` labels, and finally combine `fusion` data into `syn_fusion`.

```bash
python syn_pipelines/s8_process_label_and_combine.py \
    --file_dir="./data/few_shot" \
    --file_name=weibo \
    --is_syn=0 \
    --save_type_name=GLM4 \
    --label_prefix='' \
    --entity_label='./data/fusion_knowledge/entity_label.json' \
    --pos_label='./data/fusion_knowledge/pos_label.json'
```

---

✅ **Final Output:**
After completing all steps, you will obtain:

* `fusion` dataset — enriched with entity and POS knowledge.
* `syn_fusion` dataset — further enhanced with contextual entity explanations synthesized by LLMs.

These datasets can be directly used for **training the KnowFREE framework** to achieve robust performance in **low-resource sequence labeling tasks**.
