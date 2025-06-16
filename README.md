
## Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM
### Introduction
This repository contains the data and code for the paper [Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM](https://sites.northwestern.edu/icail2025/accepted-papers/) 


### Requirements
- torch==2.3.0
- numpy==1.26.4
- scikit-learn==1.5.0
- transformers==4.41.2
- tqdm==4.66.4

### Main process

1. Fisrt, **train a Supervised Classification Model**. To begin training, navigate to the project directory. From there, select the appropriate script based on your desired model (CNN, RNN, BERT, or Hierarchical BERT) and dataset. Please note that for different BERT models, you will need to manually download them from Hugging Face or GitHub and update the **bert_path** accordingly. Here's an example:
```shell
python main/train_CNN_cail.py
```

2. Second, run step2_probs-index2lawname.py script serves as a crucial **post-processing** utility for multi-label legal classification tasks. It efficiently takes the probability outputs from a Supervised Classification Model(such as a  BERT), identifies the top k most probable law articles for each input case, and then saves these extracted law names into a structured JSON file. 
```shell
python llm/step2_probs-index2lawname.py
```

3. Third, run step3_llm_main_random_version_Call_twice_crime-bert.py. This script facilitates **a sophisticated two-stage inference pipeline utilizing a Large Language Model (LLM) for accurate legal article prediction**. Initially, it takes the top-k law predictions from a Supervised Classification Model and the case facts to generate a comprehensive legal analysis via the LLM. Subsequently, this derived analysis, combined with the original facts and candidate laws, is used to prompt the LLM for the final legal article determination. 
```shell
python llm/step3_llm_main_random_version_Call_twice_crime-bert.py
```

* Crucially, **to manage LLM API costs effectively**, this script is designed for iterative execution in small batches. It should be run in a loop with the evaluation script (next step); you continue this cycle until the evaluation results demonstrate convergence, at which point you can cease further LLM calls.



4. Lastly, run step4_Post_processing_random_version_crime-bert.py. This script **evaluates the final legal article prediction results**.
```shell
python llm/step4_Post_processing_random_version_crime-bert.py
```

---


### Related link
1. [CAIL2018.zip](https://huggingface.co/datasets/china-ai-law-challenge/cail2018)
2. [ecthr-b](https://opil.ouplaw.com/display/10.1093/law:epil/9780199231690/law-9780199231690-e791)
3. [cail_thulac.npy](https://drive.google.com/file/d/1_j1yYuG1VSblMuMCZrqrL0AtxadFUaXC/view?usp=drivesdk+ ) 
4. [w2id_thulac.pkl](https://drive.google.com/file/d/1jnNgilApBRnA2ihldOr1Ceaci_7aFtsD/view?usp=drive_link)
5. [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) 
6. [legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased) 
7. [刑事文书BERT](https://github.com/thunlp/OpenCLaP) 



### Contact
If you have any issues or questions about this repo, feel free to contact **m13527860108@163.com.**
### License
[Apache License 2.0](./LICENSE)
