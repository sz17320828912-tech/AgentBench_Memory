# ⚙️ MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions

This project benchmarks agents with memory capabilities. Follow the steps below to set up your environment and install dependencies. 

[Full paper](https://arxiv.org/abs/2507.05257)

## 🚧 Update

- [x] (July 7th, 2025) We released the code for reproducing the main experiment. 

**🌟 More details (such as datasets collection) coming soon! 🌟**


## 🚀 Quick Setup

### 1. Create a Conda Environment

It’s recommended to use a dedicated conda environment for reproducibility:
```
conda create --name MABench python=3.10.16
```

### 2. Install Python Dependencies

```
pip install torch
pip install -r requirements.txt
pip install "numpy<2"
```
We did not include the `hipporag` in `requirements.txt` since the current version of `hipporag` will cause some conflicts on pacakge version. You can create another environment with hipporag instead.  

Sometimes you can try to supplement the lacked packages for `cognee` and `letta`. If you met some package related errors after installing `requirements.txt`. 
```
pip install letta
pip uninstall letta   
pip install cognee
pip uninstall cognee
```

## 📥 Data Download & API Settings

To use this project, you need to download the processed data files and place them in the correct directory.

### 1. Download the Data from HuggingFace 🤗 

- HuggingFace dataset [link](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench). It can be automatically downloaded if you run the code directly. 

- Do not forget the `entity2id.json` for Movie Recommendation task.


### 2. Environment Variable Settings

To run this project, you need to configure your API keys and model settings in a `.env` file at the project root.

Create a `.env` file and add the following content, replacing the placeholder values with your actual API keys:

#### OpenAI API Keys

```
OPENAI_API_KEY= ###your_openai_api_key
```

#### Settings for Cognee
```
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=  ###your_api_key
```

#### Other API Keys
```
Anthropic_API_KEY= ###your_anthropic_api
Google_API_KEY=    ###your_google_api
```

## 🏃‍♂️ Run Evaluation

Follow these steps to evaluate the benchmarking agent:


### Run Example Evaluation Command

You can run an evaluation using the following example command:

#### Long Context Agents
```
bash bash_files/eniac/run_memagent_longcontext.sh
```
- `--agent_config`: Path to the agent/model configuration file.
- `--dataset_config`: Path to the dataset configuration file.

#### Rag Agents and Agentic Memory Methods

```
bash bash_files/eniac/run_memagent_rag_agents.sh
```
#### Ablation Study for Chunk Size
```
bash bash_files/eniac/run_memagent_rag_agents_chunksize.sh
```

Remember that `hipporag (2.0.0a3)` reuqires `openai==1.58.1`, which may cause some latest OpenAI models could not be used in same environment. 

