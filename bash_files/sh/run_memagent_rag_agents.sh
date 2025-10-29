source ~/.bashrc
source /scratch/yzhu/anaconda3/etc/profile.d/conda.sh
conda activate MABench

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)


file_name=rag_agents.txt

for line in {204..204..1}
    do
        cfg=$(sed -n "$line"p ${root}/bash_files/configs/${file_name})
        agent_config=$(echo $cfg | cut -f 1 -d ' ')
        dataset_config=$(echo $cfg | cut -f 2 -d ' ')

        echo ................Start........... 
        CUDA_VISIBLE_DEVICES=0 python main.py \
                                            --agent_config      configs/agent_conf/RAG_Agents/gpt-4o-mini/${agent_config} \
                                            --dataset_config    configs/data_conf/${dataset_config} 
        echo ................End...........

    done

# bash bash_files/sh/run_memagent_rag_agents.sh   
