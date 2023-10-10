#!/bin/bash

# Main python script path
main_script="main.py"

# Common parameters
dataset="C4_sentiment-analysis"
model="text_LSTM"
data_dir="../data/C4_sentiment-analysis/"
objective="one-class"
lr="0.0007"
n_epochs="150"
lr_milestone="50"
batch_size="200"
weight_decay="0.5e-6"
pretrain="True"
ae_lr="0.0008"
ae_n_epochs="150"
ae_lr_milestone="50"
ae_batch_size="90"
ae_weight_decay="0.5e-6"

cd ..

# Iterating over classes
for class in {0..2}
do
    # Prepare the log directory path
    log_dir="../log/C4_sentiment-analysis/class${class}"

    # Create log directory if it does not exist
    mkdir -p "${log_dir}"

    # Run the python command for each class
    python ${main_script} ${dataset} ${model} ${log_dir} ${data_dir} --objective ${objective} --lr ${lr} --n_epochs ${n_epochs} --lr_milestone ${lr_milestone} --batch_size ${batch_size} --weight_decay ${weight_decay} --pretrain ${pretrain} --ae_lr ${ae_lr} --ae_n_epochs ${ae_n_epochs} --ae_lr_milestone ${ae_lr_milestone} --ae_batch_size ${ae_batch_size} --ae_weight_decay ${ae_weight_decay} --normal_class ${class};
done

