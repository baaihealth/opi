import matplotlib.pyplot as plt
import numpy as np
import json

with open("./plot_chart/llama3.1_8b_instruct_metrics.json") as infile:
    llama = json.load(infile)

with open("./plot_chart/galactica_6.7b_metrics.json") as infile1:
    galactica = json.load(infile1)
    
llama_result = []
for i in llama.keys():
    if type(llama[i])==dict:
        for j in llama[i].keys():
            llama_result = llama_result + llama[i][j]
    else:
        llama_result = llama_result + llama[i]

galactica_result = []
for i in galactica.keys():
    if type(galactica[i])==dict:
        for j in galactica[i].keys():
            galactica_result = galactica_result + galactica[i][j]
    else:
        galactica_result = galactica_result + galactica[i]


tasks = ['EC number', 'Fold type', 'Subcellular localization', 
         'Keywords', 'GO terms', 'Function', 'Tissue location prediction from gene symbol', 
         'Cancer prediction from gene symbol', 'Cancer prediction from gene name']

metrics_list = [
    ['Precision New-392','Recall New-392','F1 New-392','Precision Price-149','Recall Price-149','F1 Price-149'],
    ['Fold accuracy','Superfamily accuracy','Family accuracy'],
    ['Accuracy'],
    ['Precision CSeq','Recall CSeq','F1 CSeq','Precision ISeq','Recall ISeq','F1 ISeq','Precision USeq','Recall USeq','F1 USeq'],
    ['Precision CSeq','Recall CSeq','F1 CSeq','Precision ISeq','Recall ISeq','F1 ISeq','Precision USeq','Recall USeq','F1 USeq'],
    ['RougeL CSeq','RougeL ISeq','RougeL USeq'],
    ['Precision','Recall','F1'],
    ['Precision','Recall','F1'],
    ['Precision','Recall','F1']
]
metrics_counts = [6, 3, 1, 9, 9, 3, 3, 3, 3]

#model results list
model_1_scores = llama_result  
model_2_scores = galactica_result  

# 创建一个画布和子图
fig, axs = plt.subplots(3, 3, figsize=(20, 20))

# 遍历每个任务，并绘制柱状图
start = 0
for i, (task, num_metrics) in enumerate(zip(tasks, metrics_counts)):
    row = i // 3
    col = i % 3
    end = start + num_metrics

    # 提取当前任务的指标数据
    metrics = np.arange(num_metrics)
    model_1_data = model_1_scores[start:end]
    model_2_data = model_2_scores[start:end]

    # 绘制柱状图
    width = 0.35
   
    axs[row, col].bar(metrics - width/2, model_1_data, width, label='OPI-full-1.61M-Llama-3.1-8B-Instruct', color='skyblue')
    axs[row, col].bar(metrics + width/2, model_2_data, width, label='OPI-full-1.61M-Galactica-6.7B', color='salmon')

    # 设置标题和标签
    axs[row, col].set_title(task)
    axs[row, col].set_xticks(metrics)
    axs[row, col].set_xticklabels(metrics_list[i],rotation=45)
    #axs[row, col].set_xticklabels([f'Metric {i+1}' for i in metrics],rotation=45)
    axs[row, col].legend()

    start = end

# 调整布局
plt.tight_layout()
#plt.show()
plt.savefig('./plot_chart/llama3.1_8b_instruct-vs-galactica_6.7b.png', dpi=300, bbox_inches='tight')