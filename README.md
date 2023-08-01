# [PromptCBLUE比赛](https://tianchi.aliyun.com/competition/entrance/532084/introduction)方案分享--队伍Jarvis-XJTU

## 比赛简介

该比赛改造CBLUE数据集，形成了一个可以用于训练LLM的基于自然语言的中文医疗任务数据集。该数据集一共含有16个任务，总数据量69k+。[[官网]](https://tianchi.aliyun.com/competition/entrance/532084/introduction)

我们队伍Jarvis-XJTU的A榜成绩为第15名，B榜成绩为第7名。[[排行榜]](https://tianchi.aliyun.com/competition/entrance/532084/rankingList) 



## 环境

我们使用的基础软件环境为：

> python=3.9.5
>
> cuda=10.2

使用的硬件环境为：

> Tesla V100



## 方案基本思路

我们在A榜阶段主要尝试了如下几个思路：（均以chatglm-6B作为基座）

1. P-Tuing: 按照baseline中的配置进行设置，A榜结果58.2144
2. LoRA: 按照baseline的设置，max_steps=3000，A榜测试结果67.221
3. Batched LoRA: 为了使得训练更加稳定，我们通过重写sampler来替代原始的sampler，实现每个batch中样本的任务类型相同。实现代码放于`utils\sampler.py`。但最终结果只在个别任务上有提升。A榜测试结果66.4537.
4. LoRA: 重新进行调参，max_steps=8000, max_source_length=1024, 并进行数据增强。对Matching, Normalization等题型进行数据增强。具体思路就是将指令中LIST_LABELS中的选项进行调换。数据增强代码见`datasets\pre_data\data_augmentation.py`。A榜测试结果为69.7511.
5. 使用ChatMed数据集利用LoRA训练一个新的chatglm基座。思路是先训练一个chatmed的权重，然后将lora权重与原始的chatglm权重进行融合。融合的代码见`merge_chatglm.py`。时间原因只测试了max_steps=5000的结果，不加chatmed的A榜测试结果为69.35，添加了chatmed基座的A榜测试结果为69.5102.
6. 使用chatglm2-6b作为基座，进行LoRA微调，A榜测试结果为69.3521。相比于chatglm-6b基座，仅在对话生成和报告生成两个任务上有提升，其他都没有提升。
7. 分别对每个任务进行LoRA微调，即将数据集按任务进行拆分，然后分别在每个任务的数据集上进行LoRA微调和测试。默认设置所有的任务的max_steps=500, lora_rank=8。A榜测试结果为69.5489. __这里我们发现单任务微调的效果更好，因此确定最终的方案为单任务LoRA微调__。
8. 为了满足赛题方对于微调参数总量不超过1%的要求，我们对每个任务设置lora_rank=2。在此设置下，每个任务的可训练参数量占全部参数量的0.0592%，因此16个任务总的参数量为$0.0592\% \times 16 = 0.948\%$。设置每个任务的max_steps=500, A榜测试结果为69.6118.
9. 我们发现在CTC任务中原始标签有"(医生检测）"，这个标签的左括号为英文括号，右括号为中文括号。这样的选项形式被LLM全部预测为"(医生检测)"。我们手动对这个选项进行替换，A榜测试结果为69.8838.
10. 我们对单任务CHIP-CTC数据使用CBLUE数据集进行了增强，A榜测试结果为70.27. 具体增强的方法在下文阐述。

我们在B榜依照A榜中思路8，主要是基于chaglm-6b，对每个任务进行单任务的微调。此外，我们利用CBLUE数据集对几个单任务进行了数据增强，达到了最终的测试结果。

1. CBLUE数据增强的方法：首先，我们发现数据中存在很多的重复现象，即同一个内容使用了不同的指令模板在数据集中出现了多次。我们了避免数据重复的问题，我们依照模板从现有的PromptCBLUE数据集中，提取该任务数据的内容，并删掉重复（比如CHIP-CTC数据中有6000条数据，但是不重复的数据只有3622，因此我们只保留这3622条数据）。然后，我们发现CBLUE数据集中有更多的内容，因此从CBLUE中提取未出现过的数据，和原始数据组成新的数据。最后，随机采样官方提供的模板`templates.json`，重构得到新的数据。
2. 我们依照第1点，扩增了数据集CHIP-CTC（3622样本扩增至6000），CHIP-CDEE（1361样本扩增至3176），CMeIE（2828样本扩增至6000）。将CMeIE和CHIP-CTC的max_steps调整为800.



## 代码运行

1. 安装必要的package

   ```bash
   pip install -r requirements.txt
   ```

2. 将数据`train.json`, `dev.json`和`test.json`放于`datasets\pre_data\`文件夹下。运行`data_augmentation.py`来获得增强的数据（但在我们最终的方案并未用到）。运行`partition.ipynb`将所有数据按照任务进行分割，并输出到文件夹`datasets\pre_data\partition\`下。

3. 将CBLUE数据集中对应任务的数据放于`datasets\pre_data\partition\<task>`下，并运行该文件夹下的`analysis.ipynb`以生成经过CBLUE数据增强的数据。需要运行的只有三个数据集：CHIP-CTC，CHIP-CDEE和CMeIE。

4. 准备chatglm-6b基底。将chatglm-6b的所有文件下载至文件夹`resources\chatglm-6b`下。

5. 运行bash文件以完成训练和测试：

   ```bash
   bash experiments\partition_lora.bash
   ```

   注意根据拥有的资源数量进行gpu配置，主要是gpu数量。修改bash中的`--num_gpu`参数。此外，修改`run_lora.py`中可见的gpu序号：`os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"`.

6. 默认各个任务的测试结果存在文件夹`submission\<task>`文件下。运行`submission\post_handler.ipynb`来处理之前第10点里CHIP-CTC中的问题。然后运行`submission\fuse.ipynb`来将各任务的输出结果整合成提交要求的`test_predictions.json`。

7. 运行bash生成提交要求的zip文件：

   ```bash
   bash submission\res_zip.bash
   ```

   







