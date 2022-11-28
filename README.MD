### Usage

All hyperparameters of the code are configured in the py file. If modification is required, modification is required in the corresponding file.


- **FT** : fine-tuning experimental code, directly run the fine-tung. py file can be

- **PT** : common prompt tuning code, if you want to reproduce the results of the paper. Run the py file in the PT directory with the data set name as the file name. For example, to run the tnews dataset, execute the `PT/tnews.py` file.

- **SOFT** : Run the train file in the directory first, and then run the test file after the training is complete.

- **PT(ours)**: You need to run the `get_label_words.py` file in the `PT(ours)/label_words` directory to extract the label words from the data set before executing a file such as tnews.py. But be aware of the file path for storing the verbalizer.

- For the experimental code of Zero-shot, comment out the training code of the corresponding file.

### 用法

代码的所有超参数都在py文件内配置，若修改需要在对应文件修改。

- **FT**：fine-tuning实验代码，直接运行fine-tuning.py文件即可
- **PT**：普通prompt tuning代码，若要复现论文结果。运行PT目录下以数据集名称为文件名的py文件即可。例如，要运行tnews数据集，则执行`PT/tnews.py`文件。
- **SOFT**：需要先运行目录下的train文件，训练完毕后再运行test文件。
- **PT(ours)**: 需要先运行`PT(ours)/label_words` 目录下的get_label_words.py文件从数据集中提取标签词，然后再执行如tnews.py之类的文件。但要注意存储verbalizer的文件路径问题。
- 对于Zero-shot的实验代码，注释掉对应文件的训练代码即可。