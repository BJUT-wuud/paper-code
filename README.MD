### Usage

代码的所有超参数都在py文件内配置，若修改需要在对应文件修改。

- **FT**：fine-tuning实验代码，直接运行fine-tuning.py文件即可
- **PT**：普通prompt tuning代码，若要复现论文结果。运行PT目录下以数据集名称为文件名的py文件即可。例如，要运行tnews数据集，则执行`PT/tnews.py`文件。
- **SOFT**：需要先运行目录下的train文件，训练完毕后再运行test文件。
- **PT(ours)**: 需要先运行`PT(ours)/label_words` 目录下的get_label_words.py文件从数据集中提取标签词，然后再执行如tnews.py之类的文件。但要注意存储verbalizer的文件路径问题。
- 对于Zero-shot的实验代码，注释掉对应文件的训练代码即可。