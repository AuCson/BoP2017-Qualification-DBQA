# Qualification
BOP 2017 in China, Qualification Round.<br>
Task: (DBQA-Problem)Given a question an a set of candidate answers, select the most proper answer.
# Sentence Structure Model
1. 预处理：包括清除非unicode字符。
2. 使用Stanford-CoreNLP工具，进行中文分词，词性标注，依赖标注等。
3. 多次实验后选择的特征。
* 主谓结构的词重复度
* 动宾结构的词重复度
* 重复词在特定问题的候选答案集合内的的信息熵之和
