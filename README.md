# Team NLPOT@BoP2017：Qualification Round
BOP 2017 in China, Qualification Round.<br>
微软编程之美2017（中国），资格赛
Task: (DBQA-Problem)Given a question an a set of candidate answers, select the most proper answer.
### Prediction model based on sentence structure 基于句法结构的模型
1. 预处理：包括清除非unicode字符。(File-Preprocessor.py)<br>
因为比赛使用的数据含大量非unicode字符。这些字符会妨碍stanford-corenlp工具进行分词，标注等。另外，对于没有显式标点结尾的语句，我们为它们增加了句号。
2. 使用Stanford-CoreNLP工具，进行中文分词，词性标注，依赖标注等（请见：https://stanfordnlp.github.io/CoreNLP/ ）。.sh为处理的命令行文件。输出格式为.xml，含有POS，NER，Parse，Dependency等记录。对于生成的Parse-tree（括号形式的树），semantic-tree.py可以很方便地解析并进行一些查询特定条件的节点的操作。semantic-tree.py在之后的比赛过程中丰富了其功能。
3. 多次实验后选择的特征。
* 主谓结构的词重复度。（根据parse-tree通过一些规则判断出主谓语）
* nsubj（类似于动宾）依赖结构的词重复度<br>
* 重复词在特定问题的候选答案集合内的的信息熵之和。通俗讲，对答案中出现的与问题重复的词的个数加权相加得到分数，这一词在其他候选答案出现得越少，权重就越大。<br>
另外，还有一些经实验后弃用的特征
* 具有非平凡NER-tag的词重复度。这会降低评测用的MRR（正确答案在候选答案中评分排位的倒数）。譬如，对于某一“地点”的问题，可能诸多候选答案中都含有此地点。（事实上，上述第三个特征是对此特征的改进）
* POS中主语，宾语等的重复度。（事实上，上述第一，二个特征是对此特征的改进）<br>
根据中文的特征，词重复度单纯使用字重复度进行判断。曾使用中文WordNet，但是中文WordNet词数略显匮乏，而且会降低性能，不能满足需要。
4. 机器学习模型。提取特征后，简单使用Logistic Regression进行评分。<br>
结果：通过资格赛。自行在毎10000组数据测试，MRR在0.55-0.7内浮动。平均约为0.63。
