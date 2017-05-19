# Qualification
BOP2017, Qualification Round.
更新0519，一个效果很不理想的Sentence Structure Based Model
# Sentence Structure Model By Xisen
基于问题与回答的句子结构的模型，下面讲讲是如何实现的吧<br>
目录文件（部分并没有提交到git）
>data/<br>
>  BoP2017-train.txt<br>
>  package (问题，答案，label分为单个文件并分块存储；问题，答案每个文件10000条)<br>
>  parsed/ （经过stanford corenlp工具生成的xml，非常大，8GB）<br>
>cow/<br>
>  一个表现非常一般的将中文单词映射到英文WordNet上的模块。不知哪里有离线中英词典可以替代。<br>
>stanford corenlp/ （coreNLP工具)<br>

思路是这样的：
1. 通过coreNLP对中文句子进行分词并解析出句子的POS(词性)，NER（命名实体（时间，地名等））。
2. 目前选取如下的特征
* 问题中出现的NER在答案中也出现的比例
* 答案中出现的NER在问题中没有出现的比例
* 问题中POS标签为NN（名词）在答案中所有名词的最高WordNet相似度（Path-similarity,即WordNet网络上两个概念最短路的长度为标准打分）
* 问题中POS标签为VV（动词）的最高WordNet相似度
3. 分类模型：
暂时使用Scikit-learn中的LogisticRegression，借鉴了上学期的代码。
4. 效果：
非常差劲。几乎完全无法分辨……

目前的思路：
1. 将目前的特征III与IV中相似度计算局限在仅仅与疑问词或者关键词在句法上有弧相连的词语上，减少计算量。
2. 相似度计算中“信息熵”这一概念可能会有用—，即，某个问题的所有备选答案中反复出现的词的“反复出现”是没有太大意义的升
3. 关于分类模型我也不很懂怎样设置比较好。希望能和比较懂得人有稍微深入的交流。
