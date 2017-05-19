#!/bin/sh
echo **Please run the shell at project root directory**

cd stanford-corenlp-full-2016-10-31

#Modify it for yourself
dir=`/home/xsjin/bop2017/Qualification/Qualification/data/package`
echo The path is $dir
for FILE in `ls $dir`
do
    java -mx3g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -props StanfordCoreNLP-chinese.properties -file ${dir}/${FILE}
    echo finished $FILE
done

