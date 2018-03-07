NLP-tools
==
本项目旨在通过Tensorflow基于BiLSTM+CRF实现中文分词、词性标注、命名实体识别（NER）。
模型架构：输入层-->嵌入层-->双向长短记忆网络CELL-->输出层。
欢迎各位大佬吐槽。

模型训练
--
        corpus文件下放入语料集，语料格式：人民网/nz 1月4日/t 讯/ng 据/p [法国/nsf 国际/n。\<br> 
        执行python train.py 开始训练。\<br> 
        训练生成嵌入矩阵存入data/data.pkl。\<br> 
        训练生成checkpoint存入ckpt。\<br> 

模型超参数
--
        *分词标记方式：4tags。\<br> 
        *嵌入层向量长度：64。\<br> 
        *BiLstm层数：2。\<br> 
        *隐藏层节点数：128。\<br> 
        *最大迭代次数：6。\<br> 
        *Batch宽度：128。\<br> 
        *初始学习率：1e-4（采用动态形式，随训练进行而减小步长）。\<br> 
    
模型测试
--
        将待分词项写入test/test文件中，执行python model_test.py，生成结果存入test/test_result。\<br> 
现状
--
目前模型尚处于初步测试成功，分词部分完成，正确率94%（有待提高，超参数稍后公布）。\<br> 
后期陆续整理出POS\NER以及Parse功能。\<br> 
后期给出相关文献参考。\<br> 
