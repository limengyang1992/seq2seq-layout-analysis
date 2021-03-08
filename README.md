
# 票据类版面分析算法


当前OCR之后的版面分析工作大家都是规则写的，本人也深受规则之苦,看到ocr输出的一大堆文字和坐标就头皮发麻。最近受了chineseocr作者模板引擎的启发，做了个基于seq2seq的端到端版面分析算法，希望能够帮到各位ocrer。

思路：通过有监督方式训练句向量，然后通过seq2seq的方式学习box类别

链接：https://blog.csdn.net/mochp/article/details/109491521


## 使用方法

   - 准备数据
     
     - 首先利用自己的OCR算法，将票据图片文字识别出来
     - 将结果存入labelme
     - 在groupid里标注所需要提取的box类别
     - 将标注数据放入data对应的train和test文件夹下

   - 修改config.py其中5个参数，其余可根据情况调试
     ```
      self.class_char               # 标签列表（对应groupid）
      self.max_text_len = 20        # 最大文本长度
      self.max_box_num = 50         # 最大box个数
      self.expend_box_times = 8     # box扩增倍数
      self.rnn_hidden_size = 64     # 句向量维度
     ```
        
   - 训练句向量
     
     ```
      python processing.py   #数据预处理
      python rnn_w2v.py      #训练词向量
      python rnn_train.py    #训练句向量
     ```
   - 训练版面分析
     ```
      python s2s_train.py

     ```
   - 推理
     ```
      python infer.py

     ```

