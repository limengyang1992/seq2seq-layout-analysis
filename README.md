
### 简介
    
当前OCR之后的版面分析工作大家都是都是规则写的，本人也深受规则之苦,看到ocr输出的一大堆文字和坐标就头皮发麻。最近受了chineseocr作者模板引擎的启发，做了个基于seq2seq的端到端版面分析算法，希望能够帮到各位ocrer。

链接：https://blog.csdn.net/mochp/article/details/109491521


### 数据标注
   
    使用labelme即可，在groupid里标注box的对应类别

### 修改配置项
   
    根据场景修改config.yaml文件

### 增广可视化
    
    运行 python plot/plot.py

### 模型训练
    
    运行 python train.py

### 模型测试
    
    运行 python test.py