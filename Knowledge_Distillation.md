# Knowledge Distillation  

为了测试知识蒸馏的可用性,基于CUB-200-2011[百度网盘链接](https://pan.baidu.com/s/12vcS_oCcKSagzvVGRFw-JQ?pwd=0g4w)数据集进行实验.  
### stduent为mobilenetv2,teacher为resnet50.

普通训练mobilenetv2：  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd

计算mobilenetv2指标:  
    
    python metrice.py --task val --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw --test_tta

普通训练resnet50:  

    python main.py --model_name resnet50 --config config/config.py --save_path runs/resnet50_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd

计算resnet50指标:
    
    python metrice.py --task val --save_path runs/resnet50_admaw
    python metrice.py --task test --save_path runs/resnet50_admaw
    python metrice.py --task test --save_path runs/resnet50_admaw --test_tta

知识蒸馏, resnet50作为teacher, mobilenetv2作为student, 使用SoftTarget进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_ST --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method SoftTarget --kd_ratio 0.7 --teacher_path runs/resnet50_admaw

知识蒸馏, resnet50作为teacher, mobilenetv2作为student, 使用MGD进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_MGD1 --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method MGD --kd_ratio 0.7 --teacher_path runs/resnet50_admaw

知识蒸馏, resnet50作为teacher, mobilenetv2作为student, 使用AT进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_AT --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method AT --kd_ratio 100 --teacher_path runs/resnet50_admaw 


计算通过resnet50蒸馏mobilenetv2指标:
    
    python metrice.py --task val --save_path runs/mobilenetv2_admaw_ST
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_ST
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_ST --test_tta
    python metrice.py --task val --save_path runs/mobilenetv2_admaw_MGD
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_MGD
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_MGD --test_tta
    python metrice.py --task val --save_path runs/mobilenetv2_admaw_AT
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_AT
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_AT --test_tta

| model | val accuracy | val mpa | test accuracy | test mpa | test accuracy(TTA) | test mpa(TTA) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| mobilenetv2 | 0.74116 | 0.74200 | 0.73483 | 0.73452 | 0.77012 | 0.76979 |
| resnet50 | 0.78720 | 0.78744 | 0.77744  | 0.77670 | 0.81231 | 0.81162 |
| teacher->resnet50<br>student->mobilenetv2<br>SoftTarget | 0.77092 | 0.77179 | 0.75248 | 0.75191 | 0.77787 | 0.77752 |
| teacher->resnet50<br>student->mobilenetv2<br>MGD | 0.78888 | 0.78994 | 0.78390 | 0.78296 | 0.79940 | 0.79890 |
| teacher->resnet50<br>student->mobilenetv2<br>AT | 0.74789 | 0.74878 | 0.73870 | 0.73795 | 0.76324 | 0.76244 |

### stduent为mobilenetv2,teacher为ghostnet.

普通训练mobilenetv2：  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd

计算mobilenetv2指标:  
    
    python metrice.py --task val --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw --test_tta

普通训练ghostnet：  

    python main.py --model_name ghostnet --config config/config.py --save_path runs/ghostnet_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd

计算ghostnet指标:  
    
    python metrice.py --task val --save_path runs/ghostnet_admaw
    python metrice.py --task test --save_path runs/ghostnet_admaw
    python metrice.py --task test --save_path runs/ghostnetadmaw --test_tta

知识蒸馏, ghostnet作为teacher, mobilenetv2作为student, 使用SoftTarget进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_ST --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method SoftTarget --kd_ratio 0.7 --teacher_path runs/ghostnet_admaw

知识蒸馏, ghostnet作为teacher, mobilenetv2作为student, 使用MGD进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_MGD --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method MGD --kd_ratio 0.2 --teacher_path runs/ghostnet_admaw

知识蒸馏, ghostnet作为teacher, mobilenetv2作为student, 使用AT进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_AT --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method AT --kd_ratio 1000.0 --teacher_path runs/ghostnet_admaw

计算通过ghostnet蒸馏mobilenetv2指标:
    
    python metrice.py --task val --save_path runs/mobilenetv2_admaw_ST
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_ST
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_ST --test_tta
    python metrice.py --task val --save_path runs/mobilenetv2_admaw_MGD
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_MGD
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_MGD --test_tta
    python metrice.py --task val --save_path runs/mobilenetv2_admaw_AT
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_AT
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_AT --test_tta

| model | val accuracy | val mpa | test accuracy | test mpa | test accuracy(TTA) | test mpa(TTA) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| mobilenetv2 | 0.74116 | 0.74200 | 0.73483 | 0.73452 | 0.77012 | 0.76979 |
| ghostnet | 0.77709 | 0.77756 | 0.76367  | 0.76277 | 0.78046 | 0.77958 |
| teacher->ghostnet<br>student->mobilenetv2<br>SoftTarget | 0.77878 | 0.77968 | 0.76108 | 0.76022 | 0.77916 | 0.77807 |
| teacher->ghostnet<br>student->mobilenetv2<br>MGD | 0.75632 | 0.75723 | 0.74688 | 0.74638 | 0.77357 | 0.77302 |
| teacher->ghostnet<br>student->mobilenetv2<br>AT | 0.74846 | 0.74945 | 0.73827 | 0.73782 | 0.76625 | 0.76534 |

### 由于SP蒸馏开启AMP时,kd_loss大概率会出现nan,所在SP蒸馏实验中,我们把所有模型都不开启AMP.

#### stduent为mobilenetv2,teacher为ghostnet.  

普通训练mobilenetv2：  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --warmup --imagenet_meanstd

计算mobilenetv2指标:  
    
    python metrice.py --task val --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw --test_tta

普通训练ghostnet：  

    python main.py --model_name ghostnet --config config/config.py --save_path runs/ghostnet_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --warmup --imagenet_meanstd

计算ghostnet指标:  
    
    python metrice.py --task val --save_path runs/ghostnet_admaw
    python metrice.py --task test --save_path runs/ghostnet_admaw
    python metrice.py --task test --save_path runs/ghostnetadmaw --test_tta

知识蒸馏, ghostnet作为teacher, mobilenetv2作为student, 使用SP进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_SP --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --warmup --imagenet_meanstd \
    --kd --kd_method SP --kd_ratio 10.0 --teacher_path runs/ghostnet_admaw

计算通过ghostnet蒸馏mobilenetv2指标:

    python metrice.py --task val --save_path runs/mobilenetv2_admaw_SP
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_SP
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_SP --test_tta

| model | val accuracy | val mpa | test accuracy | test mpa | test accuracy(TTA) | test mpa(TTA) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| mobilenetv2 | 0.74509 | 0.74568 | 0.73827 | 0.73761 | 0.76969 | 0.76903 |
| ghostnet | 0.77821 | 0.77881 | 0.75807  | 0.75708 | 0.77873 | 0.77805 |
| teacher->ghostnet<br>student->mobilenetv2<br>SP | 0.74733 | 0.74836 | 0.73267 | 0.73198 | 0.75893 | 0.75850 |

#### stduent为mobilenetv2,teacher为resnet50.  

普通训练mobilenetv2：  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --warmup --imagenet_meanstd

计算mobilenetv2指标:  
    
    python metrice.py --task val --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw
    python metrice.py --task test --save_path runs/mobilenetv2_admaw --test_tta

普通训练resnet50：  

    python main.py --model_name resnet50 --config config/config.py --save_path runs/resnet50_admaw --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --warmup --imagenet_meanstd

计算resnet50指标:  
    
    python metrice.py --task val --save_path runs/resnet50_admaw
    python metrice.py --task test --save_path runs/resnet50_admaw
    python metrice.py --task test --save_path runs/resnet50_admaw --test_tta

知识蒸馏, resnet50作为teacher, mobilenetv2作为student, 使用SP进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_SP --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --warmup --imagenet_meanstd \
    --kd --kd_method SP --kd_ratio 10.0 --teacher_path runs/resnet50_admaw

| model | val accuracy | val mpa | test accuracy | test mpa | test accuracy(TTA) | test mpa(TTA) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| mobilenetv2 | 0.74509 | 0.74568 | 0.73827 | 0.73761 | 0.76969 | 0.76903 |
| resnet50 | 0.78720 | 0.78707 | 0.77400  | 0.77321 | 0.81231 | 0.81138 |
| teacher->resnet50<br>student->mobilenetv2<br>SP | 0.74116 | 0.74200 | 0.74042 | 0.73969 | 0.76840 | 0.76753 |

### 以下实验是通过训练好的自身模型再作为教师模型进行训练.

知识蒸馏, resnet50作为teacher, resnet50作为student, 使用AT进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/resnet50_admaw_AT_self --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method AT --kd_ratio 100 --teacher_path runs/resnet50_admaw 

计算通过resnet50蒸馏resnet50指标:  

    python metrice.py --task val --save_path runs/resnet50_admaw_AT_self
    python metrice.py --task test --save_path runs/resnet50_admaw_AT_self
    python metrice.py --task test --save_path runs/resnet50_admaw_AT_self --test_tta

知识蒸馏, mobilenetv2作为teacher, mobilenetv2作为student, 使用AT进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_admaw_AT_self --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method AT --kd_ratio 100 --teacher_path runs/mobilenetv2_admaw 

计算通过mobilenetv2蒸馏mobilenetv2指标:  

    python metrice.py --task val --save_path runs/mobilenetv2_admaw_AT_self
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_AT_self
    python metrice.py --task test --save_path runs/mobilenetv2_admaw_AT_self --test_tta


知识蒸馏, ghostnet作为teacher, ghostnet作为student, 使用AT进行蒸馏:  

    python main.py --model_name ghostnet --config config/config.py --save_path runs/ghostnet_admaw_AT_self --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method AT --kd_ratio 1000 --teacher_path runs/ghostnet_admaw 

计算通过ghostnet蒸馏ghostnet指标:  

    python metrice.py --task val --save_path runs/ghostnet_admaw_AT_self
    python metrice.py --task test --save_path runs/ghostnet_admaw_AT_self
    python metrice.py --task test --save_path runs/ghostnet_admaw_AT_self --test_tta

| model | val accuracy | val mpa | test accuracy | test mpa | test accuracy(TTA) | test mpa(TTA) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| mobilenetv2 | 0.74116 | 0.74200 | 0.73483 | 0.73452 | 0.77012 | 0.76979 |
| teacher->mobilenetv2<br>student->mobilenetv2<br>AT | 0.74677 | 0.74758 | 0.74430 | 0.74342 | 0.77012 | 0.76926 |
| resnet50 | 0.78720 | 0.78744 | 0.77744  | 0.77670 | 0.81231 | 0.81162 |
| teacher->resnet50<br>student->resnet50<br>AT | 0.79057 | 0.79091 | 0.79165 | 0.79026 | 0.81102 | 0.81030 |
| ghostnet | 0.77709 | 0.77756 | 0.76367  | 0.76277 | 0.78046 | 0.77958 |
| teacher->ghostnet<br>student->ghostnet<br>AT | 0.78046 | 0.78080 | 0.77142 | 0.77069 | 0.78820 | 0.78742 |

### 在V1.1版本的测试中发现efficientnet_v2网络作为teacher网络效果还不错.

普通训练mobilenetv2：  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2 --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd

计算mobilenetv2指标:  
    
    python metrice.py --task val --save_path runs/mobilenetv2
    python metrice.py --task test --save_path runs/mobilenetv2
    python metrice.py --task test --save_path runs/mobilenetv2 --test_tta

普通训练efficientnet_v2_s：  

    python main.py --model_name efficientnet_v2_s --config config/config.py --save_path runs/efficientnet_v2_s --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd

计算efficientnet_v2_s指标:  
    
    python metrice.py --task val --save_path runs/efficientnet_v2_s
    python metrice.py --task test --save_path runs/efficientnet_v2_s
    python metrice.py --task test --save_path runs/efficientnet_v2_s --test_tta

知识蒸馏, efficientnet_v2_s作为teacher, mobilenetv2作为student, 使用SoftTarget进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_ST --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method SoftTarget --kd_ratio 0.7 --teacher_path runs/efficientnet_v2_s

知识蒸馏, efficientnet_v2_s作为teacher, mobilenetv2作为student, 使用MGD进行蒸馏:  

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_MGD --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd \
    --kd --kd_method MGD --kd_ratio 0.7 --teacher_path runs/efficientnet_v2_s

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_MGD_EMA --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd --ema \
    --kd --kd_method MGD --kd_ratio 0.7 --teacher_path runs/efficientnet_v2_s

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_MGD_RDROP --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd --rdrop \
    --kd --kd_method MGD --kd_ratio 0.7 --teacher_path runs/efficientnet_v2_s

    python main.py --model_name mobilenetv2 --config config/config.py --save_path runs/mobilenetv2_MGD_EMA_RDROP --lr 1e-4 --Augment AutoAugment --epoch 150 \
    --pretrained --amp --warmup --imagenet_meanstd --rdrop --ema \
    --kd --kd_method MGD --kd_ratio 0.7 --teacher_path runs/efficientnet_v2_s

计算通过efficientnet_v2_s蒸馏mobilenetv2指标:  

    python metrice.py --task val --save_path runs/mobilenetv2_ST
    python metrice.py --task test --save_path runs/mobilenetv2_ST
    python metrice.py --task test --save_path runs/mobilenetv2_ST --test_tta

    python metrice.py --task val --save_path runs/mobilenetv2_MGD
    python metrice.py --task test --save_path runs/mobilenetv2_MGD
    python metrice.py --task test --save_path runs/mobilenetv2_MGD --test_tta

    python metrice.py --task val --save_path runs/mobilenetv2_MGD_EMA
    python metrice.py --task test --save_path runs/mobilenetv2_MGD_EMA
    python metrice.py --task test --save_path runs/mobilenetv2_MGD_EMA --test_tta

    python metrice.py --task val --save_path runs/mobilenetv2_MGD_RDROP
    python metrice.py --task test --save_path runs/mobilenetv2_MGD_RDROP
    python metrice.py --task test --save_path runs/mobilenetv2_MGD_RDROP --test_tta

    python metrice.py --task val --save_path runs/mobilenetv2_MGD_EMA_RDROP
    python metrice.py --task test --save_path runs/mobilenetv2_MGD_EMA_RDROP
    python metrice.py --task test --save_path runs/mobilenetv2_MGD_EMA_RDROP --test_tta

| model | val accuracy | val mpa | test accuracy | test mpa | test accuracy(TTA) | test mpa(TTA) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| mobilenetv2 | 0.74116 | 0.74200 | 0.73483 | 0.73452 | 0.77012 | 0.76979 |
| efficientnet_v2_s | 0.84166 | 0.84191 | 0.84460 | 0.84441 | 0.86483 | 0.86484 |
| teacher->efficientnet_v2_s<br>student->mobilenetv2<br>ST | 0.76137 | 0.76209 | 0.75161 | 0.75088 | 0.77830 | 0.77715 |
| teacher->efficientnet_v2_s<br>student->mobilenetv2<br>MGD | 0.77204 | 0.77288 | 0.77529 | 0.77464 | 0.79337 | 0.79261 |
| teacher->efficientnet_v2_s<br>student->mobilenetv2<br>MGD(EMA) | 0.77204 | 0.77267 | 0.77744 | 0.77671 | 0.80284 | 0.80201 |
| teacher->efficientnet_v2_s<br>student->mobilenetv2<br>MGD(RDrop) | 0.77204 | 0.77288 | 0.77529 | 0.77464 | 0.79337 | 0.79261 |
| teacher->efficientnet_v2_s<br>student->mobilenetv2<br>MGD(EMA,RDrop) | 0.77204 | 0.77267 | 0.77744 | 0.77671 | 0.80284 | 0.80201 |

## 关于Knowledge Distillation的一些解释

实验解释:  
1. 对于AT和SP蒸馏方法,上述实验都是使用block3和block4的特征层进行蒸馏.  
2. MPA是平均类别精度,在类别不平衡的情况下非常有用,当类别基本平衡的情况下,跟accuracy差不多.  
3. 当蒸馏loss出现nan的时候请不要开启AMP,AMP可能会导致浮点溢出导致的nan.  

目前支持的类型有:
| Name | Method | paper |  
| :----: | :----: | :----: |
| SoftTarget | logits | https://arxiv.org/pdf/1503.02531.pdf |
| MGD | features | https://arxiv.org/abs/2205.01529.pdf |
| SP | features | https://arxiv.org/pdf/1907.09682.pdf |
| AT | features | https://arxiv.org/pdf/1612.03928.pdf |

蒸馏学习跟模型,参数,蒸馏的方法,蒸馏的层都有关系,效果不好需要自行调整,其中SP和AT都可以对模型中的四个block进行组合计算蒸馏损失具体代码在utils/utils_fit.py的fitting_distill函数中可以进行修改.