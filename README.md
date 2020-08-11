# machine-learning-framework
frame for machine-learning

## 整个框架分为数据分析、数据预处理、特征工程、建模训练、调参、预测、子项目

____
## 数据分析--analysis
### main:
#### run_analysis.py
定义了一个analysis_main()方法，将dataAnalysis.py中处理全部特征、类别特征、数值特征的三个类别分别实例化，并调用。在subProjects中可直接调用run_analysis.py，也可参照其再重新定义。
                         
### untils:
#### dataAnalysis.py定义三个class分别处理全部特征、类别特征、数值特征
    Analysis_all_features: 
        show_shape()--统计样本数，特征数；
        percentage_miss()--统计特征值种类、缺失值占比、最大特征值占比、特征数据类型；
        show_corr_label_with_features()--计算各个特征与label的皮尔逊相关系数，
                                                                    可选参数show_picture选择是否绘制热图；
                                   
    Analysis_categorical_features: 
        show_features_distribution()--统计每个特征，不同特征值的数量，并绘制饼图；
        view_of_violinplot()--绘制小提琴图，查看特征分布；
        view_of_count_plot()--统计类别特征的每个类别频数；
        trans_datatype()--将类别特征的数据类型转化为category，同时填充NaN数据；
                                           
    Analysis_numeric_features: show_corr_label_with_features()--计算features与label的（皮尔逊线性）相关度，
                                                                可选参数show_picture选择是否绘制热图；
                               show_Skew_and_kurt()--计算峰度和偏度，分析特征分布；
                               view_of_box()--绘制箱型图，分析异常值分布；
                               view_of_bar_plot()--绘制柱状图；
                               view_of_distplot()--绘制数值特征和label的分布（含概率）分布曲线&线性拟合；
                               view_of_pairplot()--特征两两相互之间的关系（分布）可视化；
                                   
____        
## 数据预处理--preProcess
### main:
#### run_preprocess.py
定义了一个preprocess_main()方法，主要调用dataPreprocess.py来处理数值特征，返回处理后的数据（NaN/Inf已处理，并进行数据类型转化）、识别出的带有异常值的index。在subProjects中可直接调用run_preprocess.py，也可参照其再重新定义。
                         
### untils:
#### dataPreprocess.py--主要三个功能：异常值处理、NaN/Inf处理、数据类型转化
    异常值处理：outliers_proc()--利用箱型图（四分位线）设立上下限，识别异常值，并返回异常值的index（原数据不变），
                                 可选参数comp，确定是否显示删除异常值后的样本数、小于阈值的数据分布、大于阈值的数据分布，
                                 可选参数show_view，确定是否显示删除异常值前后的箱型图，比较差异。
                smooth_cols()--直接指定阈值（上下限）、将超出阈值部分用分位值代替，返回平滑处理后的数据。
                         
    NaN/Inf处理：replace_var_obj()--按照行或列可选用均值、中值、众数对NaN进行填充
                 replace_var()--先将Inf转化为NaN，一并处理，通过对缺失率、最大特征值占比、是否符合正态分布等条件的判断，
                                调用replace_var_obj()，分别使用均值、中值、众数对NaN进行填充。类别特征（字符串）单独处理。
                                         
    数值类型转化：reduce_mem_usage()--指定时间特征列time_features，将其从整型‘20120202’转化为datetime类，再将其他数值特征进行相应类型转化
             
____             
## 特征工程--newFeatures
### main:
#### produceFeatures.py
定义了一个featureProcess_main()方法，通过调用NewFeatures.py中的方法实现对时间特征、类别特征、数值特征、组合特征分别构建新特征，在subProjects中可直接调用run_preprocess.py，也可参照其再重新定义。一般将训练集和测试集合并后处理，得到所有新特征后，重新拆分为训练集和测试集，之后可调用平均值编码和目标编码进一步构建新特征
### untils:
#### NewFeatures.py--将训练集和测试集合并后处理，对时间特征、类别特征、数值特征、组合特征分别构建新特征
    时间特征：
            date_get_ymdw(): 从时间特征提取年、月、周、日等信息
            date_features(): 计算指定两个时间特征：f1-f2的日期差，以及当前日期与f1的时间差，当前日期与f2的日期差
                 
    数值特征：
            produce_by_single(): 单特征处理，提取特征：{}平方、立方、对数等信息
            produce_by_double(): 多特征组合，提取特征：{}两两相互交叉加减乘数后的信息
            cut_group(): 对指定特征进行分桶操作，提取每个样本对应特征位于哪个区间，即连续特征离散化
                  
    类别特征：
            count_coding(): 统计不同特征值的数量
            gen_onehot(): 对指定特征进行onehot编码，可选参数deal_with_sp是否处理特征字符，对新特征重新命名
            add_onehot_features(): 将调用gen_onehot()构造的onehot编码添加到原数据中，并删除编码前的原特征
                  
    交叉特征：
            produce_by_statistics(): 先按类别特征进行分组，在统计数值特征总数、最大值、最小值、中值、总和、均值、方差、偏度、峰度等信息
            cross_cat_num(): 先按类别特征进行分组，在统计数值特征最大值、最小值、中值等信息
            cross_qua_cat_num(): 给定（类别）特征对，统计共现次数、n unique、熵、比例偏好等信息
         
#### CodeFeatures.py
平均编码，处理特征值满足下面几点：1. 会重复，2. 根据相同的值分组会分出超过一定数量（比如100）的组的（类别）特征
         
#### targetEncoding.py
目标编码，与produce_by_statistics()类似先分组后统计，不同的是目标编码是训练集和测试集分开构建新特征的，而后者是可以合并后一起构建的，
原因是，目标编码分组后统计的是label值（测试集中没有），而produce_by_statistics()是统计其他连续特征的。
                            
#### SelectFeatures.py
指定为回归问题还是分类问题，再使用Filter、Wrapper、Embedded以及通过将特征值乱序等模式，选出最好的K个特征
    
    Filter model(过滤器)-->过滤式选择[单变量分析]
                  select_by_var(): 移除那些方差低于某个阈值的特征
                  select_by_chi2(): 卡方检验--分类问题
                  select_by_pearsonr(): 皮尔逊相关系数--回归问题
                  select_by_mic(): 互信息和最大信息系数
             
    Wrapper model(封装器)-->包裹式选择
                  select_by_RFE(): 向后消除，该过程从所有特征集开始。通过逐步删除集合中剩余的最差特征。
                  select_by_SFS(): 前向选择，该过程从一个空的特性集合开始，并逐个添加最优特征到集合中。
                                    可选参数showFig，是否绘制随着特征个数增加得分变化趋势图
                  select_by_RFECV(): 展示随着特征个数增加得分变化趋势图
                  
    Embedded-->嵌入式选择
                  select_by_linearmodel(): 基于线性模型
                  select_by_nonlinearmodel(): 基于非线性（树）模型
                  
    特征乱序
                  selec_by_features_random_weight(): 根据特征乱序后对模型的影响大小评价特征的重要性
                  selec_by_features_random_plt(): 绘制多次乱序统计重要性的变化的箱型图
                  
#### ScoreFeatures.py
同SelectFeatures.py的原理，统计出每种方法所有特征的重要性（每一种方法所有特征的重要性得分和为1），求出各个特征所有方法得到的得分的均值、最大值、中值，并绘制得分折线图，返回一个得分表
                           
____                           
## 建模训练--trainModels
### main：
#### run_train.py
训练框架，使用不同的随机种子训练runs轮，每轮拆分为folds折，每折使用bagging训练bagging_size次，每折的预测结果是bagging_size次的均值，返回训练集和测试集的预测结果，以及训练集得分
                      
### untils：
#### getKfoldIndex.py
获取指定（训练）数据集（大小）的runs轮folds折的index，以供训练时直接调用
#### defindMetrics.py
评价方法，在runs轮folds折训练后的均值计算得分
#### trainKerasModel.py
Keras神经网络模型的训练框架，models下的FCModelLib.py中调用定义好的神经网络结构
#### trainTreeModel.py
树模型（LighGBM、Xgboost）的训练框架
#### defind_log.py
自定义日志记录方法，主要记录模型训练和调参过程中的进度以及花费时间、各阶段的得分
#### time_tran.py
时间格式转化
#### showFigs.py
绘制训练过程中Loss曲线和得分曲线
         
 ____        
## 预测--predictByModels
### run_predict.py
调用已有模型预测（同时对训练集和测试集进行预测），返回训练集和测试集的预测结果，以及训练集得分
         
____         
## 调参--tuningParams
### main：
#### run_tuning.py
待完善，可调用run_predict.py的训练框架
                      
### untils：
#### bayesianOpt_demo.py
使用贝叶斯调参的demo
         
#### gridSearchCV_demo.py
网格搜索调参的demo
         
#### stepBystep_demo.py
逐步调参的demo
         
 ____
         
## 子项目集--subProjects
### config_demo.py--子项目config时可借鉴的demo结构
    
### used_car--子项目名
#### Code--代码
    config.py--配置文件
        主要可划分为：数据集相关参数、模型训练相关参数、训练框架交叉验证相关参数、数据路径相关参数
    dataAnalysis.py
    dataProcess.py
    bayesianOpt_runs_folds.py
    train.py
    predict.py
#### Data--数据
    notes--保存训练、调参过程中记录的数据，按照标记区别不同方法处理的数据集和模型训练，保存到不同的子文件夹下
        mark1
        mark2
        demo
            Figs--训练过程中保存的相关图片（Loss曲线）
            Log--训练过程中产生的日志文件
            Models--训练好可供调用的模型
            Predict--预测结果
            
    train_test_data--用于训练的数据
            原始数据
            数据预处理+特征工程后的数据
            由getKfoldIndex.py生成的文件
            
                    
   