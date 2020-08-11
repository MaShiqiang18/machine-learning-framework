# machine-learning-framework
frame for machine-learning

## ������ܷ�Ϊ���ݷ���������Ԥ�����������̡���ģѵ�������Ρ�Ԥ�⡢����Ŀ

____
## ���ݷ���--analysis
### main:
#### run_analysis.py
������һ��analysis_main()��������dataAnalysis.py�д���ȫ�������������������ֵ�������������ֱ�ʵ�����������á���subProjects�п�ֱ�ӵ���run_analysis.py��Ҳ�ɲ����������¶��塣
                         
### untils:
#### dataAnalysis.py��������class�ֱ���ȫ�������������������ֵ����
    Analysis_all_features: 
        show_shape()--ͳ������������������
        percentage_miss()--ͳ������ֵ���ࡢȱʧֵռ�ȡ��������ֵռ�ȡ������������ͣ�
        show_corr_label_with_features()--�������������label��Ƥ��ѷ���ϵ����
                                                                    ��ѡ����show_pictureѡ���Ƿ������ͼ��
                                   
    Analysis_categorical_features: 
        show_features_distribution()--ͳ��ÿ����������ͬ����ֵ�������������Ʊ�ͼ��
        view_of_violinplot()--����С����ͼ���鿴�����ֲ���
        view_of_count_plot()--ͳ�����������ÿ�����Ƶ����
        trans_datatype()--�������������������ת��Ϊcategory��ͬʱ���NaN���ݣ�
                                           
    Analysis_numeric_features: show_corr_label_with_features()--����features��label�ģ�Ƥ��ѷ���ԣ���ضȣ�
                                                                ��ѡ����show_pictureѡ���Ƿ������ͼ��
                               show_Skew_and_kurt()--�����Ⱥ�ƫ�ȣ����������ֲ���
                               view_of_box()--��������ͼ�������쳣ֵ�ֲ���
                               view_of_bar_plot()--������״ͼ��
                               view_of_distplot()--������ֵ������label�ķֲ��������ʣ��ֲ�����&������ϣ�
                               view_of_pairplot()--���������໥֮��Ĺ�ϵ���ֲ������ӻ���
                                   
____        
## ����Ԥ����--preProcess
### main:
#### run_preprocess.py
������һ��preprocess_main()��������Ҫ����dataPreprocess.py��������ֵ���������ش��������ݣ�NaN/Inf�Ѵ�����������������ת������ʶ����Ĵ����쳣ֵ��index����subProjects�п�ֱ�ӵ���run_preprocess.py��Ҳ�ɲ����������¶��塣
                         
### untils:
#### dataPreprocess.py--��Ҫ�������ܣ��쳣ֵ����NaN/Inf������������ת��
    �쳣ֵ����outliers_proc()--��������ͼ���ķ�λ�ߣ����������ޣ�ʶ���쳣ֵ���������쳣ֵ��index��ԭ���ݲ��䣩��
                                 ��ѡ����comp��ȷ���Ƿ���ʾɾ���쳣ֵ�����������С����ֵ�����ݷֲ���������ֵ�����ݷֲ���
                                 ��ѡ����show_view��ȷ���Ƿ���ʾɾ���쳣ֵǰ�������ͼ���Ƚϲ��졣
                smooth_cols()--ֱ��ָ����ֵ�������ޣ�����������ֵ�����÷�λֵ���棬����ƽ�����������ݡ�
                         
    NaN/Inf����replace_var_obj()--�����л��п�ѡ�þ�ֵ����ֵ��������NaN�������
                 replace_var()--�Ƚ�Infת��ΪNaN��һ������ͨ����ȱʧ�ʡ��������ֵռ�ȡ��Ƿ������̬�ֲ����������жϣ�
                                ����replace_var_obj()���ֱ�ʹ�þ�ֵ����ֵ��������NaN������䡣����������ַ�������������
                                         
    ��ֵ����ת����reduce_mem_usage()--ָ��ʱ��������time_features����������͡�20120202��ת��Ϊdatetime�࣬�ٽ�������ֵ����������Ӧ����ת��
             
____             
## ��������--newFeatures
### main:
#### produceFeatures.py
������һ��featureProcess_main()������ͨ������NewFeatures.py�еķ���ʵ�ֶ�ʱ�������������������ֵ��������������ֱ𹹽�����������subProjects�п�ֱ�ӵ���run_preprocess.py��Ҳ�ɲ����������¶��塣һ�㽫ѵ�����Ͳ��Լ��ϲ������õ����������������²��Ϊѵ�����Ͳ��Լ���֮��ɵ���ƽ��ֵ�����Ŀ������һ������������
### untils:
#### NewFeatures.py--��ѵ�����Ͳ��Լ��ϲ�������ʱ�������������������ֵ��������������ֱ𹹽�������
    ʱ��������
            date_get_ymdw(): ��ʱ��������ȡ�ꡢ�¡��ܡ��յ���Ϣ
            date_features(): ����ָ������ʱ��������f1-f2�����ڲ�Լ���ǰ������f1��ʱ����ǰ������f2�����ڲ�
                 
    ��ֵ������
            produce_by_single(): ������������ȡ������{}ƽ������������������Ϣ
            produce_by_double(): ��������ϣ���ȡ������{}�����໥����Ӽ����������Ϣ
            cut_group(): ��ָ���������з�Ͱ��������ȡÿ��������Ӧ����λ���ĸ����䣬������������ɢ��
                  
    ���������
            count_coding(): ͳ�Ʋ�ͬ����ֵ������
            gen_onehot(): ��ָ����������onehot���룬��ѡ����deal_with_sp�Ƿ��������ַ�������������������
            add_onehot_features(): ������gen_onehot()�����onehot������ӵ�ԭ�����У���ɾ������ǰ��ԭ����
                  
    ����������
            produce_by_statistics(): �Ȱ�����������з��飬��ͳ����ֵ�������������ֵ����Сֵ����ֵ���ܺ͡���ֵ�����ƫ�ȡ���ȵ���Ϣ
            cross_cat_num(): �Ȱ�����������з��飬��ͳ����ֵ�������ֵ����Сֵ����ֵ����Ϣ
            cross_qua_cat_num(): ��������������ԣ�ͳ�ƹ��ִ�����n unique���ء�����ƫ�õ���Ϣ
         
#### CodeFeatures.py
ƽ�����룬��������ֵ�������漸�㣺1. ���ظ���2. ������ͬ��ֵ�����ֳ�����һ������������100������ģ��������
         
#### targetEncoding.py
Ŀ����룬��produce_by_statistics()�����ȷ����ͳ�ƣ���ͬ����Ŀ�������ѵ�����Ͳ��Լ��ֿ������������ģ��������ǿ��Ժϲ���һ�𹹽��ģ�
ԭ���ǣ�Ŀ���������ͳ�Ƶ���labelֵ�����Լ���û�У�����produce_by_statistics()��ͳ���������������ġ�
                            
#### SelectFeatures.py
ָ��Ϊ�ع����⻹�Ƿ������⣬��ʹ��Filter��Wrapper��Embedded�Լ�ͨ��������ֵ�����ģʽ��ѡ����õ�K������
    
    Filter model(������)-->����ʽѡ��[����������]
                  select_by_var(): �Ƴ���Щ�������ĳ����ֵ������
                  select_by_chi2(): ��������--��������
                  select_by_pearsonr(): Ƥ��ѷ���ϵ��--�ع�����
                  select_by_mic(): ����Ϣ�������Ϣϵ��
             
    Wrapper model(��װ��)-->����ʽѡ��
                  select_by_RFE(): ����������ù��̴�������������ʼ��ͨ����ɾ��������ʣ������������
                  select_by_SFS(): ǰ��ѡ�񣬸ù��̴�һ���յ����Լ��Ͽ�ʼ�������������������������С�
                                    ��ѡ����showFig���Ƿ�������������������ӵ÷ֱ仯����ͼ
                  select_by_RFECV(): չʾ���������������ӵ÷ֱ仯����ͼ
                  
    Embedded-->Ƕ��ʽѡ��
                  select_by_linearmodel(): ��������ģ��
                  select_by_nonlinearmodel(): ���ڷ����ԣ�����ģ��
                  
    ��������
                  selec_by_features_random_weight(): ��������������ģ�͵�Ӱ���С������������Ҫ��
                  selec_by_features_random_plt(): ���ƶ������ͳ����Ҫ�Եı仯������ͼ
                  
#### ScoreFeatures.py
ͬSelectFeatures.py��ԭ��ͳ�Ƴ�ÿ�ַ���������������Ҫ�ԣ�ÿһ�ַ���������������Ҫ�Ե÷ֺ�Ϊ1������������������з����õ��ĵ÷ֵľ�ֵ�����ֵ����ֵ�������Ƶ÷�����ͼ������һ���÷ֱ�
                           
____                           
## ��ģѵ��--trainModels
### main��
#### run_train.py
ѵ����ܣ�ʹ�ò�ͬ���������ѵ��runs�֣�ÿ�ֲ��Ϊfolds�ۣ�ÿ��ʹ��baggingѵ��bagging_size�Σ�ÿ�۵�Ԥ������bagging_size�εľ�ֵ������ѵ�����Ͳ��Լ���Ԥ�������Լ�ѵ�����÷�
                      
### untils��
#### getKfoldIndex.py
��ȡָ����ѵ�������ݼ�����С����runs��folds�۵�index���Թ�ѵ��ʱֱ�ӵ���
#### defindMetrics.py
���۷�������runs��folds��ѵ����ľ�ֵ����÷�
#### trainKerasModel.py
Keras������ģ�͵�ѵ����ܣ�models�µ�FCModelLib.py�е��ö���õ�������ṹ
#### trainTreeModel.py
��ģ�ͣ�LighGBM��Xgboost����ѵ�����
#### defind_log.py
�Զ�����־��¼��������Ҫ��¼ģ��ѵ���͵��ι����еĽ����Լ�����ʱ�䡢���׶εĵ÷�
#### time_tran.py
ʱ���ʽת��
#### showFigs.py
����ѵ��������Loss���ߺ͵÷�����
         
 ____        
## Ԥ��--predictByModels
### run_predict.py
��������ģ��Ԥ�⣨ͬʱ��ѵ�����Ͳ��Լ�����Ԥ�⣩������ѵ�����Ͳ��Լ���Ԥ�������Լ�ѵ�����÷�
         
____         
## ����--tuningParams
### main��
#### run_tuning.py
�����ƣ��ɵ���run_predict.py��ѵ�����
                      
### untils��
#### bayesianOpt_demo.py
ʹ�ñ�Ҷ˹���ε�demo
         
#### gridSearchCV_demo.py
�����������ε�demo
         
#### stepBystep_demo.py
�𲽵��ε�demo
         
 ____
         
## ����Ŀ��--subProjects
### config_demo.py--����Ŀconfigʱ�ɽ����demo�ṹ
    
### used_car--����Ŀ��
#### Code--����
    config.py--�����ļ�
        ��Ҫ�ɻ���Ϊ�����ݼ���ز�����ģ��ѵ����ز�����ѵ����ܽ�����֤��ز���������·����ز���
    dataAnalysis.py
    dataProcess.py
    bayesianOpt_runs_folds.py
    train.py
    predict.py
#### Data--����
    notes--����ѵ�������ι����м�¼�����ݣ����ձ������ͬ������������ݼ���ģ��ѵ�������浽��ͬ�����ļ�����
        mark1
        mark2
        demo
            Figs--ѵ�������б�������ͼƬ��Loss���ߣ�
            Log--ѵ�������в�������־�ļ�
            Models--ѵ���ÿɹ����õ�ģ��
            Predict--Ԥ����
            
    train_test_data--����ѵ��������
            ԭʼ����
            ����Ԥ����+�������̺������
            ��getKfoldIndex.py���ɵ��ļ�
            
                    
   