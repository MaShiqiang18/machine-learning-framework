# coding=gbk
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from sklearn.metrics import accuracy_score, auc, average_precision_score, brier_score_loss, confusion_matrix, f1_score,\
    log_loss, precision_score, recall_score, roc_auc_score, roc_curve

class MyMetrics(object):
    def __init__(self, name):
        self.name = name

    def metricsFunc(self, pre_v, real_v):
        if self.name == 'MAE':
            return mean_absolute_error(pre_v, real_v)
        elif self.name == 'MSE':
            return mean_squared_error(pre_v, real_v)
        elif self.name == 'MedAE':
            return median_absolute_error(pre_v, real_v)
        elif self.name == 'EVC':
            return explained_variance_score(pre_v, real_v)
        elif self.name == 'R2':
            return r2_score(pre_v, real_v)

        #################################################
        elif self.name == 'ACC':
            return accuracy_score(pre_v, real_v)
        elif self.name == 'AUC':
            return auc(pre_v, real_v)
        elif self.name == 'APS':
            return average_precision_score(pre_v, real_v)
        elif self.name == 'BSL':
            return brier_score_loss(pre_v, real_v)
        elif self.name == 'CM':
            return confusion_matrix(pre_v, real_v)
        elif self.name == 'LS':
            return log_loss(pre_v, real_v)
        elif self.name == 'F1':
            return f1_score(pre_v, real_v)
        elif self.name == 'PS':
            return precision_score(pre_v, real_v)
        elif self.name == 'RS':
            return recall_score(pre_v, real_v)
        elif self.name == 'RAS':
            return roc_auc_score(pre_v, real_v)
        elif self.name == 'RC':
            return roc_curve(pre_v, real_v)
        else:
            print('请指定评价函数名称')
            return None