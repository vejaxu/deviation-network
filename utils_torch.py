import pandas as pd


from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.datasets import load_svmlight_file


def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def dataLoading(path):
    # loading data
    df = pd.read_csv(path) 
    
    labels = df['class']
    
    x_df = df.drop(['class'], axis=1)
    
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    
    return x, labels


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap, train_time, test_time, path = "./results/auc_performance_cl0.5.csv"):    
    csv_file = open(path, 'a') 
    row = name + "," + str(n_samples)+ ","  + str(dim) + ',' + str(n_samples_trn) + ','+ str(n_outliers_trn) + ','+ str(n_outliers)  + ',' + str(depth)+ "," + str(rauc) +"," + str(std_auc) + "," + str(ap) +"," + str(std_ap)+"," + str(train_time)+"," + str(test_time) + "\n"
    csv_file.write(row)