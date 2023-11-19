#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import precision_recall_fscore_support
import multiprocessing
import warnings

from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore")

def imputation1(df1,df2):
    df1c=df1.copy()
    df2c=df2.copy()
    #takes train and testing data and imputes values in both of them.
    #imputes with mean
    for col in df1c.columns:
        if df1c[col].isnull().any():
            mean_value = df1c[col].mean()
            df1c[col].fillna(mean_value, inplace=True)

    for col in df2c.columns:
        if df2c[col].isnull().any():
            mean_value = df2c[col].mean()
            df2c[col].fillna(mean_value, inplace=True)

    return df1c, df2c

def imputation2(df1,df2):
    df1c=df1.copy()
    df2c=df2.copy()
    #takes train and testing data and imputes values in both of them.
    #imputes with mean
    for col in df1c.columns:
        if df1c[col].isnull().any():
            median_value = df1c[col].median()
            df1c[col].fillna(median_value, inplace=True)

    for col in df2c.columns:
        if df2c[col].isnull().any():
            median_value = df2c[col].median()
            df2c[col].fillna(median_value, inplace=True)

    return df1c, df2c

def imputation3(df1,df2):
    df1c=df1.copy()
    df2c=df2.copy()
    #takes train and testing data and imputes values in both of them.
    #imputes with mean
    for col in df1c.columns:
        if df1c[col].isnull().any():
            mode_value = df1c[col].mode().iloc[0]
            df1c[col].fillna(mode_value, inplace=True)

    for col in df2c.columns:
        if df2c[col].isnull().any():
            mode_value = df2c[col].mode().iloc[0]
            df2c[col].fillna(mode_value, inplace=True)

    return df1c, df2c
    
def imputation4(df1,df2):
    df1c=df1.copy()
    df2c=df2.copy()
    #delete columns with more than 70% missing values.
    
    columns_to_drop = df1c.columns[df1c.isnull().mean() > 0.7]
    df1c = df1c.drop(columns=columns_to_drop)
    df2c = df2c.drop(columns=columns_to_drop)

    #now impute remaining missing values
    return imputation1(df1c,df2c)
    
def imputation5(df1,df2):
    df1c=df1.copy()
    df2c=df2.copy()
    #delete columns with more than 70% missing values.
    
    columns_to_drop = df1c.columns[df1c.isnull().mean() > 0.7]
    df1c = df1c.drop(columns=columns_to_drop)
    df2c = df2c.drop(columns=columns_to_drop)

    #now impute remaining missing values
    return imputation2(df1c,df2c)

def imputation6(df1,df2):
    df1c=df1.copy()
    df2c=df2.copy()
    #delete columns with more than 70% missing values.
    
    columns_to_drop = df1c.columns[df1c.isnull().mean() > 0.7]
    df1c = df1c.drop(columns=columns_to_drop)
    df2c = df2c.drop(columns=columns_to_drop)

    #now impute remaining missing values
    return imputation3(df1c,df2c)
    

def ipv4_to_number(ipv4_address):
    octets = ipv4_address.rstrip('.').split('.')
    if len(octets) != 4:
        raise ValueError("Invalid IPv4 address format")

    numerical_value = 0
    for i, octet in enumerate(octets):
        numerical_value += int(octet) * (256 ** (3 - i))
    
    return numerical_value

def numericalize(df):
    #should be done for all original dataset
    #delete column Unnamed
    #convert boolean to 1 and 0
    #keep empty cells empty
    #convert ip address to numbers
    df.drop('Unnamed: 0.1', axis=1, inplace=True)
    
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    # print("Columns with non-numeric data:")
    # for col in non_numeric_columns:
    #     data_type = df[col].dtype
    #     print(f'{data_type},{col}')
    # print('\n')

    for col in non_numeric_columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(float).fillna(df[col])

    for col in ['DST_IP','SRC_IP']:
        df[col] = df[col].apply(ipv4_to_number)
        
    return df



def train_knn(dataset,i):
    X=dataset[0]
    y=TRAIN_LABELS
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   
    knn_model = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(knn_model, param_grid, cv=5,n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split.values.ravel())

    best_params = grid_search.best_params_

    y_pred = grid_search.predict(X_val.values)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')

    new_data = {'Imputation_Strategy': i, 'Model': 'KNN', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')

def train_dt(dataset,i):
    X=dataset[0]
    y=TRAIN_LABELS
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   
    dt_model = DecisionTreeClassifier()

    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None,10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4]
    }

    grid_search = GridSearchCV(dt_model, param_grid, cv=5,n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split.values.ravel())
    best_params = grid_search.best_params_

    y_pred = grid_search.predict(X_val.values)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')

    new_data = {'Imputation_Strategy': i, 'Model': 'DT', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')

def train_rf(dataset,i):
    X=dataset[0]
    y=TRAIN_LABELS
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   
    rf_model = RandomForestClassifier()

    param_grid = {
        'n_estimators': [200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 2, 10],
    }

    grid_search = GridSearchCV(rf_model, param_grid, cv=5,n_jobs=-1)

    grid_search.fit(X_train_split, y_train_split.values.ravel())

    best_params = grid_search.best_params_

    y_pred = grid_search.predict(X_val.values)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')

    new_data = {'Imputation_Strategy': i, 'Model': 'RF', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')

def train_lr(dataset,i):
    X=dataset[0]
    y=TRAIN_LABELS
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
   
    lr_model = LogisticRegression()

    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [1, 10, 100],
        'solver': ['newton-cg', 'lbfgs'],
    }

    grid_search = GridSearchCV(lr_model, param_grid, cv=5,n_jobs=-1)

    grid_search.fit(X_train_split, y_train_split.values.ravel())

    best_params = grid_search.best_params_

    y_pred = grid_search.predict(X_val.values)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')

    new_data = {'Imputation_Strategy': i, 'Model': 'LR', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')




def train_knn2(dataset,i):
    
    df_copy = dataset.copy()

    X = df_copy[0]
    y = TRAIN_LABELS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    mutual_info_selector = SelectPercentile(mutual_info_classif, percentile=30)
    X_train_mutual_info_selected = mutual_info_selector.fit_transform(X_train, y_train)
    X_test_mutual_info_selected = mutual_info_selector.transform(X_test)

    knn_classifier = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    print('GS started')
    warnings.filterwarnings("ignore")
    grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid,cv=5, n_jobs=-1)
    #grid_search.fit(X_train_scaled, y_train_balanced.values.ravel())
    grid_search.fit(X_train_mutual_info_selected, y_train.values.ravel())
    print('GS fitted')
    best_knn_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    #y_pred = best_knn_model.predict(X_test_mutual_info_selected.values)
    y_pred = best_knn_model.predict(X_test_mutual_info_selected)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1 )

    new_data = {'Imputation_Strategy': i, 'Model': 'KNN', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')


def train_dt2(dataset,i):
    
    df_copy = dataset.copy()

    X = df_copy[0]
    y = TRAIN_LABELS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    mutual_info_selector = SelectPercentile(mutual_info_classif, percentile=30)
    X_train_mutual_info_selected = mutual_info_selector.fit_transform(X_train, y_train)
    X_test_mutual_info_selected = mutual_info_selector.transform(X_test)

    dt_classifier = DecisionTreeClassifier()

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring='precision_macro', cv=5, n_jobs=-1)
    grid_search.fit(X_train_mutual_info_selected, y_train.values.ravel())

    best_dt_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_dt_model.predict(X_test_mutual_info_selected)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1)

    new_data = {'Imputation_Strategy': i, 'Model': 'DT', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')

def train_rf2(dataset,i):
    
    df_copy = dataset.copy()

    X = df_copy[0]
    y = TRAIN_LABELS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    mutual_info_selector = SelectPercentile(mutual_info_classif, percentile=30)
    X_train_mutual_info_selected = mutual_info_selector.fit_transform(X_train, y_train)
    X_test_mutual_info_selected = mutual_info_selector.transform(X_test)

   
    rf_classifier = RandomForestClassifier()

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4]
    }

    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='precision_macro', cv=5, n_jobs=-1)
    grid_search.fit(X_train_mutual_info_selected, y_train.values.ravel())

    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_rf_model.predict(X_test_mutual_info_selected)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1)

    new_data = {'Imputation_Strategy': i, 'Model': 'RF', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')


def train_lr2(dataset,i):
    
    df_copy = dataset.copy()

    X = df_copy[0]
    y = TRAIN_LABELS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    mutual_info_selector = SelectPercentile(mutual_info_classif, percentile=30)
    X_train_mutual_info_selected = mutual_info_selector.fit_transform(X_train, y_train)
    X_test_mutual_info_selected = mutual_info_selector.transform(X_test)


    lr_classifier = LogisticRegression()

    param_grid = {
        'penalty': ['l1', 'l2','none'],
        'C': [1, 10, 100],
        'solver': ['newton-cg', 'lbfgs'],
        'max_iter': [800, 1000]
    }

    grid_search = GridSearchCV(estimator=lr_classifier, param_grid=param_grid, scoring='precision_macro', cv=5, n_jobs=-1)
    grid_search.fit(X_train_mutual_info_selected, y_train.values.ravel())

    best_lr_model = grid_search.best_estimator_

    y_pred = best_lr_model.predict(X_test_mutual_info_selected)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1)
    
    best_params = grid_search.best_params_
    new_data = {'Imputation_Strategy': i, 'Model': 'LR', 'Best_Params': best_params, 'Precision':precision, 'Recall':recall, 'F_Score':fscore}
    new_row = pd.DataFrame([new_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(new_row)
    print('-----------------------------------------------------------------------------------------')


def output(dataset):
    df_copy = dataset.copy()

    X = df_copy[0]
    unknown=df_copy[1]
    
    y = TRAIN_LABELS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    mutual_info_selector = SelectPercentile(mutual_info_classif, percentile=30)
    X_train_mutual_info_selected = mutual_info_selector.fit_transform(X_train, y_train)
    X_test_mutual_info_selected = mutual_info_selector.transform(unknown)
    
    # i got these parameters after gridsearch etc in the above code.
    rf_model = RandomForestClassifier(max_depth=None, min_samples_leaf= 2, min_samples_split= 10, n_estimators= 100)
    rf_model.fit(X_train_mutual_info_selected, y_train)
    y_pred = rf_model.predict(X_test_mutual_info_selected)
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Classes'])
    
    outputfile= 'D:\\DSML\\parth_gupta_predicted_labels.csv'
    
    #save output labels
    y_pred_df.to_csv(outputfile, index=True)


TRAIN_DATA=pd.read_csv('D:\\DSML\\training_data.csv')
#TRAIN_DATA=pd.read_csv('D:\\DSML\\dsml\\training_data_sampled.csv')
TRAIN_LABELS=pd.read_csv('D:\\DSML\\training_data_targets.csv',header=None, names=["Labels"])
#TRAIN_LABELS=pd.read_csv('D:\\DSML\\dsml\\training_data_targets_sampled.csv')
TEST_DATA=pd.read_csv('D:\\DSML\\test_data.csv')


if __name__=='__main__':
    
    print('Converting columns in training data to numerical data')
    train_data=numericalize(TRAIN_DATA)
    print('Converting columns in testing data to numerical data')
    test_data=numericalize(TEST_DATA)
    

    train_data1, test_data1= imputation1(train_data,test_data)
    train_data2, test_data2= imputation2(train_data,test_data)
    train_data3, test_data3= imputation3(train_data,test_data)
    train_data4, test_data4= imputation4(train_data,test_data)
    train_data5, test_data5= imputation5(train_data,test_data)
    train_data6, test_data6= imputation6(train_data,test_data)
   
    datasets=[[train_data1,test_data1],[train_data2,test_data2],[train_data3,test_data3],[train_data4,test_data4],[train_data5,test_data5],[train_data6,test_data6]]
       

    #p1=multiprocessing.Process(target=train_knn,args=[X_train, X_test, y_train, y_test])
    #p2=multiprocessing.Process(target=train_decision_tree,args=[X_train, X_test, y_train, y_test])

    
    print('Without using feature selection')
    for i in range(len(datasets)):
        print(f'i:{i}')
        
        p1=multiprocessing.Process(target=train_knn,args=[datasets[i],i+1])
        p2=multiprocessing.Process(target=train_dt,args=[datasets[i],i+1])
        p3=multiprocessing.Process(target=train_rf,args=[datasets[i],i+1])
        p4=multiprocessing.Process(target=train_lr,args=[datasets[i],i+1])
        
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        
    
    print('Now using feature selection.')    
    for i in range(len(datasets)):
        print(f'i:{i}')
        
        p1=multiprocessing.Process(target=train_knn2,args=[datasets[i],i+1])
        p2=multiprocessing.Process(target=train_dt2,args=[datasets[i],i+1])
        p3=multiprocessing.Process(target=train_rf2,args=[datasets[i],i+1])
        p4=multiprocessing.Process(target=train_lr2,args=[datasets[i],i+1])
        
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        
        p1.join()
        p2.join()
        p3.join()
        p4.join()
    
    
    
    print('Now the training is done.')
    print('To output the class lables from the best model:')
    output(datasets[0])
    print('Done writing the file.')