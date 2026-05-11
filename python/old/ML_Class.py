from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import subprocess as sub
from adjustText import adjust_text
from tqdm import tqdm

# ML specific libraries
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc, matthews_corrcoef
from xgboost import XGBClassifier

# conformal predictions
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_K_Fold_Xval_Classification(features_df, ID='compound', model=RandomForestClassifier(n_estimators=100),
                                   folds=5, col_to_rm=['hits', 'label', 'compound'], v=True, ctf=0.5, impute_by_mean=False):
    """
    Runs the Standard K-fold X-validation on the whole dataset randomly taking
    rows for training and test set (homogeneous sampling of the data)

    :param Dataframe features: dataset, contains labels and features
    :param str ID: datapoint identifier
    :param Sklearn_ML model: the ML model to be used
    :param int folds: the number of folds for the X-validation (default 5)
    :param list col_to_rm: the columns to be removed during ML
    :param bool v: whether to have verbose (True by default)
    :param flt ctf: threshold to consider prediction a hit (default: >= 0.5)

    :return rf, pred_df: trained model, real vs predictions (+probas)
    """

    # ----------------
    # 1. data prep
    # ----------------

    features = deepcopy(features_df.sample(frac=1, random_state=0))

    # impute missing values by mean
    if impute_by_mean:
        for col in features.columns:
            if col not in col_to_rm:
                features[col] = features[col].replace(np.NaN, features[col].mean())

    # data to be used in X-validation
    ID_name = np.array(features[[ID]])
    labels = np.array(features['label'])
    features = np.array(features.drop(col_to_rm, axis=1))
    fold_num = 1

    if v:
        print('> feature dim:', features.shape)

    pred_df = {ID: [], 'fold': [], 'real_y': [], 'pred_y': [], 'probas': []}

    # ----------------
    # 2. X-val loop
    # ----------------

    ## 5-fold X-validation splitting using sklearn:
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    kf.get_n_splits(features)

    for train_index, test_index in tqdm(kf.split(features), total=folds):

        train_X, test_X = features[train_index], features[test_index]
        train_y, test_y = labels[train_index], labels[test_index]
        test_ID = ID_name[test_index]

        # run prediction
        # ----------------
        rf = deepcopy(model)

        # Train the model on training data
        rf.fit(train_X, train_y)

        # record probabilities associated with each prediction
        probas = rf.predict_proba(test_X)

        # get binary predictions based on pre-defined threshold
        predictions = (probas[:, 1] >= ctf).astype('int')

        # gather data:
        pred_df[ID] += list(test_ID); pred_df['probas'] += list(probas[:, 1])
        pred_df['pred_y'] += list(predictions); pred_df['real_y'] += list(test_y)
        pred_df['fold'] += [fold_num] * len(list(test_ID))

        # compute/show performance metrics
        lr_acc = metrics.accuracy_score(test_y, predictions)
        lr_precision, lr_recall, _ = precision_recall_curve(test_y, probas[:, 1])
        lr_f1, pr_auc = f1_score(test_y, predictions), auc(lr_recall, lr_precision)
        fpr, tpr, thresholds = metrics.roc_curve(test_y, probas[:, 1])
        roc_auc = metrics.auc(fpr, tpr)

        # evaluation fold:
        if v:
            print('-----------------------------------------')
            print('> Fold %d Accuracy: %.2f, F1: %.2f, Roc_auc: %.2f, PR_auc: %.2f,' % (fold_num, lr_acc, lr_f1, roc_auc, pr_auc))
            tn, fp, fn, tp = metrics.confusion_matrix(test_y, predictions).ravel()
            print('>> True negative (False negative): ', tn, '(', fn, ')')
            print('>> True positive (False positive): ', tp, '(', fp, ')')

        fold_num += 1

    # ------------------------
    # 3. Global model evaluation
    # ------------------------

    pred_df = pd.DataFrame(pred_df)
    real_y = pred_df['real_y']; probas = pred_df['probas']; pred_y = pred_df['pred_y']
    precision_, recall_, _ = precision_recall_curve(real_y, probas)
    fpr_, tpr_, thresholds_ = metrics.roc_curve(real_y, probas)

    if v:
        print('\n>>> Global accuracy: %.2f, F1: %.2f, ROC_AUC: %.2f, PR_AUC: %.2f' % (metrics.accuracy_score(real_y, pred_y), f1_score(real_y, pred_y),
              metrics.auc(fpr_, tpr_), auc(recall_, precision_)))

    return rf, pred_df


def plot_roc_curve(pred_dfs, c=['cornflowerblue', 'orange', 'purple'], l=['', '', ''], selected_pts=[]):
    """
    Plot mean roc curve with nice outline
    """
    # get auroc table with thresholds:
    df_aucs = []
    texts = []
    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)

    for i in range(len(pred_dfs)):

        pred_df = pred_dfs[i]

        # get values from metrics
        real_y = pred_df['real_y']; pred_y = pred_df['pred_y']; probas = pred_df['probas']
        fpr_, tpr_, thresholds_ = metrics.roc_curve(real_y, probas)
        precision_, recall_, _ = precision_recall_curve(real_y, probas)
        mcc = metrics.matthews_corrcoef(real_y, pred_y)

        ## Extract table showing thresholds associated with fpr and tpr
        df_auc = pd.DataFrame({'fpr': fpr_, 'tpr': tpr_, 'thresholds': thresholds_}).sort_values('thresholds').assign(title=l[i])
        df_aucs.append(df_auc)

        # select points to show thresholds:
        if len(selected_pts) > 0:
            selected_thresholds = []
            for pt in selected_pts:
                df_auc['select'] = abs(df_auc['threshold'] - pt)
                selected_thresholds.append(df_auc.sort_values('select').head(1))
            selected_thresholds = pd.concat(selected_thresholds).reset_index(drop=True)

        ## Plotting
        fpr = np.linspace(0, 1, 100)
        tpr = np.interp(fpr, fpr_, tpr_)

        plt.plot(fpr, tpr, color=c[i],
                 label='%s - n=%d - roc_auc: %0.2f - pr_auc: %0.2f' % (l[i], pred_df.shape[0], metrics.auc(fpr_, tpr_), auc(recall_, precision_)),
                 linewidth=2.0, alpha=0.8)
        plt.plot(fpr, tpr, color=c[i],
                 label='%s - n=%d - roc_auc: %0.2f - MCC: %0.2f' % (l[i], pred_df.shape[0], metrics.auc(fpr_, tpr_), mcc),
                 linewidth=2.0, alpha=0.8)
        if len(selected_pts):
            plt.scatter(x=selected_thresholds['fpr'], y=selected_thresholds['tpr'])

            for index, row in selected_thresholds.iterrows():
                texts.append(plt.text(row['fpr'], row['tpr'], round(row['threshold'], 2), ha='center', c='black'))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')

    # # adjust texts on graph
    # adjust_text(texts, force_points=0.2, force_text=0.2,
    #             expand_points=(1, 1), expand_text=(1, 1),
    #             arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))

    return pd.concat(df_aucs).reset_index(drop=True)


def compute_npv(df):
    CM = metrics.confusion_matrix(df['real_y'], df['pred_y'], labels=[0, 1])
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TN / (TN + FN)


def metrics_from_pred_df(pred_df):
    """
    From pred_df (contains real_y, pred_y and probabilities) gets metrics such
    as AUC
    """
    pred_y = pred_df['pred_y']; real_y = pred_df['real_y']; probas = pred_df['probas']
    lr_acc = metrics.accuracy_score(real_y, pred_y)
    lr_precision, lr_recall, _ = precision_recall_curve(real_y, probas)
    lr_f1, pr_auc = f1_score(real_y, pred_y), auc(lr_recall, lr_precision)
    fpr, tpr, thresholds = metrics.roc_curve(real_y, probas)
    roc_auc = metrics.auc(fpr, tpr)

    res = {'roc_auc': roc_auc, 'pr_auc': pr_auc}
    return res


def classification_from_regression_pred(pred_df, ctf_pred, sign='>=', ctf_real=None, v=True):
    """
    Show confidency table, F1, acc & precision from continuous value predictions

    :param Dataframe df: Dataframe containing (real_y pred_y) columns
    :param dbl ctf: the cutoff value to split 1 vs 0 labels
    :param str sign: direction of label discretization
    :return F1, df_tbl: return F1 & confidency table
    """

    if ctf_real is None:
        ctf_real = ctf_pred

    reg_tbl = pred_df[['pred_y', 'real_y']].copy()
    if sign == '<':
        reg_tbl['pred_y'] = reg_tbl['pred_y'] < ctf_pred
        reg_tbl['real_y'] = reg_tbl['real_y'] < ctf_real
    else:
        reg_tbl['pred_y'] = reg_tbl['pred_y'] >= ctf_pred
        reg_tbl['real_y'] = reg_tbl['real_y'] >= ctf_real

    if v:
        print('> accuracy: %.2f, precision(PPV): %.2f, NPV: %.2f, F1(+): %.2f, F1(-): %.2f, MCC: %.2f' %
              (metrics.accuracy_score(reg_tbl['real_y'], reg_tbl['pred_y']),
               metrics.precision_score(reg_tbl['real_y'], reg_tbl['pred_y']),
               compute_npv(reg_tbl), f1_score(reg_tbl['real_y'], reg_tbl['pred_y']),
               f1_score(reg_tbl['real_y'], reg_tbl['pred_y'], pos_label=False),
               matthews_corrcoef(reg_tbl['real_y'], reg_tbl['pred_y'])))

    # update confusion matrix
    t = [['True +', 'False -'], ['False -', 'True +']]
    CM = metrics.confusion_matrix(reg_tbl['real_y'], reg_tbl['pred_y'], labels=[0, 1]).T.flatten()
    return f1_score(reg_tbl['real_y'], reg_tbl['pred_y']), metrics.precision_score(reg_tbl['real_y'], reg_tbl['pred_y']), pd.DataFrame(CM, index=[['pos', 'neg'], ['T', 'F']])


def get_PPV_vs_proba(ctf_pred, pred, real_y, plot=True):
    """
    Plot TPR associated to increasing probability thresholds.
    :param df df_pred: with columns [compound,label,pred_y,real_y,probas]
    :param int npts: number of proba cutoffs going from 0 to 1
    :param bool plot: whether to plot the graph (default yes)

    :return df to_plot: the raw dataframe containing data
    """
    to_plot = []
    pts = []
    for i in range(npts):
        actf = i * npts
        if actf >= size:
            tmp = c_pred[c_pred['probas'] >= actf]
            PPV = len(tmp[tmp['label'] == 1]) / tmp.shape[0]
            to_plot.append([actf, PPV, tmp.shape[0], pred.shape[0], tmp.shape[0]])
    df_to_plot = pd.DataFrame(to_plot, columns=['threshold', 'PPV', 'fraction', 'data points'])

    ## plotting
    if plot:
        plt.figure(figsize=(6, 4))
        plt.scatter('threshold', 'PPV', marker='+', data=df_to_plot, label='PPV')
        plt.plot('threshold', 'fraction', marker='+', data=df_to_plot, label='fraction data')
        plt.scatter('threshold', 'PPV', data=df_to_plot, color='#0000ff', linewidth=3, markersize=4, label='PPV')
        plt.plot('threshold', 'fraction', data=df_to_plot, marker='+', linestyle='--', color='lightgrey', linewidth=2, markersize=4, label='fraction data')
        plt.legend(); plt.xlabel('probability cutoff values')
        plt.ylim(0, 1.01); plt.xlim(0, plt.xlim()[1])
        plt.grid(axis='x')

    return df_to_plot


def plot_TPR_TNR_thresholds(df):
    """
    Plot TPR & TNR associated with probability thresholds
    """
    fpr, tpr, thresholds = metrics.roc_curve(df['real_y'], df['probas'])
    plt.plot(thresholds, 1.0 - fpr, label='TNR')
    plt.plot(thresholds, tpr, label='TPR')
    plt.legend(); plt.xlabel('probability thresholds')


def plot_PR_thresholds(df):
    """
    Plot Precision & Recall associated with probability thresholds
    """
    precision_, recall_, thresholds_ = precision_recall_curve(df['real_y'], df['probas'])
    dist_ = abs(precision_ - recall_)
    thresholds_ = np.append(thresholds_, 1.0)

    # get threshold where Precision meets Recall curve:
    m_df = pd.DataFrame({'precision': precision_, 'recall': recall_, 'threshold': thresholds_, 'distance': dist_})
    print('>> ideal cutoff:', m_df[m_df['distance'] == m_df['distance'].min()]['threshold'].item())

    plt.plot(thresholds_, precision_, label='precision')
    plt.plot(thresholds_, recall_, label='recall')
    plt.plot(thresholds_, dist_, label='distance')
    plt.legend(); plt.xlabel('probability thresholds')


def confusion_cf_from_pred_df(df_, ctf=0.5):
    df = df_.copy()
    if ctf > 0.5:
        df['pred_y'] = (df['probas'] > ctf) * 1

    print('> accuracy: %.2f, precision(PPV): %.2f, NPV: %.2f, F1: %.2f' %
          (metrics.accuracy_score(df['real_y'], df['pred_y']),
           metrics.precision_score(df['real_y'], df['pred_y']),
           compute_npv(df), f1_score(df['real_y'], df['pred_y'])))

    t = [['True +', 'False -', 'False -', 'True +']]
    CM = metrics.confusion_matrix(df['real_y'], df['pred_y'], labels=[0, 1]).T.flatten()

    # get nice contingency table
    return pd.DataFrame({'Type': t, 'Count': CM})


def compute_ROC_AUC(real_y, probas):
    """
    from 2 lists of real values and probabilities, returns ROC_auc
    """
    fpr_, tpr_, thresholds_ = metrics.roc_curve(list(real_y), list(probas))
    return metrics.auc(fpr_, tpr_)


pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute missing values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def predict_missing_values_classification(df_full, col_2_impute, get_ints=False):
    """
    TBD
    Try to fill in missing values by using the correlation between co-variates
    and Morgan fingerprint features

    :param df df_full: the complete dataset (compoundID,...,MFB,[label1,...,labelX]
    :param list col_2_impute: the list of columns to fill in blanks
    :param bool get_ints: whether to compute prediction interval (False as default)
    """
    # Specifying upfront the columns names which missing values to be imputed
    print('>> predicting labels:', col_2_impute)

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)

    # initiate final dataframe to hold all predicted endpoints
    rf_preds = pd.DataFrame({'compound': df_full['compound']})

    # predict endpoints one at a time:
    for label in col_2_impute:
        print('>> filling missing values for:', label)
        # indicate column (endpoint) to predict
        tmp = df_full.copy().rename(columns={label: 'label'}).reset_index(drop=True)

        # [SOLUBILITY ONLY] remove solubilities > 220 and transform to logs:
        if 'solubility' in label:
            tmp.loc[tmp['label'] > 220, 'label'] = np.nan
            tmp.loc[tmp['label'] <= 0.0, 'label'] = tmp[tmp['label'] > 0.0]['label'].min() * 1e-7
            tmp['label'] = np.log10(tmp['label'] / 1e6)

        # split between training and prediction sets
        c = tmp['label'].notnull()
        train_CMs, test_CMs = list(tmp[c]['compound']), list(tmp[~c]['compound'])

        ## run predictions
        rf, df_pred = K_fold_by_defined_IDs(tmp, ID='compound', ID_sets=[[train_CMs, test_CMs]], model=model,
                                            col_to_rm=['compound', 'label'], get_ints=get_ints, predict=True)

        # re-convert solubility back to uM
        if 'solubility' in label:
            df_pred['pred_y'] = 1e6 * (10 ** df_pred['pred_y'])
            df_pred['low'] = 1e6 * (10 ** df_pred['low'])
            df_pred['up'] = 1e6 * (10 ** df_pred['up'])

        # Get the predicted endpoint & prediction interval as column
        df_pred[label + '_pred'] = df_pred['pred_y']
        df_pred[label + '_P.I.'] = df_pred[['pred_y', 'low', 'up']].apply(lambda x: '%.1f (%.1f,%.1f)' % (x[0], x[1], x[2]), axis=1)

        # incrementally add predicted endpoints to final prediction table:
        rf_preds = pd.merge(rf_preds, df_pred[['compound', label + '_pred', label + '_P.I.']], on='compound', how='left')

    return rf_preds


pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chemprop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def chemprop_K_Fold_Xval_Classification(path, features, ID='smiles', folds=5, hyperopt=True, v=True, ctf=0.5):
    """
    Wrapper to run chemprop on K-Fold X-validation (can take a long time...)

    :param str path: chemprop path e.g. .../.chemprop/
    :param dataframe features: should contain only [smiles|label1|label2]...
    :param str ID: datapoint identifier
    :param int folds: the number of folds for the X-validation (default 5)
    :param bool v: whether to have verbose (True by default)
    :param flt ctf: threshold to consider prediction a hit (default: >= 0.5)

    :return rf, pred_df: trained model, real vs predictions (+probas)
    """

    # ---------------------
    # 1. data prep
    # ---------------------

    # first write train folder into processing directory:
    features.to_csv(path + 'processing/full.csv', index=False, sep=',')

    # hyperparameter estimation:
    if hyperopt:
        print('>> requested hyperparameter optimization (can take some time...)')
        c = 'python ' + path + 'hyperparameter_optimization.py --data_path ' + path + 'processing/full.csv --dataset_type classification --num_iters 30 --config_save_path ' + path + 'processing/local_config'
        print(c)
        sub.call(c, shell=True)
    else:
        sub.call('cp ' + path + 'params/HbF_20210518_config ' + path + 'processing/local_config', shell=True)

    # ---------------------
    # 2. X-val loop
    # ---------------------

    fold_num = 1
    pred_df = {ID: [], 'fold': [], 'real_y': [], 'pred_y': [], 'probas': []}
    pbar = ProgressPercent(folds)

    ## 5-fold X-validation splitting using sklearn:
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    kf.get_n_splits(features)

    pred_df = []
    for train_index, test_index in kf.split(features):

        train = features.iloc[train_index]
        test = features.iloc[test_index]

        # remove all values in test set:
        test_ori = test.copy()
        test[[x for x in test.columns if x != 'smiles']] = np.nan

        # write train and test to file
        train.to_csv(path + 'processing/train/' + str(fold_num) + '.csv', index=False, sep=',')
        test.to_csv(path + 'processing/test/' + str(fold_num) + '.csv', index=False, sep=',')

        # fit model on training set:
        c = 'python ' + path + 'train.py --data_path ' + path + 'processing/train/' + str(fold_num) + '.csv --dataset_type classification --config_path ' + path + 'processing/local_config --save_dir ' + path + 'processing/chmpt_fold/' + str(fold_num)
        # print(c)
        sub.call(c, shell=True)
        print('> done training', fold_num)

        # predict test set:
        c = 'python ' + path + 'predict.py --test_path ' + path + 'processing/test/' + str(fold_num) + '.csv --checkpoint_dir ' + path + 'processing/chmpt_fold/' + str(fold_num) + '/ --preds_path ' + path + 'processing/pred' + str(fold_num) + '.csv'
        # print(c)
        sub.call(c, shell=True)
        print('> done predicting', fold_num)

        # get test set predictions vs real values for each CM by renaming columns
        pred = pd.read_csv(path + 'processing/pred' + str(fold_num) + '.csv').rename(columns={'label': 'probas'})
        pred.columns = [x + '_p' if x != 'smiles' else x for x in pred.columns]

        pred = pd.merge(test_ori, pred, on='smiles').rename(columns={'label': 'real_y'})
        try:
            for label in test_ori.drop('smiles', axis=1):
                pred[label + '_p'] = (pred[label + '_p'] > ctf) * 1.0
        except:
            print('problem')
            return pred
        pred['fold'] = fold_num
        pred_df.append(pred)

        # get aucs metrics for each pred
        m = metrics_from_pred_df(pred)

        if v:
            pbar.increment()
            for label in test_ori.drop('smiles', axis=1):
                pbar.increment('>> fold ' + str(fold_num) + ' - roc_auc: ' + str(round(m['roc_auc'], 2)) + ' - pr_auc: ' + str(round(m['pr_auc'], 2)))

        fold_num += 1

    pred_df = pd.concat(pred_df)

    return pred_df


pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SHAP utils
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def xgb_shap_change_base(original_shap_values, Y_pred, base_value, which):
    untransformed_base_value = original_shap_values.base_values[-1]

    # Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]

    # base_value = expit(untransformed_base_value) # = 1 / (1+ np.exp(-untransformed_base_value))

    # Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = Y_pred[which] - base_value

    # The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain

    # Transforming the original shapley values to the new scale
    shap_values_transformed = original_shap_values / distance_coefficient

    # Finally resetting the base_value as it does not need to be transformed
    shap_values_transformed.base_values = [base_value] * len(original_shap_values)  # GT_modif_20210609
    shap_values_transformed.data = original_shap_values.data

    # Now returning the transformed array
    return shap_values_transformed


def xgb_shap_transform_scale(original_shap_values, Y_pred, which):
    from scipy.special import expit

    # Compute the transformed base value, which consists in applying the logit function to the base value
    from scipy.special import expit  # Importing the logit function for the base value transformation
    untransformed_base_value = original_shap_values.base_values[-1]

    # Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]

    base_value = expit(untransformed_base_value)  # = 1 / (1+ np.exp(-untransformed_base_value))

    # Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = Y_pred[which] - base_value

    # The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain

    # Transforming the original shapley values to the new scale
    shap_values_transformed = original_shap_values / distance_coefficient

    # Finally resetting the base_value as it does not need to be transformed
    shap_values_transformed.base_values = [base_value] * len(original_shap_values)  # GT_modif_20210609
    shap_values_transformed.data = original_shap_values.data

    # Now returning the transformed array
    return shap_values_transformed


pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conformal Predictions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_K_Fold_Xval_CP(features_df, ID='compound', model=RandomForestClassifier(n_estimators=100),
                       folds=5, col_to_rm=['hits', 'label', 'compound'], v=True, ctf=0.5, impute=False):
    """
    Runs the Standard K-fold X-validation on the whole dataset randomly taking
    rows for training and test set (homogeneous sampling of the data)

    :param Dataframe features: dataset, contains labels and features
    :param str ID: datapoint identifier
    :param Sklearn_ML model: the ML model to be used
    :param int folds: the number of folds for the X-validation (default 5)
    :param list col_to_rm: the columns to be removed during ML
    :param bool v: whether to have verbose (True by default)
    :param flt ctf: threshold to consider prediction a hit (default: >= 0.5)

    :return rf, pred_df: trained model, real vs predictions (+probas)
    """

    # instantiate CP
    nc = NcFactory.create_nc(model)  # column gets a default nonconformity function
    icp = IcpClassifier(nc)          # Create an inductive conformal classifier

    # ----------------
    # 1. data prep
    # ----------------

    features = deepcopy(features_df.sample(frac=1, random_state=0))

    # impute missing values by mean
    if impute:
        for col in features.columns:
            if col not in col_to_rm:
                features[col] = features[col].replace(np.NaN, features[col].mean())

    # data to be used in X-validation
    ID_name = np.array(features[[ID]])
    labels = np.array(features['label'])
    features = np.array(features.drop(col_to_rm, axis=1))
    fold_num = 1

    if v:
        print('>> feature dim:', features.shape)

    pred_df = []  # {ID:[], 'fold':[], 'real_y':[], 'pred_y':[], 'probas':[]}

    # ----------------
    # 2. X-val loop
    # ----------------

    ## 5-fold X-validation splitting using sklearn:
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    kf.get_n_splits(features)

    for train_index, test_index in kf.split(features):

        # further split test set into calibration as well
        cal_index, train_index = train_test_split(train_index, test_size=0.8, random_state=0)
        print('> train:', len(train_index), 'cal:', len(cal_index), 'test:', len(test_index))

        # get train, calibration and test sets
        train_X, test_X, cal_X = features[train_index], features[test_index], features[cal_index]
        train_y, test_y, cal_y = labels[train_index], labels[test_index], labels[cal_index]

        test_ID = ID_name[test_index]

        # run prediction
        # ----------------

        # Fit the ICP using the proper training set
        icp.fit(train_X, train_y)

        # Calibrate the ICP using the calibration set
        icp.calibrate(cal_X, cal_y)

        # Produce predictions for the test set
        p_values = icp.predict(test_X, significance=None)

        # 1.col = class, 2.col=pred.prob, 3.col-> credibility
        pred_class = icp.predict_conf(test_X)

        pred_all = np.c_[pred_class, p_values]
        df = pd.DataFrame(data=pred_all, columns=['pred_y', 'prob', 'cred', 'p_1', 'p_2'])
        df[ID] = test_ID

        # define a column where the prob always corresponds to class 1
        df['probas'] = df['prob']
        df.loc[df['pred_y'] == 0, 'probas'] = 1 - df[df['pred_y'] == 0]['prob']

        # define predicted class label based on threshold:
        # df['pred_y'] = (df['probas'] >= ctf)*1
        df['real_y'] = test_y
        df['fold'] = [fold_num] * len(list(test_ID))

        pred_df.append(df)

        # compute/show performance metrics
        test_y = df['real_y']; predictions = df['pred_y']; probas = df['probas']

        lr_acc = metrics.accuracy_score(test_y, predictions)
        lr_precision, lr_recall, _ = precision_recall_curve(test_y, probas)
        lr_f1, pr_auc = f1_score(test_y, predictions), auc(lr_recall, lr_precision)
        fpr, tpr, thresholds = metrics.roc_curve(test_y, probas)
        roc_auc = metrics.auc(fpr, tpr)

        # evaluation fold:
        if v and not predict:
            print('-----------------------------------------')
            print('> Fold %d Accuracy: %.2f, F1: %.2f, Roc_auc: %.2f, PR_auc: %.2f,' % (fold_num, lr_acc, lr_f1, roc_auc, pr_auc))
            tn, fp, fn, tp = metrics.confusion_matrix(test_y, predictions).ravel()
            print('>> True negative (False negative): ', tn, '(', fn, ')')
            print('>> True positive (False positive): ', tp, '(', fp, ')')

        fold_num += 1

    # ------------------------
    # 3. Global model evaluation
    # ------------------------

    # pred_df = pd.DataFrame(pred_df)
    pred_df = pd.concat(pred_df).reset_index(drop=True)

    real_y = pred_df['real_y']; probas = pred_df['probas']; pred_y = pred_df['pred_y']
    precision_, recall_, _ = precision_recall_curve(real_y, probas)
    fpr_, tpr_, thresholds_ = metrics.roc_curve(real_y, probas)

    if v:
        print('\n>>> Global Accuracy: %.2f, F1: %.2f, ROC_AUC: %.2f, PR_AUC: %.2f' % (metrics.accuracy_score(real_y, pred_y), f1_score(real_y, pred_y),
              metrics.auc(fpr_, tpr_), auc(recall_, precision_)))

    # plot_roc_curve(pred_df)

    return icp, pred_df


def K_fold_by_ID_CP(features_df, ID, ID_sets, model=RandomForestClassifier(n_estimators=100),
                    get_metrics=False, folds=5, col_to_rm=['label', 'compound'], v=True, ctf=0.5, impute=False, predict=False):
    """
    Very similar to K_fold_by_random_gene function above, but this time we take
    specific sets of genes for training and testing hold a portion of genes and
    predict on them.

    :param Dataframe features: dataset, contains labels and features
    :param str ID: the column to select train & test sets
    :param list ID_sets: the training and test sets [[train,test],[...],...]
    :param Sklearn_ML model: the ML model to be used
    :param bool get_metrics: whether to return metrics (e.g. AUC, f1, ...)
    :param int folds: the number of folds for the crossvalidation (default 5)
    :param list col_to_rm: the columns to be removed during ML
    :param bool v: whether to have verbose (True by default)
    :param float ctf: the cutoff for probabilities
    :param bool predict: whether to assess prediction on the test set

    :return: depends on the boolean parameters above
    """

    # instantiate CP
    nc = NcFactory.create_nc(model)  # column gets a default nonconformity function
    icp = IcpClassifier(nc)          # Create an inductive conformal classifier

    # ----------------
    # 1. data prep
    # ----------------

    features = features_df.copy()
    features['label'] = features['label'].astype(int)

    # impute missing values by mean
    if impute:
        for col in features.columns:
            if col not in col_to_rm:
                features[col] = features[col].replace(np.NaN, features[col].mean())

    # 5-fold X-validation splitting using sklearn:
    fold_num = 1
    pred_df = []  # {ID:[], 'fold':[], 'real_y':[], 'pred_y':[], 'probas':[]}
    ID_name = np.array(features[[ID]])
    print('>> dim train:', len(ID_sets[0][0]), '- test:', len(ID_sets[0][1]))

    # ----------------
    # 2. X-val loop
    # ----------------

    for train_IDs, test_IDs in ID_sets:

        # further split test set into calibration as well
        cal_IDs, train_IDs = train_test_split(train_IDs, test_size=0.8, random_state=0)

        train_y = features[features[ID].isin(train_IDs)][['label']]
        train_X = features[features[ID].isin(train_IDs)].drop(col_to_rm, axis=1)
        cal_y = features[features[ID].isin(cal_IDs)][['label']].astype(int)
        print(cal_y)
        cal_X = features[features[ID].isin(cal_IDs)].drop(col_to_rm, axis=1)
        test_y = np.array(features[features[ID].isin(test_IDs)][['label']])
        test_X = np.array(features[features[ID].isin(test_IDs)].drop(col_to_rm, axis=1))

        # test_ID = test_IDs

        # run prediction
        # ----------------

        # Fit the ICP using the proper training set
        icp.fit(train_X, train_y)
        return
        # Calibrate the ICP using the calibration set
        icp.calibrate(cal_X, cal_y)

        # Produce predictions for the test set
        p_values = icp.predict(test_X, significance=None)

        # 1.col = class, 2.col=pred.prob, 3.col-> credibility
        pred_class = icp.predict_conf(test_X)

        pred_all = np.c_[pred_class, p_values]
        df = pd.DataFrame(data=pred_all, columns=['pred_y', 'prob', 'cred', 'p_1', 'p_2'])
        df[ID] = test_ID

        # define a column where the prob always corresponds to class 1
        df['probas'] = df['prob']
        df.loc[df['pred_y'] == 0, 'probas'] = 1 - df[df['pred_y'] == 0]['prob']

        # define predicted class label based on threshold:
        # df['pred_y'] = (df['probas'] >= ctf)*1
        df['real_y'] = test_y
        df['fold'] = [fold_num] * len(list(test_ID))

        pred_df.append(df)

        # compute/show performance metrics

        test_y = df['real_y']; predictions = df['pred_y']; probas = df['probas']

        if predict == False:
            lr_acc = metrics.accuracy_score(test_y, predictions)
            lr_precision, lr_recall, _ = precision_recall_curve(test_y, probas)
            lr_f1, pr_auc = f1_score(test_y, predictions), auc(lr_recall, lr_precision)
            fpr, tpr, thresholds = metrics.roc_curve(test_y, probas)
            roc_auc = metrics.auc(fpr, tpr)

        # evaluation fold:
        if v and not predict:
            print('-----------------------------------------')
            print('> Fold %d Acc(ctf=%.1f): %.2f, F1(ctf=%.1f): %.2f, Roc_auc: %.2f, PR_auc: %.2f,' % (fold_num, ctf, lr_acc, ctf, lr_f1, roc_auc, pr_auc))
            tn, fp, fn, tp = metrics.confusion_matrix(test_y, predictions).ravel()
            print('>> True negative (False negative): ', tn, '(', fn, ')')
            print('>> True positive (False positive): ', tp, '(', fp, ')')

        fold_num += 1

    # ------------------------
    # 3. Global model evaluation
    # ------------------------

    # pred_df = pd.DataFrame(pred_df)
    pred_df = pd.concat(pred_df).reset_index(drop=True)

    real_y = pred_df['real_y']; probas = pred_df['probas']; pred_y = pred_df['pred_y']
    if predict == False:
        precision_, recall_, _ = precision_recall_curve(real_y, probas)
        fpr_, tpr_, thresholds_ = metrics.roc_curve(real_y, probas)

    if v and not predict:
        print('\n>>> Global accuracy: %.2f, F1: %.2f, roc_auc: %.2f, PR_AUC: %.2f' % (metrics.accuracy_score(real_y, pred_y), f1_score(real_y, pred_y),
              metrics.auc(fpr_, tpr_), auc(recall_, precision_)))

    return icp, pred_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute missing values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def predict_missing_values_classification_CP(df_full, col_2_impute):
    """
    Try to fill in missing values by using Morgan Fingerprint features (single task)
    :param df df_full: the complete dataset (compoundID,...,MFP1,...,MFPB,label1,...,labelN)
    :param list col_2_impute: the list of columns to fill in blanks
    :return df: the df with probabilities + uncertainty
    """
    # Specifying upfront the columns names which missing values to be imputed
    print('>> predicting labels:', col_2_impute)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)

    # create empty dataframe to hold all predicted endpoints
    rf_preds = pd.DataFrame({'compound': df_full['compound']})

    # predict endpoints one at a time:
    for label in tqdm(col_2_impute):

        # print('>> filling missing values for:', label)
        # indicate column (endpoint) to predict
        tmp = df_full.copy().rename(columns={label: 'label'}).reset_index(drop=True).drop([x for x in col_2_impute if x not in ['label', label]], axis=1)

        # split between training and prediction sets:
        c = tmp['label'].notnull()
        train_CMs, test_CMs = list(tmp[c]['compound']), list(tmp[~c]['compound'])

        ## run predictions
        icp, df_pred = K_fold_by_ID_CP(tmp, ID='compound', ID_sets=[[train_CMs, test_CMs]], model=model,
                                       col_to_rm=['compound', 'label', 'smiles'], v=False, ctf=0.5, predict=True)

        df_pred = df_pred.rename(columns={'probas': label + '_probas', 'cred': label + '_cred'})

        # incrementally add predicted endpoints to final prediction table:
        rf_preds = pd.merge(rf_preds, df_pred[['compound', label + '_probas', label + '_cred']], on='compound', how='left')

    return rf_preds


pass
