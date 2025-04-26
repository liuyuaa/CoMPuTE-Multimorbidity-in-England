import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import top_k_accuracy_score

def get_adjusted_shap_abnormal_option(shap_values_i, normal_ranges_x, opt):
    if opt == 'adjust':
        mask_shap = (shap_values_i.values > 0).astype(int)  # select samples with positive relationships
        for i in range(shap_values_i.shape[1]):
            fea_i = shap_values_i.feature_names[i]
            lower_bound, upper_bound = normal_ranges_x[fea_i]
            mask_fea_i = (shap_values_i.data[:, i] > upper_bound) | (shap_values_i.data[:, i] < lower_bound)  # select samples with abnormal values
            mask_fea_i = mask_fea_i.reshape(-1, 1)
            mask_fea = mask_fea_i if i == 0 else np.concatenate([mask_fea, mask_fea_i], axis=1)
        shap_adjusted = shap_values_i.values * mask_shap * mask_fea
        mask_nan = ~np.isnan(shap_values_i.data)  # select samples with observed feature values
        shap_adjusted = shap_adjusted * mask_nan
        scores = np.abs(shap_adjusted).sum(axis=0) / (1e-90 + (shap_adjusted > 0).sum(axis=0))
        scores[np.isnan(scores)] = 0
        scores[(shap_adjusted > 0).sum(axis=0) < 500] = np.nan  # filter out case without enough samples
    abnormal = []
    for i in range(shap_values_i.shape[1]):
        fea_i = shap_values_i.feature_names[i]
        lower_bound, upper_bound = normal_ranges_x[fea_i]
        valid_mask = (mask_shap[:, i] * mask_fea[:, i] * mask_nan[:, i]) > 0
        tt = shap_values_i.data[valid_mask, i]
        if (tt > upper_bound).sum() > (tt < lower_bound).sum():
            abnormal.append(1)
        else:
            abnormal.append(-1)
    return scores, np.array(abnormal)

# Clinical marker list
pick_biomarkers = ['BMI', 'SBP', 'DBP', ..., 'eGFR']
pick_biomarkers = sorted(list(set(pick_biomarkers)))

normal_ranges_M = {'BMI': (18.5, 24.9), 'DBP': (60, 80), 'SBP': (90, 120),}
normal_ranges_F = {'BMI': (18.5, 24.9), 'DBP': (60, 80), 'SBP': (90, 120),}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=12)
    parser.add_argument('--n_sample', type=int, default=10000)
    parser.add_argument('--num_booster', type=int, default=500)
    args = parser.parse_args()

    max_depth = args.max_depth
    n_samples = args.n_sample
    num_booster = args.num_booster

    for gender_x in ['M', 'F']:
        n_profile = 21 if gender_x == 'M' else 18
        male_profile, female_profile = [gender_x + str(i) for i in range(1, n_profile + 1)]

        n_shap = 21 * 10000 if gender_x == 'M' else 18 * 10000
        label_list = [f"{gender_x}{i}" for i in range(1, n_profile + 1)]
        y_label2num = {k: v for v, k in enumerate(label_list)}

        df_xgb = pd.read_csv(f'./data/marker/prof_biomarker_{gender_x}.csv', header=0)
        df_xgb['y_label'] = df_xgb['label'].apply(lambda x: y_label2num[x])
        x0, y0 = df_xgb.index.tolist(), df_xgb['label']
        x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=0.2, random_state=61, stratify=y0)
        df_xgb_train, df_xgb_test = df_xgb.loc[sorted(x_train)], df_xgb.loc[sorted(x_test)]

        # XGBoost Training with GPU
        print("Training XGBoost model on GPU...")
        x_train, y_train = df_xgb_train[pick_biomarkers], df_xgb_train['y_label']
        x_test, y_test = df_xgb_test[pick_biomarkers], df_xgb_test['y_label']
        train_dmatrix = xgb.DMatrix(x_train, label=y_train)
        test_dmatrix = xgb.DMatrix(x_test, label=y_test)
        num_classes = len(label_list)
        watchlist = [(train_dmatrix, 'train'), (test_dmatrix, 'test'), ]

        params = {'objective': 'multi:softprob', 'num_class': len(label_list), 'max_depth': max_depth, 'learning_rate': 0.1,
                  'eval_metric': ['mlogloss', 'auc'], 'random_state': 61, 'tree_method': 'hist',
                  "device": "cuda" # Enable GPU training
                  }
        model = xgb.train(params, train_dmatrix, num_boost_round=num_booster, evals=watchlist, verbose_eval=5, early_stopping_rounds=20)
        # Evaluating XGBoost
        print("\nEvaluating Model...")
        y_pred_probs_test = model.predict(test_dmatrix)
        y_pred_test = np.argmax(y_pred_probs_test, axis=1)
        auroc_test = roc_auc_score(y_test, y_pred_probs_test, average='macro', multi_class='ovr')
        auprc_test = average_precision_score(y_test, y_pred_probs_test, average='macro')
        hit_1_test = top_k_accuracy_score(y_test, y_pred_probs_test, k=1, labels=list(range(num_classes)))
        hit_3_test = top_k_accuracy_score(y_test, y_pred_probs_test, k=3, labels=list(range(num_classes)))
        hit_5_test = top_k_accuracy_score(y_test, y_pred_probs_test, k=5, labels=list(range(num_classes)))
        print(f"Testing\nAUROC: {auroc_test:.2f}, AUPRC: {auprc_test:.2f}, "
              f"Hit@1:{hit_1_test:.2f}, Hit@3:{hit_3_test:.2f}, Hit@5:{hit_5_test:.2f}\n")

        y_pred_probs_train = model.predict(train_dmatrix)
        y_pred_train = np.argmax(y_pred_probs_train, axis=1)
        auroc_train = roc_auc_score(y_train, y_pred_probs_train, average='macro', multi_class='ovr')
        auprc_train = average_precision_score(y_train, y_pred_probs_train, average='macro')
        hit_1_train = top_k_accuracy_score(y_train, y_pred_probs_train, k=1, labels=list(range(num_classes)))
        hit_3_train = top_k_accuracy_score(y_train, y_pred_probs_train, k=3, labels=list(range(num_classes)))
        hit_5_train = top_k_accuracy_score(y_train, y_pred_probs_train, k=5, labels=list(range(num_classes)))
        print(f"Training\nAUROC: {auroc_train:.2f}, AUPRC: {auprc_train:.2f}, "
              f"Hit@1:{hit_1_train:.2f}, Hit@3:{hit_3_train:.2f}, Hit@5:{hit_5_train:.2f}\n")

        # SHAP Value Calculation
        explainer = shap.TreeExplainer(model)
        if x_train.shape[0] < n_shap:
            x_shap = x_train
        else:
            x_shap = x_train.sample(n=n_shap, random_state=61)
        print('Start shap analysis...')
        shap_values = explainer(x_shap)  # SHAP values for all classes
        shap_path = f'./results/shap_{gender_x}.pkl'
        with open(shap_path, 'wb') as f:
            pickle.dump(shap_values, f)
        print("SHAP computation completed and saved.")

        # SHAP adjustment with reference range
        normal_ranges_x = normal_ranges_M if gender_x == 'M' else normal_ranges_F
        max_depth, num_booster, n_sample = 12, 500, 10000
        n_shap = 210000 if gender_x == 'M' else 180000

        with open(f'./results/shap_{gender_x}.pkl', 'rb') as f:
            shap_values = pickle.load(f)
        clus_names = male_profile if gender_x == 'M' else female_profile
        df_fi_shap = pd.DataFrame()
        abnormal_mat = []
        for i, clus_name_i in enumerate(clus_names):
            ii = clus_names.index(clus_name_i)
            shap_values_i = shap_values[:, :, ii]
            mean_abs_shap_i, abnormal_i = get_adjusted_shap_abnormal_option(shap_values_i, normal_ranges_x, opt='adjust')
            abnormal_mat.append(abnormal_i)
            df_fi_shap_i = pd.DataFrame({'fea': shap_values_i.feature_names, clus_name_i: mean_abs_shap_i})
            df_fi_shap = df_fi_shap_i if i == 0 else pd.merge(df_fi_shap, df_fi_shap_i, on='fea', how='left')
        df_fi_shap = df_fi_shap.set_index('fea')
        normalized_df = df_fi_shap.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)  # min-max normalization
        abnormal_mat = np.array(abnormal_mat).T
        normalized_df = normalized_df * abnormal_mat
        normalized_df.to_csv(f'./results/relevance_marker_prof_{gender_x}.csv', index=True)
