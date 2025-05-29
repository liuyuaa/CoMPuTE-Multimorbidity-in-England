from stepmix.stepmix import StepMix
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--age_min', type=int, default=0, help='0/18/25/35/45/55/65/75/85')
    parser.add_argument('--age_max', type=int, default=17, help='17/24/34/44/54/64/74/84/120')
    parser.add_argument('--seed', type=int, default=61)
    args = parser.parse_args()

    age_group_i = [args.age_min, args.age_max]
    cond_list = ['AF', 'Anx', 'Ast', 'CKD', 'Can', 'CHD', 'COPD', 'Dem', 'Dep', 'Diab', 'HF', 'Hyp', 'Ost', 'PAD', 'Park', 'RA', 'StroTIA', 'SMI']
    cond_date_list = [x for x in cond_list]
    n_components = list(range(2, 15)) # Possible number of clusters
    results = {'n_components': [], 'aic': [], 'bic': [], 'caic': [], 'entropy': [], 'relative_entropy': [], 'll': [], 'converged': [], 'age_group': [], 'gender': []}

    for gender_x in ['M', 'F']:
        df_complete_i = pd.read_csv('./data/cond_vector/multcond_samples_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_' + gender_x + '.csv', header=0)  # Condition vectors for multimorbid population in this stratum
        df_i = df_complete_i[cond_date_list]
        n_divergence = 0
        for n_component_k in n_components:
            print('=*' * 20)
            print(age_group_i, gender_x, n_component_k)
            model = StepMix(n_components=n_component_k, measurement="binary", structural='binary', verbose=1, progress_bar=1, random_state=args.seed, max_iter=3000)
            model.fit(df_i)

            label_ik = model.predict(df_i)
            df_label_ik = pd.DataFrame(label_ik, columns=['label'])
            df_pat_label_ik = pd.concat((df_complete_i['patid'], df_label_ik), axis=1)
            df_pat_label_ik.to_csv('./results/lca_'+ str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_' + gender_x + '_' + str(n_component_k) + '_df_label.csv', index=False)  # Cluster assignment

            df_class_ik_distribution = pd.concat((df_i[cond_date_list], df_label_ik), axis=1)
            df_class_ik_distribution = df_class_ik_distribution.groupby('label')[cond_date_list].sum()
            df_class_ik_distribution = df_class_ik_distribution.transpose()  # Transpose to have 'label' as columns and conditions as row index
            df_class_ik_distribution.to_csv('./results/' + 'lca_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_' + gender_x + '_' + str(n_component_k) + '_df_cond_distribution.csv', index=True)  # Condition distribution within clusters

            results['n_components'].append(n_component_k)
            results['aic'].append(model.aic(df_i))
            results['bic'].append(model.bic(df_i))
            results['caic'].append(model.caic(df_i))
            results['entropy'].append(model.entropy(df_i))
            results['ll'].append(model.score(df_i))
            results['relative_entropy'].append(model.relative_entropy(df_i))
            results['converged'].append(model.converged_)
            results['age_group'].append(age_group_i)
            results['gender'].append(gender_x)
            print('convergence:', model.converged_)

            if not model.converged_:
                n_divergence += 1
            if n_divergence == 5:
                break
    df_results_ik = pd.DataFrame(results)
    df_results_ik.to_csv('./results/lca_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_df_scores.csv', index=False)  # BIC scores for each cluster number
