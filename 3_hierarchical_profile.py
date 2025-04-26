import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Obtain age_gender2nk based on the plotted figures from 2_cluster_number.py
age_gender2nk = {'[0, 17]_M': 5, '[18, 24]_M': 5, '[25, 34]_M': 5, '[35, 44]_M': 6, '[45, 54]_M': 7, '[55, 64]_M': 7,
                 '[65, 74]_M': 8, '[75, 84]_M': 8, '[85, 120]_M': 9,
                 '[0, 17]_F': 4, '[18, 24]_F': 5, '[25, 34]_F': 5, '[35, 44]_F': 5, '[45, 54]_F': 7, '[55, 64]_F': 8,
                 '[65, 74]_F': 8, '[75, 84]_F': 8, '[85, 120]_F': 8}

age_group = [[0, 17], [18, 24], [25, 34], [35, 44], [45, 54], [55, 64], [65, 74], [75, 84], [85, 120]]
cond_date_list = ['AF', 'Anx', 'Ast', 'CKD', 'Can', 'CHD', 'COPD', 'Dem', 'Dep', 'Diab', 'HF', 'Hyp', 'Ost', 'PAD', 'Park', 'RA', 'StroTIA', 'SMI']
cond_date_list = sorted(cond_date_list)

# Calculate the condition prevalence and exclusivity in cluster
gender2cluster_dict = {}
for ix, gender_x in enumerate(['M', 'F']):
    age_clus2cond_prev, age_clus2cond_exclu = {}, {}
    for i, age_group_i in enumerate(age_group):
        num_cluser_ix = age_gender2nk[str(age_group_i) + '_' + gender_x]
        df_complete_i = pd.read_csv('./data/multcond_samples_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_' + gender_x + '.csv', header=0)
        n_pat = df_complete_i.shape[0]
        df_prev_i = df_complete_i[cond_date_list].sum(axis=0) / n_pat

        df_pat_label_ik = pd.read_csv('./results/lca_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_' + gender_x + '_' + str(num_cluser_ix) + '_df_label.csv', header=0)
        df_pat_label_ik.loc[:, 'label'] = df_pat_label_ik.loc[:, 'label'] + 1  # ----- cluster id from 1 -----
        cluster2num = dict(df_pat_label_ik['label'].value_counts().items())

        df_class_cond_dist = pd.read_csv('./results/lca_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_' + gender_x + '_' + str(num_cluser_ix) + '_df_cond_distribution.csv', header=0)
        df_class_cond_dist.set_index('Unnamed: 0', inplace=True)
        df_class_cond_dist.columns = [str(int(col) + 1) for col in df_class_cond_dist.columns]  # --- cluster id from 1 -----

        ## -------------------------- Normalize for each cluster for condition prevalence ---------------------
        df_cluster_norm = pd.DataFrame(None)
        for cls_name in range(1, num_cluser_ix + 1):
            df_cluster_norm[str(cls_name)] = df_class_cond_dist[str(cls_name)] / cluster2num[cls_name]
        df_cluster_norm = pd.concat([df_cluster_norm, df_prev_i.rename('Prev')], axis=1)

        ## -------------------------- Normalize for each condition for condition exclusivity ------------------
        df_cond_norm = df_class_cond_dist.div(df_class_cond_dist.sum(axis=1) + 1e-10, axis=0)
        df_cond_norm = pd.concat([df_cond_norm, df_class_cond_dist.sum(axis=1).rename('#Patient')], axis=1)
        last_column_values = df_cond_norm.iloc[:, -1].values.tolist()
        df_cond_norm = df_cond_norm.iloc[:, :-1]
        df_cond_norm.index.name = ''

        for col_j in range(1, num_cluser_ix + 1):
            age_clus2cond_prev[str(age_group_i) + '_' + str(col_j)] = df_cluster_norm[str(col_j)].values
        for col_j in range(1, num_cluser_ix + 1):
            age_clus2cond_exclu[str(age_group_i) + '_' + str(col_j)] = df_cond_norm[str(col_j)].values

    gender2cluster_dict[gender_x] = {'cond_prev': age_clus2cond_prev, 'cond_exclu': age_clus2cond_exclu}

# Hierarchical clustering for cluster merging to get profiles
for gender_x in ['M', 'F']:
    age_clus2cond_prev, age_clus2cond_exclu = gender2cluster_dict[gender_x]['cond_prev'], gender2cluster_dict[gender_x]['cond_exclu']
    cluster2vec = {}
    for k, v in age_clus2cond_prev.items():
        cluster2vec[k] = np.concatenate([v, age_clus2cond_exclu[k]])
    cluster_names = [x+'_'+gender_x for x in list(cluster2vec.keys())]
    cluster_vecs = np.array(list(cluster2vec.values()))

    linkage_matrix = linkage(cluster_vecs, method='ward')
    tick_fontsize = 14
    plt.figure(figsize=(20, 10))
    plt.title("Male", fontsize=20)
    plt.ylabel("Distance", fontsize=tick_fontsize, fontname='Arial')
    plt.xlabel("Cluster ID", fontsize=tick_fontsize, fontname='Arial')
    dendro = dendrogram(linkage_matrix, labels=cluster_names, leaf_rotation=90, leaf_font_size=tick_fontsize, orientation='top')
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)  # Adjust as needed
    xtick_labels = []
    for label in dendro['ivl']:
        tmp = label.split('_')
        age_group_i = eval(tmp[0])
        if age_group_i[0] == 0:
            xx_ag = '<18'
        elif age_group_i[0] == 85:
            xx_ag = '>84'
        else:
            xx_ag = f'{age_group_i[0]}-{age_group_i[1]}'
        xxx = xx_ag + ', C' + tmp[1]
        xtick_labels.append(xxx)
    plt.gca().set_xticklabels(xtick_labels, rotation=90, horizontalalignment='left', fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(f'./results/cluster_merge_{gender_x}.pdf', dpi=500, bbox_inches='tight')
    plt.show()