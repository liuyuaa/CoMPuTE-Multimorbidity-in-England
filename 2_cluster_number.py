import pandas as pd
import matplotlib.pyplot as plt

# Plot BIC curves along cluster number
age_group = [[0, 17], [18, 24], [25, 34], [35, 44], [45, 54], [55, 64], [65, 74], [75, 84], [85, 120]]
metric_list = ['aic', 'bic', 'caic', 'll', 'entropy', 'relative_entropy']
for gender_x in ['M', 'F']:
    for i, age_group_i in enumerate(age_group):
        df_scores = pd.read_csv('./results/lca_' + str(age_group_i[0]) + '_' + str(age_group_i[1]) + '_df_scores.csv', header=0)
        scores_lca = df_scores[df_scores['gender']==gender_x]
        plt.figure(figsize=(15, 6))
        for j, col_j in enumerate(metric_list):
            plt.subplot(2, 3, j + 1)
            plt.plot(scores_lca['n_components'], scores_lca[col_j], '-o', color='k')
            for k in range(scores_lca.shape[0]):
                color = 'k' if scores_lca['converged'].values[k] else 'r'
                plt.plot(scores_lca['n_components'].values[k], scores_lca[col_j].values[k], 'o-', color=color)
            plt.ylabel(col_j.upper(), fontsize=16)
            plt.xlabel('LCA: Cluster Number', fontsize=16)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            plt.title(f'Age Group:{str(age_group_i)}, Gender:{gender_x}', fontsize=18)
        plt.tight_layout()

# Set the age_gender2nk based on the plotted curves, to determine the optimal number of clusters for each stratum
# age_gender2nk = {'[0, 17]_M': 5, '[18, 24]_M': 5, '[25, 34]_M': 5, '[35, 44]_M': 6, '[45, 54]_M': 7, '[55, 64]_M': 7,
#                  '[65, 74]_M': 8, '[75, 84]_M': 8, '[85, 120]_M': 9,
#                  '[0, 17]_F': 4, '[18, 24]_F': 5, '[25, 34]_F': 5, '[35, 44]_F': 5, '[45, 54]_F': 7, '[55, 64]_F': 8,
#                  '[65, 74]_F': 8, '[75, 84]_F': 8, '[85, 120]_F': 8}
