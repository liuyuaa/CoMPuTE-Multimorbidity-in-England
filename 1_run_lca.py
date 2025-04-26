import os

for age_group in [[0, 17], [18, 24], [25, 34], [35, 44], [45, 54], [55, 64], [65, 74], [75, 84], [85, 120]]:
    age_min, age_max = age_group
    print('age_group_i:', age_group)
    command = (f"python 1_lca_profile.py --age_min {age_min} --age_max {age_max}")
    os.system(command)
    print('done\n\n\n')
