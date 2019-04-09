# This file will unzip, reformat and merge the three files downloaded from the French lottery's portal

import pandas as pd

# Load all three CSV
# Pandas does all the hard work of unzipping and parsing the CSV
df_2008 = pd.read_csv('raw_data/loto_200810.zip', sep=';')
df_2017 = pd.read_csv('raw_data/loto_201703.zip', sep=';')
df_2019 = pd.read_csv('raw_data/loto_201902.zip', sep=';')

# From 2008 until 20017 the prize categories had different names:
# old rang1 = 5 + 1 = new rang1
# old rang2 = 5 + 0 = new rang2
# old rang3 = 4 + 1 and 4 + 0 = new rang3 and rang4
# old rang4 = 3 + 1 and 3 + 0 = new rang5 and rang6
# old rang5 = 2 + 1 and 2 + 0 = new rang7 and rang8
# old rang6 = 1 + 1 and 0 + 1 = new rang9
# For simplicity, we'll only the three definitions that were kept stable overtime


def rename_and_cut(df, extra_columns):
    columns = {
        'date_de_tirage': 'date',
        'boule_1': 'ball_1',
        'boule_2': 'ball_2',
        'boule_3': 'ball_3',
        'boule_4': 'ball_4',
        'boule_5': 'ball_5',
        'numero_chance': 'lucky_ball',
        'nombre_de_gagnant_au_rang1': 'wins_5_1',
        'nombre_de_gagnant_au_rang2': 'wins_5_0'
    }
    columns.update(extra_columns)
    df = df.rename(columns=columns)
    return df[list(columns.values())]


df_2008 = rename_and_cut(df_2008, {
    'nombre_de_gagnant_au_rang6': 'wins_1_1_and_0_1'
})
df_2017 = rename_and_cut(df_2017, {
    'nombre_de_gagnant_au_rang9': 'wins_1_1_and_0_1'
})
df_2019 = rename_and_cut(df_2019, {
    'nombre_de_gagnant_au_rang9': 'wins_1_1_and_0_1'
})

df = pd.concat([df_2008, df_2017, df_2019])
path = 'data/data.csv'
df.to_csv(path, index=False)

print('Wrote {0} lines into {1}'.format(len(df), path))

print('Some stats for the fun of it:')
print(df.describe())
