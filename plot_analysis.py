import os
import time
import logging
import configparser
import pandas as pd
from pathlib import Path
import pybliometrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, mannwhitneyu, kruskal, shapiro
import math
from itertools import combinations
import scikit_posthocs as sp
from sklearn.datasets import load_iris
import seaborn as sns
 

# Set up the filepath for data and the output results
file_path = r"G:\Drive condivisi\fair index paper SCoSC\science of science\risultati\prova finale 2\\authors_number_per_reference_all.xlsx"
os.chdir(r"G:\Drive condivisi\fair index paper SCoSC\science of science\risultati\prova finale 2")
OUTPUT_DIR = Path(r"G:\Drive condivisi\fair index paper SCoSC\science of science\risultati\prova finale 2")

# loads the results excel file
df_all = pd.read_excel(file_path)
print("File loaded correctly")

# Deletes all the articles including more than 40 authors by checking the column "Number of Authors in Paper"
df_all = df_all[df_all["Number of Authors in Paper"] <= 40]
print("Filtered articles with over 40 authors")


# ==================== TOTAL INDEX PLOT ====================

###### TIME SERIES

# Dati di esempio (sostituisci con i tuoi dati reali)
publication_counts = {year: 0 for year in range(2001, 2024, 3)}
publication_indeces = {year: [] for year in range(2001, 2024, 3)}
publication_indeces_sems = {year: [] for year in range(2001, 2024, 3)}

for year in range(2001, 2024, 3):
    indeces = []

    for index, row in df_all.iterrows():
        if year == row['Publication Year']:
           publication_counts[year] += 1
           if pd.notna(row['Index']):
              indeces.append(row['Index'])

    if len(indeces) > 0:
       n = len(indeces)
       indeces_mean = sum(indeces) / len(indeces)
       sems_indeces = np.std(indeces, ddof = 1) / np.sqrt(len(indeces))  # Errore standard
       publication_indeces[year].append(indeces_mean)
       publication_indeces_sems[year].append(sems_indeces)
    else:
       indeces_mean = 0

year = [year for year in range(2001, 2024, 3)]
pub_indeces = []
pub_indeces_sems = []
for y in range(2001, 2024, 3):
    if publication_indeces[y]:
        pub_indeces.append(publication_indeces[y][0])
    else:
        pub_indeces.append(0)
    if publication_indeces_sems[y]:
        pub_indeces_sems.append(publication_indeces_sems[y][0])
    else:
        pub_indeces_sems.append(0)

# Plotting
plt.figure(figsize=(10, 6))

# Aggiungi una linea continua
plt.plot(np.arange(len(year)), pub_indeces, color='black', linestyle='-', label='Tendenza')

# Aggiungi i punti e le barre di errore
for idx, y in enumerate(year):
    plt.scatter(idx, pub_indeces[idx], color='black', label='Dati' if idx == 0 else "")
    plt.errorbar(idx, pub_indeces[idx], yerr=pub_indeces_sems[idx],
                  fmt='none', ecolor='gray', capsize=5, label='Errore' if idx == 0 else "")

# Imposta le etichette sull'asse x
plt.xticks(np.arange(len(year)), year, rotation=45)

# Aggiungi etichette e titolo
plt.title('Serie Temporale con Barre di Errore')
plt.xlabel('Anno')
plt.ylabel('Indice di Pubblicazione')
plt.legend()
plt.tight_layout()

# Salva e mostra il grafico
allindeces_temporal_series_outputpath = OUTPUT_DIR / "allindex_temporal_series_new.eps"
plt.savefig(allindeces_temporal_series_outputpath, format='eps')
plt.show()




###### HISTOGRAM

# Dati
index_values = df_all["Index"].dropna()
index_values_list = index_values.tolist()
bin_width = 0.01
range_min = min(index_values_list)
range_max = max(index_values_list)
num_bins = math.ceil((range_max - range_min) / bin_width)

# Statistiche
department_mean = 0.33541
percentile_10 = np.percentile(index_values_list, 10)
percentile_90 = np.percentile(index_values_list, 90)
media = np.mean(index_values_list)

plt.figure(figsize=(8, 6))

# Istogramma con contorno nero e riempimento più chiaro
plt.hist(index_values_list, bins=num_bins, color='white', edgecolor='black', alpha=1)

# Linee verticali con diversi stili e colori
plt.axvline(media, color = (0, 0.2, 0), linestyle=':', linewidth=2, label=f'Total Mean = {media:.3f}')
plt.axvline(department_mean, color='red', linestyle='--', linewidth=2, label=f'Department mean = {department_mean:.5f}')
plt.axvline(percentile_10, color='purple', linestyle='-.', linewidth=2, label=f'10th percentile = {percentile_10:.3f}')
plt.axvline(percentile_90, color='orange', linestyle='-', linewidth=2, label=f'90th percentile = {percentile_90:.3f}')

# Etichette e legenda
plt.xlabel('Index')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()

# # Salva e mostra il grafico
plot_output_path = OUTPUT_DIR / "total_index.eps"
plt.savefig(plot_output_path, format='eps')
plt.show()




### ANALYSIS AND PLOT BY REGION ###

# Group index values by region
group_index_region = df_all.groupby("Region")["Index"]
region_names = ['Northern Italy', 'Central Italy', 'Southern Italy and Islands']
index_data = [group_index_region.get_group(r).dropna().tolist() for r in region_names]

# Create a DataFrame for seaborn
import pandas as pd
df_plot = pd.DataFrame({'Region': [], 'Index': []})
for region, values in zip(region_names, index_data):
    df_region = pd.DataFrame({'Region': [region] * len(values), 'Index': values})
    df_plot = pd.concat([df_plot, df_region], ignore_index=True)

# Create strip plot
plt.figure(figsize=(8, 6))
sns.stripplot(data=df_plot, x='Region', y='Index', jitter=True, color='black', size=5)

# Add labels
plt.xlabel("Regione")
plt.ylabel("Index")
plt.title("Distribuzione dell'Index per Regione")

plt.tight_layout()
plot_output_path = OUTPUT_DIR / "stripplot_index_per_regione.eps"
plt.savefig(plot_output_path, format='eps')
plt.show()


### ANALISI E PLOT PER TIPO DI UNIVERSITÀ (Public vs Private) ###

# Filtra solo le righe con "public" o "private" nel campo "University Type"
df_pubpriv = df_all[df_all["University Type"].isin(["public", "private"])]

# Crea un DataFrame per seaborn
df_plot = pd.DataFrame({'University Type': [], 'Index': []})
for type_name in ["public", "private"]:
    values = df_pubpriv[df_pubpriv["University Type"] == type_name]["Index"].dropna().tolist()
    df_type = pd.DataFrame({'University Type': [type_name] * len(values), 'Index': values})
    df_plot = pd.concat([df_plot, df_type], ignore_index=True)

# Crea strip plot
plt.figure(figsize=(8, 6))
sns.stripplot(data=df_plot, x='University Type', y='Index', jitter=True, color='black', size=5)

# Add labels
plt.xlabel("University Type")
plt.ylabel("Index")
plt.title("Distribuzione dell'Index per Tipo di Università")

plt.tight_layout()
plot_output_path = OUTPUT_DIR / "stripplot_index_per_unitype.eps"
plt.savefig(plot_output_path, format='eps')
plt.show()



### ANALISI E PLOT PER TIPO DI UNIVERSITÀ (Non-Telematic vs Telematic) ###

# Filtra solo le righe con "non-telematic" o "telematic" nel campo "Private Type"
df_privtele = df_all[df_all["Private Type"].isin(["non-telematic", "telematic"])]

# Crea un DataFrame per seaborn
df_plot = pd.DataFrame({'Private Type': [], 'Index': []})
for type_name in ["non-telematic", "telematic"]:
    values = df_privtele[df_privtele["Private Type"] == type_name]["Index"].dropna().tolist()
    df_type = pd.DataFrame({'Private Type': [type_name] * len(values), 'Index': values})
    df_plot = pd.concat([df_plot, df_type], ignore_index=True)

# Crea strip plot
plt.figure(figsize=(8, 6))
sns.stripplot(data=df_plot, x='Private Type', y='Index', jitter=True, color='black', size=5)

# Add labels
plt.xlabel("Private Type")
plt.ylabel("Index")
plt.title("Distribuzione dell'Index per Tipo di Università Telematica")

plt.tight_layout()
plot_output_path = OUTPUT_DIR / "stripplot_index_per_private_type.eps"
plt.savefig(plot_output_path, format='eps')
plt.show()


### ANALYSIS AND PLOT BY REGION ###


# Group index values by region
group_index_region = df_all.groupby("Region")["Index"]
region_names = ['Northern Italy', 'Central Italy', 'Southern Italy and Islands']
index_data = [group_index_region.get_group(r).dropna().tolist() for r in region_names]
box_width = 0.3

# calculates means
means = [np.mean(vals) for vals in index_data]
sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) for vals in index_data]




### ANALISI E PLOT PER TIPO DI UNIVERSITÀ (Public vs Private && Non-Telematic vs Telematic) ###


# Filtra solo le righe con "pubblica" o "privata" nel campo "University Type"
df_pubpriv = df_all[df_all["University Type"].isin(["public", "private"])]
df_privtele = df_all[df_all["Private Type"].isin(["non-telematic", "telematic"])]

# Raggruppa per University Type per il campo "Number of Authors in Paper"
group_index_type = df_pubpriv.groupby("University Type")["Index"]
mean_index_type = group_index_type.mean()
stderr_authors_type = group_index_type.sem()
type_names = ['public', 'private']
index_data = [group_index_type.get_group(r).dropna().tolist() for r in type_names]

# calculates means and std
means = [np.mean(vals) for vals in index_data]
sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) for vals in index_data]




### ANALISI E PLOT PER TIPO DI UNIVERSITÀ (Public vs Private && Non-Telematic vs Telematic) ###


# Group by Private Type for the “Index” field
group_pv_index_type = group_index_region = df_privtele.groupby("Private Type")["Index"]
box_width=0.2
pvtype_names = ['non-telematic', 'telematic']
index_data = [group_pv_index_type.get_group(r).dropna().tolist() for r in pvtype_names]

# calculates means
means = [np.mean(vals) for vals in index_data]
sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) for vals in index_data]




Original_Publications = 30
alfa = 4
beta = 0.01
w = 0
RAI = 0.403
RAI_field = 0.203
N_authors = 20
Adjusted_Publications = Original_Publications * math.exp(-alfa * (RAI- RAI_field))
Adjusted_Publications_2 = Original_Publications * math.exp(-alfa * (RAI- RAI_field)) * math.exp(-beta *((1-w)*(N_authors-1)))




Original_Publications = 11
alfa = 4
beta = 0.04
w = 0.8
RAI = 0.203
RAI_field = 0.203
N_authors = 6
Adjusted_Publications = Original_Publications * math.exp(-alfa * (RAI- RAI_field))
Adjusted_Publications_2 = Original_Publications * math.exp(-alfa * (RAI- RAI_field)) * math.exp(-beta *((1-w)*(N_authors-1)))






### STATISTICAL ANALYSIS PUBLIC VS PRIVATE ###

print("\n=== Analysis: Number of authors per article (Public vs Private) ===")
# Estrai i dati per "Number of Authors in Paper"
authors_pub = df_pubpriv[df_pubpriv["University Type"] == "public"]["Number of Authors in Paper"].dropna()
authors_priv = df_pubpriv[df_pubpriv["University Type"] == "private"]["Number of Authors in Paper"].dropna()

# Test di Shapiro-Wilk per "Number of Authors in Paper"
stat_pub, p_pub = shapiro(authors_pub)
stat_priv, p_priv = shapiro(authors_priv)

print(f"Public: mean = {authors_pub.mean():.2f}, median = {authors_pub.median():.2f}, std = {authors_pub.std():.2f}")
print(f"Shapiro-Wilk test (Public): statistic = {stat_pub:.4f}, p-value = {p_pub:.4f}")
print(f"Private: mean = {authors_priv.mean():.2f}, median = {authors_priv.median():.2f}, std = {authors_priv.std():.2f}")
print(f"Shapiro-Wilk test (Private): statistic = {stat_priv:.4f}, p-value = {p_priv:.4f}")

# Test di Mann-Whitney per "Number of Authors in Paper"
u_stat, p_val = mannwhitneyu(authors_pub, authors_priv, alternative='two-sided')
print(f"Mann-Whitney U test (authors): U = {u_stat}, p-value = {p_val:.4f}")

print("\n=== Analysis: Index (Public vs Private) ===")

# Estrai i dati per "Index"
index_pub = df_pubpriv[df_pubpriv["University Type"] == "public"]["Index"].dropna()
index_priv = df_pubpriv[df_pubpriv["University Type"] == "private"]["Index"].dropna()

# Test di Shapiro-Wilk per "Index"
stat_pub_idx, p_pub_idx = shapiro(index_pub)
stat_priv_idx, p_priv_idx = shapiro(index_priv)

print(f"Public: mean = {index_pub.mean():.2f}, median = {index_pub.median():.2f}, std = {index_pub.std():.2f}")
print(f"Shapiro-Wilk test (Public): statistic = {stat_pub_idx:.4f}, p-value = {p_pub_idx:.4f}")
print(f"Private: mean = {index_priv.mean():.2f}, median = {index_priv.median():.2f}, std = {index_priv.std():.2f}")
print(f"Shapiro-Wilk test (Private): statistic = {stat_priv_idx:.4f}, p-value = {p_priv_idx:.4f}")

# Test di Mann-Whitney per "Index"
u_stat_idx, p_val_idx = mannwhitneyu(index_pub, index_priv, alternative='two-sided')
print(f"Mann-Whitney U test (Index): U = {u_stat_idx}, p-value = {p_val_idx:.4f}")



### STATISTICAL ANALYSIS REGIONS COMPARISON ###

# Analisi per Regione: Numero di autori per articolo
region_labels = ['Northern Italy', 'Central Italy', 'Southern Italy and Islands']
authors_by_region = [df_all[df_all["Region"] == reg]["Number of Authors in Paper"].dropna() for reg in region_labels]

print("\n=== Analysis: Number of authors per article among regions ===")
for reg, data in zip(region_labels, authors_by_region):
    print(f"{reg}: mean = {data.mean():.2f}, median = {data.median():.2f}, std = {data.std():.2f}")

stat_kw, p_kw = kruskal(*authors_by_region)
print(f"Kruskal-Wallis test (authors per region): H = {stat_kw:.2f}, p-value = {p_kw:.4f}")
p_authors_regions = sp.posthoc_dunn(authors_by_region, p_adjust='bonferroni')

# Analisi per Regione: Index
index_by_region = [df_all[df_all["Region"] == reg]["Index"].dropna() for reg in region_labels]

print("\n=== Analysis: Index among regions ===")
for reg, data in zip(region_labels, index_by_region):
    print(f"{reg}: mean = {data.mean():.2f}, median = {data.median():.2f}, std = {data.std():.2f}")

stat_kw_idx, p_kw_idx = kruskal(*index_by_region)
print(f"Kruskal-Wallis test (Index per region): H = {stat_kw_idx:.2f}, p-value = {p_kw_idx:.4f}")
p_index_regions = sp.posthoc_dunn(index_by_region, p_adjust='bonferroni')
#print(f"Bonferroni correction, p-value = {p_index_regions:.4f}")



### STATISTICAL ANALYSIS NON-TELEMATIC VS TELEMATICHE ###

print("\n=== Analysis: Number of authors per article (Non-Telematic vs Telematic) ===")
# Estrai i dati per "Number of Authors in Paper"
authors_priv = df_privtele[df_privtele["Private Type"]=="non-telematic"]["Number of Authors in Paper"].dropna()
authors_tele = df_privtele[df_privtele["Private Type"]=="telematic"]["Number of Authors in Paper"].dropna()

print(f"Private: mean = {authors_priv.mean():.2f}, median = {authors_priv.median():.2f}, std = {authors_priv.std():.2f}")
print(f"Telematiche: mean = {authors_tele.mean():.2f}, median = {authors_tele.median():.2f}, std = {authors_tele.std():.2f}")

u_stat, p_val = mannwhitneyu(authors_priv, authors_tele, alternative='two-sided')
print(f"Mann-Whitney U test (authors): U = {u_stat}, p-value = {p_val:.4f}")

print("\n=== Analysis: Index (Non-Telematic vs Telematic) ===")
index_priv = df_privtele[df_privtele["Private Type"]=="non-telematic"]["Index"].dropna()
index_tele = df_privtele[df_privtele["Private Type"]=="telematic"]["Index"].dropna()

print(f"Private: mean = {index_priv.mean():.2f}, median = {index_priv.median():.2f}, std = {index_priv.std():.2f}")
print(f"Telematiche: mean = {index_tele.mean():.2f}, median = {index_tele.median():.2f}, std = {index_tele.std():.2f}")

u_stat_idx, p_val_idx = mannwhitneyu(index_priv, index_tele, alternative='two-sided')
print(f"Mann-Whitney U test (Index): U = {u_stat_idx}, p-value = {p_val_idx:.4f}")






