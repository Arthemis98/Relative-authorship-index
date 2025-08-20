
import os
import time
import logging
import configparser
import pandas as pd
from pathlib import Path
import pybliometrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, mannwhitneyu, kruskal
import math
from itertools import combinations
import scikit_posthocs as sp
from sklearn.datasets import load_iris


# Set up the filepath
file_path = r"G:\Drive condivisi\fair index paper SCoSC\science of science\risultati\new_attempt\\authors_number_per_reference_all_authors.xlsx"
os.chdir(r"G:\Drive condivisi\fair index paper SCoSC\science of science\risultati\all_italian_papers")
OUTPUT_DIR = Path(r"G:\Drive condivisi\fair index paper SCoSC\science of science\risultati\new_attempt")

# loads the results excel file
df_all = pd.read_excel(file_path)
print("File loaded correctly")

# Deletes all the articles including more than 40 authors by checking the column "Number of Authors in Paper"
df_all = df_all[df_all["Number of Authors in Paper"] <= 40]
print("Filtered articles with over 40 authors")


# ==================== TOTAL INDEX PLOT ====================

###### HISTOGRAM
index_values = df_all["Index"].dropna()
index_values_list = index_values.tolist()
bin_width = 0.01
range_min = min(index_values_list)
range_max = max(index_values_list)
num_bins = math.ceil((range_max - range_min) / bin_width)

# stats
department_mean = 0.33541
percentile_10 = np.percentile(index_values_list, 10)
percentile_90 = np.percentile(index_values_list, 90)
media = np.mean(index_values_list)

plt.figure(figsize=(8, 6))
plt.hist(index_values_list, bins=num_bins, color='steelblue', edgecolor='steelblue', alpha=0.8)

# vertical lines
plt.axvline(media, color='green', linestyle='--', linewidth=2, label=f'Total Mean = {media:.3f}')
plt.axvline(department_mean, color='red', linestyle='--', linewidth=2, label=f'Department mean = {department_mean:.5f}')
plt.axvline(percentile_10, color='orange', linestyle='--', linewidth=2, label=f'10th percentile = {percentile_10:.3f}')
plt.axvline(percentile_90, color='orange', linestyle='--', linewidth=2, label=f'90th percentile = {percentile_90:.3f}')

# labels and caption
plt.legend()
plt.tight_layout()
plot_output_path = OUTPUT_DIR / "total_index.eps"
plt.savefig(plot_output_path,  format='eps')
plt.show()


###### TEMPORAL SERIES
publication_counts = {year: 0 for year in range(2001, 2024, 3)}
publication_indeces = {year: [] for year in range(2001, 2024, 3)}

for year in range(2001, 2024, 3):
    indeces = []
    
    for index, row in df_all.iterrows():
        if year == row['Publication Year']:
           publication_counts[year] += 1 
           if pd.notna(row['Index']):
              indeces.append(row['Index'])
              
    if len(indeces) > 0:
       indeces_mean = sum(indeces) / len(indeces)  
    else:
       indeces_mean = 0
       
    publication_indeces[year].append(indeces_mean)
 
year = [year for year in range(2001, 2024, 3)]
pub_indeces = [publication_indeces[year] for year in range(2001, 2024, 3)]
plt.plot(year, pub_indeces)
plt.scatter(year, pub_indeces)
plt.tight_layout()
plt.xticks(year)
allindeces_temporal_series_outputpath = OUTPUT_DIR / "allindex_temporal_series.eps"
plt.savefig(allindeces_temporal_series_outputpath, format='eps')
#Shows and saves the plot
plt.show()



### ANALYSIS AND PLOT BY REGION ###


# Group index values by region
group_index_region = df_all.groupby("Region")["Index"]
region_names = ['Northern Italy', 'Central Italy', 'Southern Italy and Islands']
index_data = [group_index_region.get_group(r).dropna().tolist() for r in region_names]
box_width = 0.3

# calculates means
means = [np.mean(vals) for vals in index_data]

# creates the boxplot
fig, ax = plt.subplots(figsize=(8, 6))

# Boxplot with orange median
box = ax.boxplot(
    index_data,
    widths=box_width,  
    patch_artist=False,  
    showmeans=True,  
    meanline=True,  
    notch=False,
)

# Add labels
ax.set_xticklabels(region_names)
ax.set_xlabel("Regione")
ax.set_ylabel("Index")
ax.set_title("Boxplot dell'Index per Regione\n(mediana = arancione, media = verde tratteggiata)")
plt.tight_layout()
plot_output_path = OUTPUT_DIR / "index_per_regione.eps"
plt.savefig(plot_output_path,  format='eps')
plt.show()



### ANALISI E PLOT PER TIPO DI UNIVERSITÀ (Public vs Private && Non-Telematic vs Telematic) ###


# Filtra solo le righe con "pubblica" o "privata" nel campo "University Type"
df_pubpriv = df_all[df_all["University Type"].isin(["public", "private"])]
df_privtele = df_all[df_all["Private Type"].isin(["non-telematic", "telematic"])]

# Raggruppa per University Type per il campo "Number of Authors in Paper"
group_index_type = df_pubpriv.groupby("University Type")["Index"]
mean_index_type = group_index_type.mean()
stderr_authors_type = group_index_type.sem()
box_width = 0.2
type_names = ['public', 'private']
index_data = [group_index_type.get_group(r).dropna().tolist() for r in type_names]

# calculates means
means = [np.mean(vals) for vals in index_data]

# creates boxplot
fig, ax = plt.subplots(figsize=(8, 6))

# Boxplot with orange median
box = ax.boxplot(
    index_data,
    widths=box_width,  
    patch_artist=False,  
    showmeans=True,  
    meanline=True,  
    notch=False,
)

# Add labels
ax.set_xticklabels(type_names)
ax.set_xlabel("Type")
ax.set_ylabel("Index")
ax.set_title("Boxplot of the Index per Type\n(median = orange, mean = dotted green)")
plt.tight_layout()
plot_output_path = OUTPUT_DIR / "index_per_type.eps"
plt.savefig(plot_output_path,  format='eps')
plt.show()





# Group by Private Type for the “Index” field
group_pv_index_type = group_index_region = df_privtele.groupby("Private Type")["Index"]
box_width=0.2
pvtype_names = ['non-telematic', 'telematic']
index_data = [group_pv_index_type.get_group(r).dropna().tolist() for r in pvtype_names]

# calculates means
means = [np.mean(vals) for vals in index_data]

# creates the boxplot
fig, ax = plt.subplots(figsize=(8, 6))

# Boxplot with orange median
box = ax.boxplot(
    index_data,
    widths=box_width,  
    patch_artist=False,  
    showmeans=True,  
    meanline=True,  
    notch=False,
)

# Add labels
ax.set_xticklabels(pvtype_names)
ax.set_xlabel("Type")
ax.set_ylabel("Index")
ax.set_title("Boxplot of the Index per Type\n(median = orange, mean = dotted green)")
plt.tight_layout()
plot_output_path = OUTPUT_DIR / "index_per_private_type.eps"
plt.savefig(plot_output_path,  format='eps')
plt.show()





### STATISTICAL ANALYSIS PUBLIC VS PRIVATE ###

print("\n=== Analysis: Number of authors per article (Public vs Private) ===")
# Estrai i dati per "Number of Authors in Paper"
authors_pub = df_pubpriv[df_pubpriv["University Type"]=="public"]["Number of Authors in Paper"].dropna()
authors_priv = df_pubpriv[df_pubpriv["University Type"]=="private"]["Number of Authors in Paper"].dropna()

print(f"Public: mean = {authors_pub.mean():.2f}, median = {authors_pub.median():.2f}, std = {authors_pub.std():.2f}")
print(f"Private: mean = {authors_priv.mean():.2f}, median = {authors_priv.median():.2f}, std = {authors_priv.std():.2f}")

u_stat, p_val = mannwhitneyu(authors_pub, authors_priv, alternative='two-sided')
print(f"Mann-Whitney U test (authors): U = {u_stat}, p-value = {p_val:.4f}")

print("\n=== Analysis: Index (Pubbliche vs Private) ===")
index_pub = df_pubpriv[df_pubpriv["University Type"]=="public"]["Index"].dropna()
index_priv = df_pubpriv[df_pubpriv["University Type"]=="private"]["Index"].dropna()

print(f"Public: mean = {index_pub.mean():.2f}, median = {index_pub.median():.2f}, std = {index_pub.std():.2f}")
print(f"Private: mean = {index_priv.mean():.2f}, median = {index_priv.median():.2f}, std = {index_priv.std():.2f}")

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



