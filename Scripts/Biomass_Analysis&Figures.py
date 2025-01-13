import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams
from matplotlib import cm
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import ttest_ind

#linear regression of cytometric dataset
file_path = "/path/to/data/Moisture_and_cell_count.xlsx"
moisture_data = pd.read_excel(file_path)

cleaned_data = moisture_data.dropna(subset=['Average cell count (per gram of frozen feces)', 'Moisture content (%)'])

x = cleaned_data['Average cell count (per gram of frozen feces)']
X = sm.add_constant(x)
y = cleaned_data['Moisture content (%)']
model = sm.OLS(y, X).fit()

regression_summary = model.summary()
print(regression_summary)


#linear regression of Metacardis dataset
file_path = "/path/to/data/Supplemental_table9.xlsx" 
Metacardis_df = pd.read_excel(file_path, index_col = 0)

numeric_cols = Metacardis_df.select_dtypes(include=[np.number]).columns
Metacardis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
Metacardis_df.dropna(inplace=True)

Metacardis_df.loc[:, 'B:H_read_ratio'] = Metacardis_df['High quality clean read count'] / Metacardis_df['Homo sapiens GRCh37.p10 may2014 excluded read count']
Metacardis_df.loc[:, 'Log_B:H_read_ratio'] = np.log(Metacardis_df['B:H_read_ratio'] + 1e-9)

x = Metacardis_df['Log_B:H_read_ratio']
y = Metacardis_df['Microbial load']
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())


#Ordinal logistic regression of Gutpuzzle dataset
file_path = "/path/to/data/Gut_puzzle_data.xlsx" 
Gut_puzzle = pd.read_excel(file_path, index_col = 0)


#Figure2_generation

cleaned_data = moisture_data.dropna(subset=['Average cell count (per gram of frozen feces)', 'Moisture content (%)'])

x = cleaned_data['Average cell count (per gram of frozen feces)']
X = sm.add_constant(x)
y = cleaned_data['Moisture content (%)']

model = sm.OLS(y, X).fit()


# First subplot: Scatter plot
fig, axes = plt.subplots(2, 2, figsize=(7.09, 7))

font_label = {'fontsize': 7, 'color': 'black'}
font_ticks = {'fontsize': 7, 'color': 'black'}

x_pred = np.linspace(x.min(), x.max(), 100)
X_pred = sm.add_constant(x_pred)
predictions = model.get_prediction(X_pred)
predicted_means = predictions.predicted_mean
conf_int = predictions.conf_int()

axes[0, 0].scatter(cleaned_data['Average cell count (per gram of frozen feces)'],
                cleaned_data['Moisture content (%)'], alpha=0.5, s=30)
axes[0, 0].plot(x_pred, predicted_means, color='red', linewidth=1.2)
axes[0, 0].fill_between(x_pred, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label='95% CI')
axes[0, 0].set_xlabel('Average cell count (cells/g)', **font_label)
axes[0, 0].set_ylabel('Stool moisture content (%)', **font_label)
axes[0, 0].tick_params(axis='both', labelsize=font_ticks['fontsize'], colors=font_ticks['color'])
axes[0, 0].yaxis.set_tick_params(pad=-2.5)
axes[0, 0].text(0.97, 0.95, 'P < 0.001', transform=axes[0, 0].transAxes, fontsize=8, 
             verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
axes[0, 0].grid(False)

for spine in axes[0, 0].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)


x = Metacardis_df['Log_B:H_read_ratio']  
y = Metacardis_df['Microbial load']      

x_with_const = sm.add_constant(x)

model = sm.OLS(y, x_with_const).fit()

x_pred = np.linspace(x.min(), x.max(), 100)
x_pred_with_const = sm.add_constant(x_pred)
predictions = model.get_prediction(x_pred_with_const)
predicted_means = predictions.predicted_mean
conf_int = predictions.conf_int()

    
#Second subplot: Scatter plot
axes[0, 1].scatter(Metacardis_df['Log_B:H_read_ratio'], Metacardis_df['Microbial load'], 
                color='#006666', alpha=0.5, s=20)
axes[0, 1].plot(x_pred, predicted_means, color='red', linewidth=1.2, label='Regression line')
axes[0, 1].fill_between(x_pred, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label='95% CI')
axes[0, 1].set_xlabel('Log Bacterial-to-Host Read Ratio', **font_label)
axes[0, 1].set_ylabel('Average cell count (cells/g)', **font_label)
axes[0, 1].yaxis.set_tick_params(pad=-2.5)
axes[0, 1].text(0.97, 0.95, 'P = 0.005', transform=axes[0, 1].transAxes, fontsize=7, 
             verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
axes[0, 1].grid(False)

for spine in axes[0, 1].spines.values():
    spine.set_edgecolor('#4D4D4D')  # Set to dark grey for a more natural look
    spine.set_linewidth(0.7)

#Third plot: Box plot 
x = Gut_puzzle['Log_B:H_read_ratio']
y = Gut_puzzle['bristol']

colors = cm.Blues(np.linspace(0.9, 0.1, len(np.unique(y))))  # Adjust from light to dark blue

for i, val in enumerate(sorted(y.unique())):
    axes[1, 0].boxplot(x[y == val], vert=False, patch_artist=True, 
               boxprops=dict(facecolor=colors[i], color='black'),
               medianprops=dict(color='red'), 
               positions=[i + 1], widths=0.6)
    
for i, val in enumerate(sorted(y.unique()), start=1):
    axes[1, 0].plot(x[y == val], [i] * len(x[y == val]), 'o', color='black', alpha=0.7, markersize=3)

#Labels and title for the Box plot
axes[1, 0].set_xlabel('Log Bacterial-to-Host Read Ratio', **font_label)
axes[1, 0].set_ylabel('Bristol Stool Score', **font_label)
axes[1, 0].set_xticks(np.arange(7, 11, 0.5)) 
axes[1, 0].yaxis.set_tick_params(pad=-2.5)
axes[1, 0].text(0.97, 0.95, 'P = 0.441', transform=axes[1, 0].transAxes, fontsize=7, 
             verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
axes[1, 0].grid(False)

axes[0, 0].text(-0.15, 1.1, 'a', transform=axes[0, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[0, 1].text(-0.15, 1.1, 'b', transform=axes[0, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[1, 0].text(-0.15, 1.1, 'c', transform=axes[1, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')


for spine in axes[1, 0].spines.values():
    spine.set_edgecolor('#4D4D4D')  
    spine.set_linewidth(0.7)

axes[1, 1].axis('off')

plt.tight_layout()
plt.show()





#Ordinal logistic regression of human read fraction (Gutpuzzle)
Gut_puzzle['Total read'] = Gut_puzzle['Human Read Count'] + Gut_puzzle['Bacterial Read Count'] + Gut_puzzle['Cellular organisms'] + Gut_puzzle['Eukaryota'] + Gut_puzzle['Archaea'] + Gut_puzzle['Viruses']  
Gut_puzzle['Human read fraction'] = Gut_puzzle['Human Read Count']/Gut_puzzle['Total read']
Gut_puzzle['log_Human_read_fraction'] = np.log(Gut_puzzle['Human read fraction'] + 1e-9)

x = Gut_puzzle['log_Human_read_fraction']
Y = Gut_puzzle['bristol']  

model_fraction = OrderedModel(Y, x, distr='logit')
result_fraction = model_fraction.fit(method='bfgs')

print(result_fraction.summary())

#Supplementary Figure generation
x = Gut_puzzle['log_Human_read_fraction']
y = Gut_puzzle['bristol']

fig, ax = plt.subplots(figsize=(3.5, 3.5))

font_label = {'fontsize': 7, 'color': 'black'}
font_ticks = {'fontsize': 6, 'color': 'black'}

plt.rcParams.update({'xtick.labelsize': 6, 'ytick.labelsize': 6})

colors = cm.Greens(np.linspace(0.8, 0.1, len(y.unique())))

positions = range(1, len(y.unique()) + 1)  
for i, val in enumerate(sorted(y.unique())):
    ax.boxplot(
        x[y == val],
        positions=[positions[i]],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='red'),
        widths=0.6,
    )
    ax.plot(x[y == val], [positions[i]] * len(x[y == val]), 'o', color='black', alpha=0.7, markersize=3)

ax.set_xlabel('Log Human Read Fraction', fontsize=7)
ax.set_ylabel('Bristol Stool Score', fontsize=7)

step_size = 1.0  
min_val, max_val = np.floor(min(x)), np.ceil(max(x))  
ax.set_xticks(np.arange(min_val, max_val + step_size, step_size))

plt.grid(False)

for spine in ax.spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)

ax.text(0.02, 0.95, "P = 0.418", transform=ax.transAxes, fontsize=7, color='black', ha='left', va='top')

plt.show()




#linear regression of B:M read ratio and log copy number(qPCR)
cleaned_Mouse = Mouse_df.dropna(subset=['Copy no./ul', 'B:M_ratio'])

cleaned_Mouse['Log_B:M_ratio'] = np.log(cleaned_Mouse['B:M_ratio'] + 1e-9)
cleaned_Mouse['Log_Copy no./ul'] = np.log(cleaned_Mouse['Copy no./ul'] + 1e-9)


x = cleaned_Mouse['Log_B:M_ratio']
X = sm.add_constant(x)
y = cleaned_Mouse['Log_Copy no./ul']

model = sm.OLS(y, X).fit()

regression_summary = model.summary()
print(regression_summary)


#linear regression of B:M read ratio and log plant normalized total biomass
cleaned_Mouse_df['Log_Total_Biomass'] = np.log(cleaned_Mouse_df['Total_Biomass_plant2'] + 1e-9)

x = cleaned_Mouse_df['Log_B:M_ratio']
X = sm.add_constant(x)
y = cleaned_Mouse_df['Log_Total_Biomass']

model = sm.OLS(y, X).fit()

regression_summary = model.summary()
print(regression_summary)


#Linear regreseion of cow milk dataset
file_path = "/path/to/data/Cow_metagenomic_reads.xlsx" 
Cow_df = pd.read_excel(file_path)

Cow_df.loc[:, 'B:C_read_ratio'] = Cow_df['Bacterial_reads_excluding_spike-in_species'] / Cow_df['Bos_taurus_reads']
Cow_df.loc[:, 'Log_B:C_read_ratio'] = np.log(Cow_df['B:C_read_ratio'] + 1e-9)
Cow_df.loc[:, 'total_spike-in_reads'] = Cow_df['Imechtella_halotolerans_reads_(spike-in)'] + Cow_df['Allobacillus_haltolerans_reads_(spike-in)']
Cow_df.loc[:, "endogenous:spike-in_ratio"] = Cow_df['Bacterial_reads_excluding_spike-in_species'] / Cow_df['total_spike-in_reads']
Cow_df.loc[:, 'Log_endogenous:spike-in_ratio'] = np.log(Cow_df['endogenous:spike-in_ratio'] + 1e-9)

x = Cow_df['Log_B:C_read_ratio']
y = Cow_df['Log_endogenous:spike-in_ratio']
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

#Figure 3 generation
fig, ax = plt.subplots(2, 2, figsize=(7.09, 7))  

font_label = {'fontsize': 7, 'color': 'black'}
font_ticks = {'fontsize': 7, 'color': 'black'}

# Plot 1: Log Bacterial-to-Mouse Read Ratio vs Log qPCR Copy Number
x = cleaned_Mouse['Log_B:M_ratio']
X = sm.add_constant(x)
y = cleaned_Mouse['Log_Copy no./ul']
model = sm.OLS(y, X).fit()
x_pred = np.linspace(x.min(), x.max(), 100)
X_pred = sm.add_constant(x_pred)
predicted_means = model.get_prediction(X_pred).predicted_mean
conf_int = model.get_prediction(X_pred).conf_int()

ax[0, 0].scatter(x, y, color='#6BBE23', alpha=0.5, s=30)
ax[0, 0].plot(x_pred, predicted_means, color='red', linewidth=1.2, label='Regression line')
ax[0, 0].fill_between(x_pred, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label='95% CI')
ax[0, 0].text(0.05, 0.95, 'p < 0.001', transform=ax[0, 0].transAxes, fontsize=7, verticalalignment='top')
ax[0, 0].set_xlabel('Log Bacterial-to-Host Read Ratio', **font_label)
ax[0, 0].set_ylabel('Log qPCR Copy Number', **font_label)
ax[0, 0].text(-0.15, 1.1, 'a', transform=ax[0, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
ax[0, 0].grid(False)

for spine in ax[0, 0].spines.values():
    spine.set_edgecolor('#4D4D4D') 
    spine.set_linewidth(0.7) 

# Plot 2: Log Bacterial-to-Mouse Read Ratio vs Log Plant-Normalized Total Biomass
x2 = cleaned_Mouse_df['Log_B:M_ratio']
X2 = sm.add_constant(x2)
y2 = cleaned_Mouse_df['Log_Total_Biomass']
model2 = sm.OLS(y2, X2).fit()
x_pred2 = np.linspace(x2.min(), x2.max(), 100)
X_pred2 = sm.add_constant(x_pred2)
predicted_means2 = model2.get_prediction(X_pred2).predicted_mean
conf_int2 = model2.get_prediction(X_pred2).conf_int()

ax[0, 1].scatter(x2, y2, color='#6E4A31', alpha=0.5, s=30)
ax[0, 1].plot(x_pred2, predicted_means2, color='red', linewidth=1.2, label='Regression line')
ax[0, 1].fill_between(x_pred2, conf_int2[:, 0], conf_int2[:, 1], color='gray', alpha=0.2, label='95% CI')
ax[0, 1].text(0.05, 0.95, 'p < 0.001', transform=ax[0, 1].transAxes, fontsize=7, verticalalignment='top')
ax[0, 1].set_xlabel('Log Bacterial-to-Host Read Ratio', **font_label)
ax[0, 1].set_ylabel('Log Plant-Normalized Total Biomass', **font_label)
ax[0, 1].text(-0.15, 1.1, 'b', transform=ax[0, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
ax[0, 1].grid(False)

for spine in ax[0, 1].spines.values():
    spine.set_edgecolor('#4D4D4D')  
    spine.set_linewidth(0.7) 

# Plot 3: Log Bacterial Spike-In Ratio vs Endogenous Bacteria-to-Spike-in Read Ratio
x3 = Cow_df['Log_B:C_read_ratio']
y3 = Cow_df['Log_endogenous:spike-in_ratio']
model3 = sm.OLS(y3, sm.add_constant(x3)).fit()
x_pred3 = np.linspace(x3.min(), x3.max(), 100)
predicted_means3 = model3.get_prediction(sm.add_constant(x_pred3)).predicted_mean
conf_int3 = model3.get_prediction(sm.add_constant(x_pred3)).conf_int()

ax[1, 0].scatter(x3, y3, color='#CC5500', alpha=0.5, s=30)
ax[1, 0].plot(x_pred3, predicted_means3, color='red', linewidth=1.2, label='Regression line')
ax[1, 0].fill_between(x_pred3, conf_int3[:, 0], conf_int3[:, 1], color='gray', alpha=0.2, label='95% CI')
ax[1, 0].text(0.05, 0.95, 'p < 0.001', transform=ax[1, 0].transAxes, fontsize=7, verticalalignment='top')
ax[1, 0].set_xlabel('Log Endogenous Bacterial-to-Host Read Ratio', **font_label)
ax[1, 0].set_ylabel('Log Endogenous Bacterial-to-Spike-in Read Ratio', **font_label)
ax[1, 0].text(-0.15, 1.1, 'c', transform=ax[1, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
ax[1, 0].grid(False)

for spine in ax[1, 0].spines.values():
    spine.set_edgecolor('#4D4D4D')  
    spine.set_linewidth(0.7) 

ax[1, 1].axis('off')

#plt.savefig('figure3.pdf', dpi=300, bbox_inches='tight', format='pdf')

plt.tight_layout()
plt.show()



#Dense time-series plot of Bioml dataset (B:H)
file_path = "/path/to/data/Bioml_reads.xlsx" 
Bioml_df = pd.read_excel(file_path)

cleaned_Bioml_df = Bioml_df.dropna(subset=['Bacteria read','Human read', 'B:H ratio', 'Eukaryota', 'Archaea', 'Total reads'])
cleaned_Bioml_df.columns = cleaned_Bioml_df.columns.str.strip()
cleaned_Bioml_df.loc[:, 'Log_B:H_ratio'] = np.log(cleaned_Bioml_df['B:H ratio'] + 1e-9)

colors = {
    'am': (0, 0, 1),      # Blue
    'ae': (1, 0.5, 0),    # Orange
    'an': (0, 0.5, 0),    # Green
    'ao': (1, 0, 0)       # Red
}


plt.figure(figsize=(14, 8))

for individual in ['am', 'ao', 'an', 'ae']:
    individual_data = cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]
    plt.plot(individual_data['Days'], individual_data['Log_B:H_ratio'], 
             label=individual, alpha=0.8, color=colors[individual])

plt.xlabel('Days', fontsize=14)
plt.ylabel('Log B:H Ratio', fontsize=14)
plt.title('Time Series of Log B:H Ratio by Individual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Individual ID')
plt.grid(False)
plt.tight_layout()
plt.show()

#Dense time-series plot of Bioml dataset (B:Total)
cleaned_Bioml_df = Bioml_df.dropna(subset=['Bacteria read','Human read', 'B:H ratio', 'Eukaryota', 'Archaea', 'Total reads'])
cleaned_Bioml_df.columns = cleaned_Bioml_df.columns.str.strip()
cleaned_Bioml_df.loc[:, 'B:T ratio'] = cleaned_Bioml_df['Bacteria read']/cleaned_Bioml_df['Total reads']
cleaned_Bioml_df.loc[:, 'Log_B:T_ratio'] = np.log(cleaned_Bioml_df['B:T ratio'] + 1e-9)


colors = {
    'am': (0, 0, 1),      # Blue
    'ae': (1, 0.5, 0),    # Orange
    'an': (0, 0.5, 0),    # Green
    'ao': (1, 0, 0)       # Red
}

plt.figure(figsize=(14, 8))

for individual in ['am', 'ao', 'an', 'ae']:
    individual_data = cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]
    plt.plot(individual_data['Days'], individual_data['Log_B:T_ratio'], 
             label=individual, alpha=0.8, color=colors[individual])

plt.xlabel('Days', fontsize=14)
plt.ylabel('Log B:T Ratio', fontsize=14)
plt.title('Time Series of Log Bacterial-to-total read Ratios by Individual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Individual ID')
plt.grid(False)
plt.tight_layout()
plt.show()

#Dense time-series plot of Bioml dataset (H:Total)
cleaned_Bioml_df = Bioml_df.dropna(subset=['Bacteria read','Human read', 'B:H ratio', 'Eukaryota', 'Archaea', 'Total reads'])
cleaned_Bioml_df.columns = cleaned_Bioml_df.columns.str.strip()
cleaned_Bioml_df.loc[:, 'H:T ratio'] = cleaned_Bioml_df['Human read']/cleaned_Bioml_df['Total reads']
cleaned_Bioml_df.loc[:, 'Log_H:T_ratio'] = np.log(cleaned_Bioml_df['H:T ratio'] + 1e-9)


colors = {
    'am': (0, 0, 1),      # Blue
    'ae': (1, 0.5, 0),    # Orange
    'an': (0, 0.5, 0),    # Green
    'ao': (1, 0, 0)       # Red
}


plt.figure(figsize=(14, 8))

for individual in ['am', 'ao', 'an', 'ae']:
    individual_data = cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]
    plt.plot(individual_data['Days'], individual_data['Log_H:T_ratio'], 
             label=individual, alpha=0.8, color=colors[individual])

plt.xlabel('Days', fontsize=14)
plt.ylabel('Log H:T Ratio', fontsize=14)
plt.title('Time Series of Log Human-to-total read Ratios by Individual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Individual ID')
plt.grid(False)
plt.tight_layout()
plt.show()

#Boxplot of B:H (Bioml)
individuals = ['am', 'ao', 'an', 'ae']
colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0)]
median_color = 'yellow'

plt.figure(figsize=(10, 6))
log_bh_data = [cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]['Log_B:H_ratio'] for individual in individuals]
box = plt.boxplot(log_bh_data, labels=individuals, patch_artist=True,
                  medianprops=dict(color=median_color))

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

y_lines = {
    ('am', 'ae'): 11.4,
    ('am', 'an'): 10.9,
    ('ae', 'an'): 10.3,
    ('ae', 'ao'): 7.5,  
    ('an', 'ao'): 7.8   
}

significance_labels = {
    ('am', 'ae'): '***',
    ('am', 'an'): '***',
    ('ae', 'an'): '***',
    ('ae', 'ao'): '***',
    ('an', 'ao'): '***'
}


for (ind1, ind2), y in y_lines.items():
    x1, x2 = individuals.index(ind1) + 1, individuals.index(ind2) + 1
    if (ind1, ind2) == ('ae', 'ao'):
        plt.plot([x1, x1, x2, x2], [y + 0.1, y, y, y + 0.1], lw=1.5, c='k')
        plt.text((x1 + x2) * 0.5, y - 0.1, significance_labels[(ind1, ind2)], ha='center', va='top', color='black', fontsize=12)
    elif (ind1, ind2) == ('an', 'ao'):
        plt.plot([x1, x1, x2, x2], [y + 0.1, y, y, y + 0.1], lw=1.5, c='k')
        plt.text((x1 + x2) * 0.5, y - 0.1, significance_labels[(ind1, ind2)], ha='center', va='top', color='black', fontsize=12)
    else:
        plt.plot([x1, x1, x2, x2], [y - 0.1, y, y, y - 0.1], lw=1.5, c='k')
        plt.text((x1 + x2) * 0.5, y, significance_labels[(ind1, ind2)], ha='center', va='bottom', color='black', fontsize=12)


plt.title("Boxplot of Log B:H Ratios for Each Individual")
plt.xlabel("Individual ID")
plt.ylabel("Log B:H Ratio")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)

plt.show()


#T-tesst pairwise comprison (B:H)
ttest_results = {}

individual_pairs = [('am', 'ae'), ('am', 'an'), ('am', 'ao'), ('ae', 'an'), ('ae', 'ao'), ('an', 'ao')]

for ind1, ind2 in individual_pairs:
    data1 = cleaned_Bioml_df[cleaned_Bioml_df['id'] == ind1]['Log_B:H_ratio']
    data2 = cleaned_Bioml_df[cleaned_Bioml_df['id'] == ind2]['Log_B:H_ratio']
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    
    if p_value < 0.000166:
        significance = '***'
    elif p_value < 0.00166:
        significance = '**'
    elif p_value < 0.0083:
        significance = '*'
    else:
        significance = None 
    
    result = {'t_stat': t_stat, 'p_value': p_value}
    if significance:
        result['significance'] = significance  
    ttest_results[(ind1, ind2)] = result

for pair, result in ttest_results.items():
    print(f"Comparison between {pair[0]} and {pair[1]}:")
    significance_display = f", significance = {result['significance']}" if 'significance' in result else ""
    print(f" t-statistic = {result['t_stat']:.4f}, p-value = {result['p_value']:.4e}{significance_display}")

#Boxplot of B:T (Bioml)
individuals = ['am', 'ao', 'an', 'ae']
colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0)]
median_color = 'yellow'

plt.figure(figsize=(10, 6))
log_bh_data = [cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]['Log_B:T_ratio'] for individual in individuals]
box = plt.boxplot(log_bh_data, labels=individuals, patch_artist=True,
                  medianprops=dict(color=median_color))

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

y_lines = {
    ('am', 'ae'): 0.00031,
    ('am', 'an'): 0.00024,  
    ('ao', 'an'): 0.00010,
    ('ao', 'ae'): 0.00038   
}

significance_labels = {
    ('am', 'ae'): '***',
    ('am', 'an'): '***',
    ('ao', 'an'): '***',
    ('ao', 'ae'): '***'
}

for (ind1, ind2), y in y_lines.items():
    x1, x2 = individuals.index(ind1) + 1, individuals.index(ind2) + 1
    plt.plot([x1, x1, x2, x2], [y - 0.00001, y, y, y - 0.00001], lw=1, c='k')
    significance = significance_labels[(ind1, ind2)]
    plt.text((x1 + x2) * 0.5, y, significance, ha='center', va='bottom', color='black', fontsize=12)


plt.ylim(-0.0007, 0.0005)

plt.title("Boxplot of Log Bacterial-to-Total Read Ratios for Each Individual")
plt.xlabel("Individual ID")
plt.ylabel("Log B:T Ratio")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)

plt.show()

#T-tesst pairwise comprison (B:T)
ttest_results = {}

individual_pairs = [('am', 'ae'), ('am', 'an'), ('am', 'ao'), ('ae', 'an'), ('ae', 'ao'), ('an', 'ao')]

num_tests = len(individual_pairs)

for ind1, ind2 in individual_pairs:
    data1 = cleaned_Bioml_df[cleaned_Bioml_df['id'] == ind1]['Log_B:T_ratio']
    data2 = cleaned_Bioml_df[cleaned_Bioml_df['id'] == ind2]['Log_B:T_ratio']
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    
    corrected_p_value = p_value * num_tests  # Multiply by the number of tests

    if corrected_p_value < 0.000166:
        significance = '***'
    elif corrected_p_value < 0.00166:
        significance = '**'
    elif corrected_p_value < 0.0083:
        significance = '*'
    else:
        significance = None  

    result = {'t_stat': t_stat, 'p_value': p_value, 'corrected_p_value': corrected_p_value}
    if significance:
        result['significance'] = significance 
    ttest_results[(ind1, ind2)] = result

for pair, result in ttest_results.items():
    print(f"Comparison between {pair[0]} and {pair[1]}:")
    significance_display = f", significance = {result['significance']}" if 'significance' in result else ""
    print(f" t-statistic = {result['t_stat']:.4f}, p-value = {result['p_value']:.4e}, corrected p-value = {result['corrected_p_value']:.4e}{significance_display}")

#Boxplot of H:T (Bioml)
individuals = ['am', 'ao', 'an', 'ae']
colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0)]
median_color = 'yellow'

plt.figure(figsize=(10, 6))
log_ht_data = [cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]['Log_H:T_ratio'] for individual in individuals]
box = plt.boxplot(log_ht_data, labels=individuals, patch_artist=True,
                  medianprops=dict(color=median_color))

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

y_lines = {
    ('am', 'ae'): -7.3,
    ('am', 'an'): -7.6,
    ('ao', 'an'): -11.2,  
    ('an', 'ae'): -7.9,
    ('ao', 'ae'): -11.6  
}


significance_labels = {
    ('am', 'ae'): '***',
    ('am', 'an'): '***',
    ('ao', 'an'): '***',
    ('an', 'ae'): '***',
    ('ao', 'ae'): '***'
}

for (ind1, ind2), y in y_lines.items():
    x1, x2 = individuals.index(ind1) + 1, individuals.index(ind2) + 1
    if (ind1, ind2) == ('ao', 'an') or (ind1, ind2) == ('ao', 'ae'):
        plt.plot([x1, x1, x2, x2], [y + 0.1, y, y, y + 0.1], lw=1, c='k')
        plt.text((x1 + x2) * 0.5, y - 0.1, significance_labels[(ind1, ind2)], ha='center', va='top', color='black', fontsize=12)
    else:
        plt.plot([x1, x1, x2, x2], [y - 0.1, y, y, y - 0.1], lw=1, c='k')
        plt.text((x1 + x2) * 0.5, y, significance_labels[(ind1, ind2)], ha='center', va='bottom', color='black', fontsize=12)

plt.title("Boxplot of Log Human-to-Total Read Ratios for Each Individual")
plt.xlabel("Individual ID")
plt.ylabel("Log H:T Ratio")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)

plt.ylim(-12, -7)

plt.show()


#T-tesst pairwise comprison (H:T)
ttest_results = {}

individual_pairs = [('am', 'ae'), ('am', 'an'), ('am', 'ao'), ('ae', 'an'), ('ae', 'ao'), ('an', 'ao')]

num_tests = len(individual_pairs)

for ind1, ind2 in individual_pairs:
    data1 = cleaned_Bioml_df[cleaned_Bioml_df['id'] == ind1]['Log_H:T_ratio']
    data2 = cleaned_Bioml_df[cleaned_Bioml_df['id'] == ind2]['Log_H:T_ratio']
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    
    corrected_p_value = p_value * num_tests  

    if corrected_p_value < 0.000166:
        significance = '***'
    elif corrected_p_value < 0.00166:
        significance = '**'
    elif corrected_p_value < 0.0083:
        significance = '*'
    else:
        significance = None  

    result = {'t_stat': t_stat, 'p_value': p_value, 'corrected_p_value': corrected_p_value}
    if significance:
        result['significance'] = significance  
    ttest_results[(ind1, ind2)] = result

for pair, result in ttest_results.items():
    print(f"Comparison between {pair[0]} and {pair[1]}:")
    significance_display = f", significance = {result['significance']}" if 'significance' in result else ""
    print(f" t-statistic = {result['t_stat']:.4f}, p-value = {result['p_value']:.4e}, corrected p-value = {result['corrected_p_value']:.4e}{significance_display}")


#Figure 4 generation

fig, axes = plt.subplots(3, 2, figsize=(7.2, 7))

font_label = {'fontsize': 7, 'color': 'black'}
font_ticks = {'fontsize': 7, 'color': 'black'}

plt.rcParams.update({'xtick.labelsize': 7, 'ytick.labelsize': 7})
    
    
cleaned_Bioml_df = Bioml_df.dropna(subset=['Human read', 'B:H ratio'])
cleaned_Bioml_df.columns = cleaned_Bioml_df.columns.str.strip()
cleaned_Bioml_df['Log_B:H_ratio'] = np.log(cleaned_Bioml_df['B:H ratio'] + 1e-9)

colors = {'am': (0, 0, 1), 'ae': (1, 0.5, 0), 'an': (0, 0.5, 0), 'ao': (1, 0, 0)}


for individual in ['am', 'ao', 'an', 'ae']:
    individual_data = cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]
    axes[0, 0].plot(individual_data['Days'], individual_data['Log_B:H_ratio'], 
                    label=individual, alpha=0.8, color=colors[individual])
     
    
axes[0, 0].set_xlabel('Time point (Day)', **font_label)
axes[0, 0].set_ylabel('Log B:H Ratio', **font_label)
axes[0, 0].legend(title='Individual ID', fontsize=4.5, title_fontsize=4.5)
axes[0, 0].grid(False)


for spine in axes[0, 0].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)

    

individuals = ['am', 'ao', 'an', 'ae']
box_colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0)]
median_color = 'yellow'

log_bh_data = [cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]['Log_B:H_ratio'] for individual in individuals]
box = axes[0, 1].boxplot(log_bh_data, labels=individuals, patch_artist=True,  
                         medianprops=dict(color=median_color))


for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

#y-coordinates for each pair comparison line
y_lines = {
    ('am', 'ae'): 11.5,
    ('am', 'an'): 10.9,
    ('ae', 'an'): 10.5,
    ('ae', 'ao'): 7.4,  
    ('an', 'ao'): 7.8   
}

# Significance levels for each pair based on Bonferroni-corrected p-values
significance_labels = {
    ('am', 'ae'): '***',
    ('am', 'an'): '***',
    ('ae', 'an'): '***',
    ('ae', 'ao'): '***',
    ('an', 'ao'): '***'
}
    

for (ind1, ind2), y in y_lines.items():
    x1, x2 = individuals.index(ind1) + 1, individuals.index(ind2) + 1
    if (ind1, ind2) == ('ae', 'ao'):
        axes[0, 1].plot([x1, x1, x2, x2], [y + 0.1, y, y, y + 0.1], lw=0.8, c='k')
        axes[0, 1].text((x1 + x2) * 0.5, y - 0.09, significance_labels[(ind1, ind2)], ha='center', va='top', color='black', fontsize=12)
    elif (ind1, ind2) == ('an', 'ao'):
        axes[0, 1].plot([x1, x1, x2, x2], [y + 0.1, y, y, y + 0.1], lw=0.8, c='k')
        axes[0, 1].text((x1 + x2) * 0.5, y - 0.09, significance_labels[(ind1, ind2)], ha='center', va='top', color='black', fontsize=12)
    else:
        axes[0, 1].plot([x1, x1, x2, x2], [y - 0.1, y, y, y - 0.1], lw=0.8, c='k')
        axes[0, 1].text((x1 + x2) * 0.5, y - 0.21, significance_labels[(ind1, ind2)], ha='center', va='bottom', color='black', fontsize=12)

axes[0, 1].set_ylim(7, 12)         
    

axes[0, 1].set_xlabel("Individual ID", **font_label)
axes[0, 1].set_ylabel("Log B:H Ratio", **font_label)
axes[0, 1].grid(False)


for spine in axes[0, 1].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)


    
cleaned_Bioml_df = Bioml_df.dropna(subset=['Bacteria read','Human read', 'B:H ratio', 'Eukaryota', 'Archaea', 'Total reads'])
cleaned_Bioml_df.columns = cleaned_Bioml_df.columns.str.strip()
cleaned_Bioml_df.loc[:, 'H:T ratio'] = cleaned_Bioml_df['Human read']/cleaned_Bioml_df['Total reads']
cleaned_Bioml_df.loc[:, 'Log_H:T_ratio'] = np.log(cleaned_Bioml_df['H:T ratio'] + 1e-9)


colors = {
    'am': (0, 0, 1),    
    'ae': (1, 0.5, 0),    
    'an': (0, 0.5, 0),    
    'ao': (1, 0, 0)       
}


for individual in ['am', 'ao', 'an', 'ae']:
    individual_data = cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]
    axes[1, 0].plot(individual_data['Days'], individual_data['Log_H:T_ratio'], 
             label=individual, alpha=0.8, color=colors[individual])

axes[1, 0].set_xlabel('Time Point (Day)', **font_label)
axes[1, 0].set_ylabel('Log H:Total Ratio', **font_label)
axes[1, 0].legend(title='Individual ID', fontsize=4.5, title_fontsize=4.5)
axes[1, 0].grid(False)
    
for spine in axes[1, 0].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)     

    
individuals = ['am', 'ao', 'an', 'ae']
colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0)]
median_color = 'yellow'

plt.figure(figsize=(10, 6))
log_ht_data = [cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]['Log_H:T_ratio'] for individual in individuals]
box = axes[1, 1].boxplot(log_ht_data, labels=individuals, patch_artist=True,
                  medianprops=dict(color=median_color))

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


y_lines = {
    ('am', 'ae'): -7.0,
    ('am', 'an'): -7.5,
    ('ao', 'an'): -11.2,  
    ('an', 'ae'): -7.8,
    ('ao', 'ae'): -11.7  
}


significance_labels = {
    ('am', 'ae'): '***',
    ('am', 'an'): '***',
    ('ao', 'an'): '**',
    ('an', 'ae'): '***',
    ('ao', 'ae'): '***'
}


for (ind1, ind2), y in y_lines.items():
    x1, x2 = individuals.index(ind1) + 1, individuals.index(ind2) + 1
    if (ind1, ind2) == ('ao', 'an') or (ind1, ind2) == ('ao', 'ae'):
        axes[1, 1].plot([x1, x1, x2, x2], [y + 0.1, y, y, y + 0.1], lw=0.8, c='k')
        axes[1, 1].text((x1 + x2) * 0.5, y - 0.09, significance_labels[(ind1, ind2)], ha='center', va='top', color='black', fontsize=12)
    else:
        axes[1, 1].plot([x1, x1, x2, x2], [y - 0.1, y, y, y - 0.1], lw=0.8, c='k')
        axes[1, 1].text((x1 + x2) * 0.5, y - 0.24, significance_labels[(ind1, ind2)], ha='center', va='bottom', color='black', fontsize=12)

axes[1, 1].set_ylim(-12.2, -6.5)
        

axes[1, 1].set_xlabel("Individual ID", **font_label)
axes[1, 1].set_ylabel("Log H:Total Ratio", **font_label)
axes[1, 1].grid(False)


for spine in axes[1, 1].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7) 

    
    
cleaned_Bioml_df = Bioml_df.dropna(subset=['Bacteria read','Human read', 'B:H ratio', 'Eukaryota', 'Archaea', 'Total reads'])
cleaned_Bioml_df.columns = cleaned_Bioml_df.columns.str.strip()
cleaned_Bioml_df.loc[:, 'B:T ratio'] = cleaned_Bioml_df['Bacteria read']/cleaned_Bioml_df['Total reads']
cleaned_Bioml_df.loc[:, 'Log_B:T_ratio'] = np.log(cleaned_Bioml_df['B:T ratio'] + 1e-9)


colors = {
    'am': (0, 0, 1),      
    'ae': (1, 0.5, 0),    
    'an': (0, 0.5, 0),   
    'ao': (1, 0, 0)       
}


for individual in ['am', 'ao', 'an', 'ae']:
    individual_data = cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]
    axes[2, 0].plot(individual_data['Days'], individual_data['Log_B:T_ratio'], 
             label=individual, alpha=0.8, color=colors[individual])

axes[2, 0].set_xlabel('Time Point (Day)', **font_label)
axes[2, 0].set_ylabel('Log B:Total Ratio', **font_label)
axes[2, 0].legend(title='Individual ID', fontsize=4.5, title_fontsize=4.5)
axes[2, 0].grid(False)    


for spine in axes[2, 0].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7) 

    
    
individuals = ['am', 'ao', 'an', 'ae']
colors = [(0, 0, 1), (1, 0, 0), (0, 0.5, 0), (1, 0.5, 0)]
median_color = 'yellow'


plt.figure(figsize=(10, 6))
log_bh_data = [cleaned_Bioml_df[cleaned_Bioml_df['id'] == individual]['Log_B:T_ratio'] for individual in individuals]
box = axes[2, 1].boxplot(log_bh_data, labels=individuals, patch_artist=True,
                  medianprops=dict(color=median_color))


for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


y_lines = {
    ('am', 'ae'): 0.00022,
    ('am', 'an'): 0.00011,
    ('ao', 'an'): 0.00001,
    ('ao', 'ae'): 0.00032    
}


significance_labels = {
    ('am', 'ae'): '***',
    ('am', 'an'): '***',
    ('ao', 'an'): '***',
    ('ao', 'ae'): '***'
}

for (ind1, ind2), y in y_lines.items():
    if (ind1, ind2) not in significance_labels or significance_labels[(ind1, ind2)] == '':
        continue  
    
    x1, x2 = individuals.index(ind1) + 1, individuals.index(ind2) + 1
    axes[2, 1].plot([x1, x1, x2, x2], [y - 0.00002, y, y, y - 0.00002], lw=0.8, c='k')
    significance = significance_labels[(ind1, ind2)]
    axes[2, 1].text((x1 + x2) * 0.5, y - 0.00005, significance, ha='center', va='bottom', color='black', fontsize=12)

axes[2, 1].set_ylim(-0.00065, 0.0004)


axes[2, 1].set_xlabel("Individual ID", **font_label)
axes[2, 1].set_ylabel("Log B:Total Ratio", **font_label)
axes[2, 1].grid(False)


for spine in axes[2, 1].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)
    

axes[0, 0].text(-0.15, 1.1, 'a', transform=axes[0, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[0, 1].text(-0.15, 1.1, 'b', transform=axes[0, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[1, 0].text(-0.15, 1.1, 'c', transform=axes[1, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[1, 1].text(-0.15, 1.1, 'd', transform=axes[1, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[2, 0].text(-0.15, 1.1, 'e', transform=axes[2, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[2, 1].text(-0.15, 1.1, 'f', transform=axes[2, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')

    
fig.tight_layout()
#fig.savefig('figure-New.png', dpi=300, bbox_inches='tight', format='png')
plt.show()
    


#Time_series plot of Human post-antibiotics dataset
file_path = "/path/to/data/Ratio_read.xlsx" 
recovery_df = pd.read_excel(file_path, index_col = 0)

recovery_df = recovery_df.reset_index() 

recovery_df['Log_B:H read ratio'] = np.log(recovery_df['B:H read ratio'] + 1e-9)

recovery_df['Time point'] = pd.Categorical(recovery_df['Time point'], categories=["Day 0", "Day 4", "Day 8", "Day 42", "Day 180"], ordered=True)

sns.set(style="whitegrid")

plt.figure(figsize=(14, 7))
sns.lineplot(data=recovery_df, x="Time point", y='Log_B:H read ratio', linewidth=2, color='#000080')

plt.title('Distribution of Bacterial-to-Huma Ratio Over Time')
plt.xlabel('Time Point')
plt.ylabel('Log Bacterial-to-Human Read Ratio')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Running t-tests for each time point compared to the baseline
p_values = []
for time_point in time_points:
    current_data = recovery_df[recovery_df['Time point'] == time_point]['B:H read ratio']
    t_stat, p_value = stats.ttest_ind(baseline_data, current_data, equal_var=False)
    p_values.append(p_value)

significance_labels = []
for p in p_values:
    if p < 0.001:
        significance_labels.append('***')
    elif p < 0.01:
        significance_labels.append('**')
    elif p < 0.05:
        significance_labels.append('*')
    else:
        significance_labels.append('')

results_df = pd.DataFrame({
    'Time Point': time_points,
    'P-Value': p_values,
    'Significant': significance_labels
})

print(results_df)


##Time_series plot of Mouse post-antibiotics dataset
file_path = "/path/to/data/Mouse_metagenomic_read_data.xlsx" 
Mouse_df = pd.read_excel(file_path)

cleaned_Mouse_df = Mouse_df.dropna(subset=['Total_Biomass_plant2', 'Bacterial_read', 'Mouse_read', 'B:M_ratio'])

cleaned_Mouse_df['Log_B:M_ratio'] = np.log(cleaned_Mouse_df['B:M_ratio'] + 1e-9)
cleaned_Mouse_df['Log_Total_Biomass'] = np.log(cleaned_Mouse_df['Total_Biomass_plant2'] + 1e-9)

cleaned_Mouse_df['Days'] = cleaned_Mouse_df['Days'].astype(str)

cleaned_Mouse_df = cleaned_Mouse_df[~cleaned_Mouse_df['Days'].isin([25, 28])]

cleaned_Mouse_df['Days'] = pd.Categorical(cleaned_Mouse_df['Days'], categories=["0", "3", "6", "7", "10", "13", "16", "19", "22"], ordered=True)

sns.set(style="whitegrid")

plt.figure(figsize=(14, 7))
sns.lineplot(data=cleaned_Mouse_df, x="Days", y="Log_B:M_ratio", color='#800000', linewidth=2)

plt.title('Distribution of Log Bacterial-to-Mouse Ratio Over Time')
plt.xlabel('Time Point (Day)')
plt.ylabel('log Bacterial-to-Mouse Read Ratio')
plt.grid(visible=True, alpha=0.2)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Running t-tests for each time point compared to the baseline
p_values = []
for time_point in time_points:
    current_data = cleaned_Mouse_df[cleaned_Mouse_df['Days'] == time_point]['B:M_ratio']
    t_stat, p_value = stats.ttest_ind(baseline_data, current_data, equal_var=False)
    p_values.append(p_value)

significance_labels = []
for p in p_values:
    if p < 0.001:
        significance_labels.append('***')
    elif p < 0.01:
        significance_labels.append('**')
    elif p < 0.05:
        significance_labels.append('*')
    else:
        significance_labels.append('')

results_df = pd.DataFrame({
    'Time Point': time_points,
    'P-Value': p_values,
    'Significant': significance_labels
})

print(results_df)
    

#Figure 5 generation

fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.5))

font_label = {'fontsize': 7, 'color': 'black'}
font_ticks = {'fontsize': 7, 'color': 'black'}

plt.rcParams.update({'xtick.labelsize': 7, 'ytick.labelsize': 7})

recovery_df['Time point'] = recovery_df['Time point'].str.replace('Day ', '')

recovery_df['Time point'] = pd.Categorical(recovery_df['Time point'], categories=["0", "4", "8", "42", "180"], ordered=True)



sns.lineplot(data=recovery_df, x='Time point', y='Log_B:H read ratio', ax=axes[0], color='#000080', linewidth=1.2)
sns.scatterplot(data=recovery_df, x='Time point', y='Log_B:H read ratio', ax=axes[0], color='#000080', alpha=0.5, s=20)


axes[0].axvspan(0, 1, color='grey', alpha=0.3)
axes[0].text(1, 7.5, '***', color='black', fontsize=12, ha='center')
axes[0].set_xlabel('Time Point (Day)', **font_label)
axes[0].set_ylabel('Log Bacterial-to-Host Read Ratio', **font_label)
axes[0].grid(False)

for spine in axes[0].spines.values():
    spine.set_edgecolor('#4D4D4D') 
    spine.set_linewidth(0.7)


    
sns.lineplot(data=cleaned_Mouse_df, x='Days', y='Log_B:M_ratio', ax=axes[1], color='#800000', linewidth=1.2)
sns.scatterplot(data=cleaned_Mouse_df, x='Days', y='Log_B:M_ratio', ax=axes[1], color='#800000', alpha=0.5, s=20)


axes[1].axvspan(0, 2, color='grey', alpha=0.3)

axes[1].text(1, 1.5, '***', color='black', fontsize=12, ha='center')
axes[1].text(2, 1.5, '***', color='black', fontsize=12, ha='center')
axes[1].text(3, 1.5, '***', color='black', fontsize=12, ha='center')


axes[1].set_xlabel('Time Point (Day)', **font_label)
axes[1].set_ylabel('Log Bacterial-to-Host Read Ratio', **font_label)
axes[1].grid(False)


axes[0].text(-0.15, 1.1, 'a', transform=axes[0].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
axes[1].text(-0.15, 1.1, 'b', transform=axes[1].transAxes, fontsize=9, fontweight='bold', va='top', ha='right')


for spine in axes[1].spines.values():
    spine.set_edgecolor('#4D4D4D')
    spine.set_linewidth(0.7)

plt.tight_layout()

plt.show()





