# =============================================================================
# DiD ANALYSIS — WAR IMPACT ON UKRAINIAN EDUCATION (REGION-LEVEL)
# Fixed: wild bootstrap, no clustering, trend-break analysis added
# =============================================================================

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '.'
MATPLOTLIB_NEW = tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 9)
print(f"matplotlib: {matplotlib.__version__} | new API: {MATPLOTLIB_NEW}")

# =============================================================================
# PART 0.1. IMPACT INDEX — USER-SPECIFIED
# =============================================================================
impact = pd.DataFrame({
    'region': [
        'Вінницька','Волинська','Дніпропетровська','Донецька','Житомирська',
        'Закарпатська','Запорізька','Івано-Франківська','Київська','Кіровоградська',
        'Луганська','Львівська','Миколаївська','Одеська','Полтавська',
        'Рівненська','Сумська','Тернопільська','Харківська','Херсонська',
        'Хмельницька','Черкаська','Чернівецька','Чернігівська','м. Київ'
    ],
    'Index': [
        0.0488, 0.0129, 0.2308, 0.7260, 0.1104,
        0.0096, 0.2143, 0.0119, 0.1762, 0.0906,
        0.7260, 0.0156, 0.2716, 0.0757, 0.1268,
        0.0183, 0.3406, 0.0151, 0.5259, 0.5973,
        0.0843, 0.0892, 0.0137, 0.1936, 0.1001
    ],
    'group': [
        'Control','Control','Treated','Treated','Out',
        'Control','Treated','Control','Out','Out',
        'Treated','Control','Treated','Out','Out',
        'Control','Treated','Control','Treated','Treated',
        'Out','Out','Control','Out','Out'
    ]
})
i_min, i_max = impact['Index'].min(), impact['Index'].max()
impact['Index_norm'] = 0.02 + (impact['Index'] - i_min)/(i_max - i_min)*0.98

print("\n" + "=" * 80)
print("IMPACT INDEX & GROUPS")
print("=" * 80)
print(impact[['region','Index','Index_norm','group']]
      .sort_values('Index', ascending=False).to_string(index=False))
print("\nGroup counts:")
print(impact['group'].value_counts().to_string())

# =============================================================================
# PART 0.2. EDUCATION PANEL
# =============================================================================
raw = {
'Вінницька':        [(2018,795,162,152.1,None,25),(2019,772,165,150.6,None,28),
                     (2020,747,167,152.4,28.6,28),(2021,654,166,155.3,30.3,19),
                     (2022,637,168,147.4,25.5,22),(2024,602,159,140.7,30.9,28)],
'Волинська':        [(2018,629,139,155.8,None,13),(2019,614,142,152.6,None,14),
                     (2020,595,145,155.8,28.1,17),(2021,561,145,156.1,29.6,15),
                     (2022,551,145,148.2,25.1,20),(2024,518,137,142.1,30.2,24)],
'Дніпропетровська': [(2018,906,330,151.3,None,61),(2019,880,337,149.7,None,60),
                     (2020,862,342,151.5,27.3,63),(2021,848,342,160.5,29.2,42),
                     (2022,836,340,151.9,24.7,44),(2024,820,308,144.8,29.9,38)],
'Донецька':         [(2018,521,162,148.1,None,9),(2019,514,167,147.9,None,10),
                     (2020,509,170,149.6,25.8,10),(2021,491,168,np.nan,26.6,np.nan),
                     (2022,298,105,np.nan,22.8,np.nan),(2024,256,77,np.nan,28.0,np.nan)],
'Житомирська':      [(2018,667,135,153.0,None,12),(2019,638,137,151.6,None,13),
                     (2020,618,139,155.9,29.3,18),(2021,578,138,153.0,30.6,12),
                     (2022,551,138,145.2,25.9,12),(2024,511,130,139.8,31.3,14)],
'Закарпатська':     [(2018,665,165,152.1,None,10),(2019,666,168,150.9,None,12),
                     (2020,635,171,152.0,27.2,13),(2021,586,171,154.9,29.0,12),
                     (2022,581,169,147.6,24.5,12),(2024,565,159,141.3,29.6,14)],
'Запорізька':       [(2018,557,168,147.6,None,24),(2019,538,171,146.1,None,24),
                     (2020,534,172,148.4,28.9,25),(2021,496,170,160.2,30.9,13),
                     (2022,446,156,158.9,26.0,13),(2024,318,115,150.9,32.7,12)],
'Івано-Франківська':[(2018,705,156,156.4,None,21),(2019,689,159,152.1,None,23),
                     (2020,664,161,153.4,28.4,27),(2021,606,161,157.9,30.5,19),
                     (2022,589,161,149.3,25.7,22),(2024,550,155,143.3,31.0,30)],
'Київська':         [(2018,706,212,149.5,None,7),(2019,702,223,149.5,None,8),
                     (2020,703,232,148.5,24.6,8),(2021,656,240,150.4,26.0,6),
                     (2022,640,241,148.0,22.0,7),(2024,630,236,139.2,26.5,7)],
'Кіровоградська':   [(2018,325,95,152.1,None,5),(2019,320,96,151.9,None,5),
                     (2020,309,97,152.9,29.3,5),(2021,303,97,152.2,31.1,5),
                     (2022,293,97,145.5,26.3,6),(2024,285,91,139.8,32.0,7)],
'Луганська':        [(2018,280,55,144.7,None,4),(2019,276,55,144.9,None,4),
                     (2020,273,56,146.0,29.7,5),(2021,254,55,np.nan,31.4,np.nan),
                     (2022,83,27,np.nan,25.4,np.nan),(2024,83,21,np.nan,32.6,np.nan)],
'Львівська':        [(2018,1208,277,162.1,None,99),(2019,1194,283,160.1,None,106),
                     (2020,1181,287,162.6,26.6,124),(2021,1130,289,164.8,28.0,81),
                     (2022,1114,292,156.8,23.6,97),(2024,1077,282,149.3,28.5,113)],
'Миколаївська':     [(2018,500,117,152.0,None,10),(2019,484,119,151.2,None,10),
                     (2020,477,120,152.7,29.5,11),(2021,453,119,161.4,31.4,5),
                     (2022,436,114,154.1,26.7,6),(2024,416,101,147.0,32.3,7)],
'Одеська':          [(2018,809,265,153.7,None,53),(2019,803,272,152.3,None,55),
                     (2020,786,278,154.1,26.4,61),(2021,736,282,162.1,28.5,34),
                     (2022,725,275,154.2,24.2,36),(2024,707,253,152.5,29.2,56)],
'Полтавська':       [(2018,619,134,151.0,None,14),(2019,606,136,148.2,None,15),
                     (2020,584,138,150.7,29.9,14),(2021,534,139,158.6,31.6,14),
                     (2022,528,140,151.4,26.7,16),(2024,500,132,144.9,32.4,21)],
'Рівненська':       [(2018,608,159,157.0,None,14),(2019,590,163,154.7,None,14),
                     (2020,588,166,157.1,26.5,15),(2021,553,168,155.8,28.7,12),
                     (2022,548,168,147.2,24.3,15),(2024,536,163,141.9,29.2,17)],
'Сумська':          [(2018,443,96,156.0,None,10),(2019,422,98,154.4,None,11),
                     (2020,404,99,155.7,29.6,11),(2021,375,98,165.8,31.5,8),
                     (2022,365,96,157.0,26.8,9),(2024,321,87,149.7,32.8,7)],
'Тернопільська':    [(2018,726,110,154.7,None,17),(2019,699,111,154.1,None,17),
                     (2020,687,112,155.4,29.4,20),(2021,653,111,156.4,30.3,17),
                     (2022,623,110,148.5,25.7,19),(2024,553,102,142.1,31.1,21)],
'Харківська':       [(2018,746,246,157.8,None,108),(2019,740,254,155.0,None,119),
                     (2020,734,259,157.8,27.0,136),(2021,715,261,171.6,28.2,46),
                     (2022,614,234,160.9,23.7,57),(2024,612,207,152.7,29.3,53)],
'Херсонська':       [(2018,428,111,149.0,None,9),(2019,417,113,148.1,None,9),
                     (2020,395,114,151.8,27.4,9),(2021,384,114,153.4,30.0,12),
                     (2022,156,66,np.nan,25.4,np.nan),(2024,174,57,np.nan,31.2,np.nan)],
'Хмельницька':      [(2018,682,134,154.6,None,13),(2019,649,137,152.8,None,13),
                     (2020,621,139,155.0,29.7,15),(2021,578,139,153.4,31.1,12),
                     (2022,561,140,146.6,26.3,14),(2024,513,133,141.0,31.9,14)],
'Черкаська':        [(2018,577,115,152.2,None,14),(2019,556,116,150.8,None,14),
                     (2020,533,118,153.3,30.0,14),(2021,493,117,152.7,31.6,11),
                     (2022,475,118,145.9,26.8,13),(2024,432,113,138.9,32.4,14)],
'Чернівецька':      [(2018,406,103,160.5,None,15),(2019,403,105,158.2,None,14),
                     (2020,396,107,160.3,25.7,16),(2021,349,108,160.5,27.5,12),
                     (2022,343,109,152.4,23.3,14),(2024,335,104,147.0,28.2,20)],
'Чернігівська':     [(2018,508,95,153.6,None,9),(2019,487,97,150.9,None,10),
                     (2020,472,97,154.0,30.6,10),(2021,426,97,163.8,32.2,7),
                     (2022,402,94,154.8,27.3,8),(2024,377,87,144.4,33.1,8)],
'м. Київ':          [(2018,505,301,158.7,None,295),(2019,535,314,156.3,None,324),
                     (2020,566,325,158.1,15.5,381),(2021,579,335,162.6,17.1,192),
                     (2022,581,339,154.5,14.5,225),(2024,600,335,147.9,17.1,216)],
}

rows = []
for reg, records in raw.items():
    for r in records:
        rows.append([reg] + list(r))
df = pd.DataFrame(rows, columns=['region','year','n_schools','n_children',
                                 'SKB','subvention','applications'])
df = df.merge(impact[['region','Index','Index_norm','group']], on='region', how='left')

df['post']     = (df['year'] >= 2022).astype(int)
df['treated']  = (df['group'] == 'Treated').astype(int)
df['control']  = (df['group'] == 'Control').astype(int)
df['did_bin']  = df['treated'] * df['post']
df['did_cont'] = df['Index_norm'] * df['post']

# Trend variables for trend-break analysis
df['t']       = df['year'] - 2018                     # 0,1,2,3,4,6
df['t_war']   = np.maximum(0, df['year'] - 2021)      # 0,0,0,0,1,3
df['Treated_t']      = df['treated'] * df['t']
df['Treated_t_war']  = df['treated'] * df['t_war']

OUTCOMES = ['n_schools','n_children','SKB','subvention','applications']
OUTCOME_LABELS = {
    'n_schools':    'Кількість закладів',
    'n_children':   'Кількість дітей, тис.',
    'SKB':          'Середній конкурсний бал (СКБ)',
    'subvention':   'Субвенція на 1 учня, тис. грн',
    'applications': 'Усього заяв на вступ'
}

# =============================================================================
# PART 0.3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

desc_rows = []
for y in OUTCOMES:
    for grp in ['Treated','Control']:
        for per, mask in [('Pre-war', df['year'] <= 2021),
                          ('War',     df['year'] >= 2022)]:
            sub = df[(df['group'] == grp) & mask][y].dropna()
            if len(sub) > 0:
                desc_rows.append({
                    'outcome': y, 'group': grp, 'period': per,
                    'N': len(sub), 'mean': sub.mean(), 'sd': sub.std(),
                    'min': sub.min(), 'median': sub.median(), 'max': sub.max()
                })
desc_df = pd.DataFrame(desc_rows)
print(desc_df.round(3).to_string(index=False))
desc_df.to_csv(f'{OUT_DIR}/descriptive_stats.csv', index=False)

pivot = desc_df.pivot_table(index=['outcome','group'], columns='period',
                            values='mean', aggfunc='first')
pivot['Change_abs'] = pivot['War'] - pivot['Pre-war']
pivot['Change_pct'] = (pivot['Change_abs'] / pivot['Pre-war'] * 100)
print("\nCHANGE PRE→WAR:")
print(pivot.round(3).to_string())
pivot.to_csv(f'{OUT_DIR}/change_by_group.csv')

# =============================================================================
# PART 1. BALANCING TEST (2021)
# =============================================================================
pre = df[(df['year'] == 2021) & (df['group'].isin(['Treated','Control']))]

print("\n" + "=" * 80)
print("BALANCING TEST (2021)")
print("=" * 80)
balance = []
for y in OUTCOMES:
    t = pre[pre['treated']==1][y].dropna()
    c = pre[pre['control']==1][y].dropna()
    if len(t)>1 and len(c)>1:
        stat, p = stats.ttest_ind(t, c, equal_var=False)
        diff = c.mean() - t.mean()
        se = np.sqrt(c.var(ddof=1)/len(c) + t.var(ddof=1)/len(t))
        balance.append([y, c.mean(), t.mean(), stat, p,
                        diff-1.96*se, diff+1.96*se])
bal_df = pd.DataFrame(balance, columns=['outcome','mean_control','mean_treated',
                                        't','p','CI_low','CI_high'])
print(bal_df.round(4).to_string(index=False))
bal_df.to_csv(f'{OUT_DIR}/balancing_test.csv', index=False)

# =============================================================================
# PART 2. GRAPHICS
# =============================================================================
# 2.1. Parallel trends
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, y in enumerate(OUTCOMES):
    ax = axes[i]
    sub = df[df['group'].isin(['Treated','Control'])]
    means = sub.groupby(['year','group'])[y].mean().unstack()
    means.plot(ax=ax, marker='o', linewidth=2)
    ax.axvline(2021.5, color='red', linestyle='--', alpha=0.5, label='Початок війни')
    ax.set_title(OUTCOME_LABELS[y], fontsize=11)
    ax.set_xlabel('Рік'); ax.grid(alpha=0.3); ax.legend()
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/parallel_trends.png', dpi=150); plt.close()

# 2.2. Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, y in enumerate(OUTCOMES):
    ax = axes[i]
    sub = df[df['group'].isin(['Treated','Control'])].copy()
    sub['period'] = np.where(sub['year'] <= 2021, 'Pre-war', 'War')
    sub['label']  = sub['group'] + ' · ' + sub['period']
    order_box = ['Control · Pre-war','Control · War',
                 'Treated · Pre-war','Treated · War']
    data_box = [sub[sub['label']==lab][y].dropna().values for lab in order_box]
    if MATPLOTLIB_NEW:
        box = ax.boxplot(data_box, patch_artist=True, widths=0.6,
                         tick_labels=order_box)
    else:
        box = ax.boxplot(data_box, patch_artist=True, widths=0.6,
                         labels=order_box)
    for patch, c in zip(box['boxes'], ['#8DBCB8','#5C9E9A','#E8A6A0','#C97B74']):
        patch.set_facecolor(c)
    ax.set_title(OUTCOME_LABELS[y], fontsize=11)
    ax.tick_params(axis='x', rotation=15); ax.grid(alpha=0.3, axis='y')
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/boxplots_by_group.png', dpi=150); plt.close()

# 2.3. % change
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, y in enumerate(OUTCOMES):
    ax = axes[i]
    vals = []
    for grp in ['Control','Treated']:
        pre_m = df[(df['group']==grp) & (df['year']<=2021)][y].dropna().mean()
        war_m = df[(df['group']==grp) & (df['year']>=2022)][y].dropna().mean()
        vals.append((war_m - pre_m) / pre_m * 100 if pre_m else np.nan)
    bars = ax.bar(['Control','Treated'], vals,
                  color=['#5C9E9A','#C97B74'], edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(OUTCOME_LABELS[y], fontsize=11)
    ax.set_ylabel('Зміна, %'); ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:+.1f}%',
                ha='center', va='bottom' if v>=0 else 'top', fontsize=10)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/change_by_group.png', dpi=150); plt.close()

# 2.4. Trajectories
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, y in enumerate(OUTCOMES):
    ax = axes[i]
    sub = df[df['group'].isin(['Treated','Control'])]
    for grp, color, alpha in [('Control','#5C9E9A',0.35),
                              ('Treated','#C97B74',0.55)]:
        for reg in sub[sub['group']==grp]['region'].unique():
            r = sub[sub['region']==reg].sort_values('year')
            ax.plot(r['year'], r[y], color=color, alpha=alpha, linewidth=0.9)
    means = sub.groupby(['year','group'])[y].mean().unstack()
    if 'Control' in means:
        ax.plot(means.index, means['Control'], color='black',
                linewidth=2.5, label='Control (mean)')
    if 'Treated' in means:
        ax.plot(means.index, means['Treated'], color='darkred',
                linewidth=2.5, label='Treated (mean)')
    ax.axvline(2021.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title(OUTCOME_LABELS[y], fontsize=11)
    ax.set_xlabel('Рік'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/region_trajectories.png', dpi=150); plt.close()

# 2.5. Index distribution
fig, ax = plt.subplots(figsize=(10, 6))
groups = ['Control','Out','Treated']
colors = {'Control':'#5C9E9A','Out':'#B0B0B0','Treated':'#C97B74'}
for grp in groups:
    vals = impact[impact['group']==grp]['Index'].values
    ax.scatter(vals, [groups.index(grp)]*len(vals), s=120,
               color=colors[grp], alpha=0.75, edgecolor='black',
               linewidth=0.5, label=grp)
    for v in vals:
        ax.annotate(f'{v:.3f}', (v, groups.index(grp)),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=8)
ax.axvline(0.30, color='red', linestyle='--', alpha=0.6, label='Поріг Treated')
ax.axvline(0.14, color='blue', linestyle='--', alpha=0.6, label='Поріг Control')
ax.set_yticks(range(3)); ax.set_yticklabels(groups)
ax.set_xlabel('Impact Index (2024)')
ax.set_title('Розподіл областей за Impact Index', fontsize=12)
ax.grid(alpha=0.3, axis='x'); ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/index_distribution.png', dpi=150); plt.close()

print("\n" + "=" * 80)
print("GRAPHICS SAVED")
print("=" * 80)

# =============================================================================
# PART 3. BINARY DiD (fixed — no clustering interactions)
# =============================================================================
def run_binary(outcome):
    sub = df[df['group'].isin(['Treated','Control'])].copy().set_index(['region','year'])
    y = sub[outcome].dropna()
    X = sub.loc[y.index, ['did_bin']]
    return PanelOLS(y, X, entity_effects=True, time_effects=True)\
           .fit(cov_type='clustered', cluster_entity=True)

print("\n" + "=" * 80)
print("BINARY DiD")
print("=" * 80)
binary_res = {}
for y in OUTCOMES:
    try:
        r = run_binary(y); binary_res[y] = r
        print(f"{y:15s}  b3={r.params['did_bin']:10.4f}  "
              f"se={r.std_errors['did_bin']:8.4f}  "
              f"t={r.tstats['did_bin']:7.3f}  p={r.pvalues['did_bin']:.4f}")
    except Exception as e:
        print(f"{y}: {e}")

# =============================================================================
# PART 4. EVENT STUDY
# =============================================================================
def run_event(outcome):
    sub = df[df['group'].isin(['Treated','Control'])].copy().set_index(['region','year'])
    years = sorted(sub.index.get_level_values('year').unique())
    X = pd.DataFrame(index=sub.index)
    for yr in years:
        if yr == 2021: continue
        X[f'treat_{yr}'] = ((sub.index.get_level_values('year')==yr) &
                            (sub['treated']==1)).astype(int)
    y = sub[outcome].dropna()
    X = X.loc[y.index]
    return PanelOLS(y, X, entity_effects=True, time_effects=True)\
           .fit(cov_type='clustered', cluster_entity=True)

print("\n" + "=" * 80)
print("EVENT STUDY (ref = 2021)")
print("=" * 80)
es_res = {}
for y in OUTCOMES:
    try:
        es_res[y] = run_event(y)
        print(f"\n--- {y} ---")
        print(es_res[y].params.round(3).to_string())
    except Exception as e:
        print(f"{y}: {e}")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, y in enumerate(OUTCOMES):
    if y not in es_res: continue
    ax = axes[i]
    r = es_res[y]
    coefs = r.params; ci = r.conf_int()
    yrs = [int(c.split('_')[1]) for c in coefs.index]
    ax.errorbar(yrs, coefs.values,
                yerr=[coefs.values-ci.iloc[:,0].values,
                      ci.iloc[:,1].values-coefs.values],
                fmt='o', capsize=4, color='#C97B74')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(2021.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'Event study: {OUTCOME_LABELS[y]}', fontsize=11)
    ax.set_xlabel('Рік'); ax.grid(alpha=0.3)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/event_study.png', dpi=150); plt.close()

# =============================================================================
# PART 5. TREND-BREAK ANALYSIS (NEW)
# Question: did the trend for SKB / n_schools INTENSIFY in Treated after war?
# =============================================================================
def run_trend_break(outcome):
    """
    Model: Y = a + b1*t + b2*t_war + b3*(Treated*t) + b4*(Treated*t_war)
              + region FE + e
    - b1: pre-war slope (Control)
    - b2: change in slope post-war (Control)
    - b3: pre-war slope difference (Treated - Control)
    - b4: DiD-in-slopes = (Δslope_Treated) - (Δslope_Control)
    """
    sub = df[df['group'].isin(['Treated','Control'])].copy().set_index(['region','year'])
    y = sub[outcome].dropna()
    X = sub.loc[y.index, ['t','t_war','Treated_t','Treated_t_war']]
    # NOTE: 'treated' main effect is absorbed by entity FE
    return PanelOLS(y, X, entity_effects=True, time_effects=False)\
           .fit(cov_type='clustered', cluster_entity=True)

print("\n" + "=" * 80)
print("TREND-BREAK ANALYSIS")
print("Y = a + b1*t + b2*t_war + b3*(Treated*t) + b4*(Treated*t_war) + FE")
print("b4 < 0 → war INTENSIFIED decline in Treated vs Control")
print("=" * 80)
trend_res = {}
for y in OUTCOMES:
    try:
        r = run_trend_break(y); trend_res[y] = r
        print(f"\n--- {OUTCOME_LABELS[y]} ---")
        for name in ['t','t_war','Treated_t','Treated_t_war']:
            b  = r.params[name]; se = r.std_errors[name]
            p  = r.pvalues[name]; t  = r.tstats[name]
            print(f"  {name:15s}  b={b:9.4f}  se={se:7.4f}  t={t:7.3f}  p={p:.4f}")
        print(f"  → b4 (DID-in-slopes) = {r.params['Treated_t_war']:.4f}, "
              f"p={r.pvalues['Treated_t_war']:.4f}")
        if r.pvalues['Treated_t_war'] < 0.05:
            direction = "ПОСИЛИВСЯ (погіршився)" if r.params['Treated_t_war'] < 0 \
                        else "ЗМІНИВСЯ (покращився)"
            print(f"  ✓ Тренд {direction} у Treated (p < 0.05)")
        elif r.pvalues['Treated_t_war'] < 0.10:
            print(f"  ~ Маргінальна зміна тренду (p < 0.10)")
        else:
            print(f"  ✗ Немає статистичної зміни тренду (p = {r.pvalues['Treated_t_war']:.4f})")
    except Exception as e:
        print(f"{y}: {e}")

# Slopes visualization
def get_slopes(outcome):
    """Estimate slopes per group and period by simple OLS"""
    res = {}
    for grp in ['Control','Treated']:
        for per, mask in [('Pre-war', df['year'] <= 2021),
                          ('War',     df['year'] >= 2022)]:
            sub = df[(df['group']==grp) & mask]
            sub = sub[['year', outcome]].dropna()
            if len(sub) > 2:
                slope, intercept, r, p, se = stats.linregress(sub['year'], sub[outcome])
                res[(grp, per)] = (slope, se, p)
            else:
                res[(grp, per)] = (np.nan, np.nan, np.nan)
    return res

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
targets = ['n_schools', 'SKB', 'n_children']
for i, y in enumerate(targets):
    ax = axes[i]
    slopes = get_slopes(y)
    labels, vals, errs = [], [], []
    for grp in ['Control','Treated']:
        for per in ['Pre-war','War']:
            lab = f'{grp}\n{per}'
            labels.append(lab)
            vals.append(slopes[(grp, per)][0])
            errs.append(slopes[(grp, per)][1])
    colors = ['#5C9E9A','#8DBCB8','#C97B74','#E8A6A0']
    bars = ax.bar(labels, vals, yerr=errs, color=colors,
                  edgecolor='black', linewidth=0.6, capsize=4)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'Slope by group/period: {OUTCOME_LABELS[y]}', fontsize=11)
    ax.set_ylabel('Нахил тренду (од. на рік)')
    ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x()+b.get_width()/2, v, f'{v:+.1f}',
                    ha='center', va='bottom' if v>=0 else 'top', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/slopes_by_group.png', dpi=150); plt.close()

# Trend break plot (visual with fitted lines)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for i, y in enumerate(targets):
    ax = axes[i]
    for grp, color in [('Control','#5C9E9A'), ('Treated','#C97B74')]:
        sub = df[df['group']==grp]
        # Pre-war fit
        pre_sub = sub[sub['year']<=2021][['year', y]].dropna()
        if len(pre_sub) > 2:
            s, ic, r, p, se = stats.linregress(pre_sub['year'], pre_sub[y])
            xs = np.linspace(2018, 2021, 10)
            ax.plot(xs, ic + s*xs, color=color, linewidth=2.2,
                    linestyle='-', label=f'{grp} (pre-war)')
        # War fit
        war_sub = sub[sub['year']>=2022][['year', y]].dropna()
        if len(war_sub) > 1:
            s2, ic2, r2, p2, se2 = stats.linregress(war_sub['year'], war_sub[y])
            xs = np.linspace(2022, 2024, 10)
            ax.plot(xs, ic2 + s2*xs, color=color, linewidth=2.2,
                    linestyle='--', label=f'{grp} (war)')
    ax.axvline(2021.5, color='red', linestyle=':', alpha=0.6)
    ax.set_title(OUTCOME_LABELS[y], fontsize=11)
    ax.set_xlabel('Рік'); ax.set_ylabel(OUTCOME_LABELS[y])
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/trend_break.png', dpi=150); plt.close()

# =============================================================================
# PART 6. CONTINUOUS DiD
# =============================================================================
def run_cont(outcome):
    sub = df.copy().set_index(['region','year'])
    y = sub[outcome].dropna()
    X = sub.loc[y.index, ['did_cont']]
    return PanelOLS(y, X, entity_effects=True, time_effects=True)\
           .fit(cov_type='clustered', cluster_entity=True)

print("\n" + "=" * 80)
print("CONTINUOUS DiD")
print("=" * 80)
cont_res = {}
for y in OUTCOMES:
    try:
        r = run_cont(y); cont_res[y] = r
        print(f"{y:15s}  b3={r.params['did_cont']:10.4f}  "
              f"se={r.std_errors['did_cont']:8.4f}  "
              f"t={r.tstats['did_cont']:7.3f}  p={r.pvalues['did_cont']:.4f}  "
              f"| +0.1 → {0.1*r.params['did_cont']:.4f}")
    except Exception as e:
        print(f"{y}: {e}")

# =============================================================================
# PART 7. ROBUSTNESS
# =============================================================================
OCC = ['Донецька','Луганська','Херсонська']

def run_cont_no_occ(outcome):
    sub = df[~df['region'].isin(OCC)].copy().set_index(['region','year'])
    y = sub[outcome].dropna()
    X = sub.loc[y.index, ['did_cont']]
    return PanelOLS(y, X, entity_effects=True, time_effects=True)\
           .fit(cov_type='clustered', cluster_entity=True)

print("\n" + "=" * 80)
print("ROBUSTNESS 1: exclude occupied regions")
print("=" * 80)
for y in OUTCOMES:
    try:
        r = run_cont_no_occ(y)
        print(f"{y:15s}  b3={r.params['did_cont']:10.4f}  p={r.pvalues['did_cont']:.4f}")
    except Exception as e:
        print(f"{y}: {e}")

def run_placebo(outcome, fake_year=2020):
    sub = df[df['year'] <= 2021].copy()
    sub['post_f'] = (sub['year'] >= fake_year).astype(int)
    sub['did_f']  = sub['Index_norm'] * sub['post_f']
    sub = sub.set_index(['region','year'])
    y = sub[outcome].dropna()
    X = sub.loc[y.index, ['did_f']]
    return PanelOLS(y, X, entity_effects=True, time_effects=True)\
           .fit(cov_type='clustered', cluster_entity=True)

print("\n" + "=" * 80)
print("ROBUSTNESS 2: PLACEBO (fake war 2020)")
print("=" * 80)
for y in OUTCOMES:
    try:
        r = run_placebo(y)
        print(f"{y:15s}  b3={r.params['did_f']:10.4f}  p={r.pvalues['did_f']:.4f}")
    except Exception as e:
        print(f"{y}: {e}")

# =============================================================================
# PART 8. WILD CLUSTER BOOTSTRAP (FIXED)
# =============================================================================
import statsmodels.api as sm

def wild_bootstrap(outcome, n_boot=499, seed=42):
    """
    Wild cluster bootstrap-t (Rademacher, CGM 2008).
    FIX: restricted model built manually via statsmodels OLS with FE dummies,
         avoiding linearmodels crash when X is empty.
    """
    np.random.seed(seed)

    sub = df.copy().set_index(['region','year'])
    y   = sub[outcome].dropna()
    X   = sub.loc[y.index, ['did_cont']]

    regions = sub.loc[y.index].index.get_level_values('region')
    years   = sub.loc[y.index].index.get_level_values('year')

    # Build FE dummies manually
    reg_dum = pd.get_dummies(regions, prefix='r', drop_first=True).astype(float).values
    yr_dum  = pd.get_dummies(years,   prefix='y', drop_first=True).astype(float).values
    FE      = np.hstack([reg_dum, yr_dum])
    y_arr   = np.asarray(y).flatten()

    # Restricted model: y = FE only  (beta = 0 imposed)
    fit_r    = sm.OLS(y_arr, FE).fit()
    fitted_r = np.asarray(fit_r.fittedvalues).flatten()
    resid_r  = y_arr - fitted_r

    # Observed t-stat
    r_obs = PanelOLS(y, X, entity_effects=True, time_effects=True)\
            .fit(cov_type='clustered', cluster_entity=True)
    t_obs = float(r_obs.tstats['did_cont'])

    # Bootstrap
    clusters = sub.loc[y.index].index.get_level_values('region').values
    uniq     = np.unique(clusters)

    t_boot = []
    for _ in range(n_boot):
        w = pd.Series(np.random.choice([-1, 1], size=len(uniq)), index=uniq)
        w_obs = w.loc[clusters].values
        y_star_arr = fitted_r + resid_r * w_obs
        y_star = pd.Series(y_star_arr, index=y.index, name=y.name)

        try:
            r_b = PanelOLS(y_star, X, entity_effects=True, time_effects=True)\
                  .fit(cov_type='clustered', cluster_entity=True)
            t_boot.append(float(r_b.tstats['did_cont']))
        except Exception:
            continue

    if len(t_boot) < 10:
        return t_obs, np.nan, np.nan

    t_boot = np.array(t_boot)
    p_wild  = float(np.mean(np.abs(t_boot) >= np.abs(t_obs)))
    se_boot = float(np.std(t_boot, ddof=1))
    return t_obs, p_wild, se_boot

print("\n" + "=" * 80)
print("WILD CLUSTER BOOTSTRAP (Cameron-Gelbach-Miller)")
print("=" * 80)

wild_results = {}   # <-- FIX: initialize BEFORE the loop
for y in OUTCOMES:
    try:
        t_obs, p_wild, se_wild = wild_bootstrap(y, n_boot=499)
        if np.isnan(p_wild):
            print(f"{y:15s}  t_obs={t_obs:7.3f}  [bootstrap failed — insufficient draws]")
            continue
        wild_results[y] = (t_obs, p_wild, se_wild)
        print(f"{y:15s}  t_obs={t_obs:7.3f}  p_wild={p_wild:.4f}  SE_boot={se_wild:.4f}")
    except Exception as e:
        print(f"{y}: {e}")

# =============================================================================
# PART 9. EXPORT (SAFE)
# =============================================================================
def collect(results_dict, label):
    rows = []
    for y, r in results_dict.items():
        rows.append({'model': label, 'outcome': y,
                     'coef': r.params.iloc[0],
                     'se': r.std_errors.iloc[0],
                     't': r.tstats.iloc[0],
                     'p': r.pvalues.iloc[0],
                     'n_obs': r.nobs})
    return pd.DataFrame(rows)

out = pd.concat([collect(binary_res, 'binary'),
                 collect(cont_res,   'continuous'),
                 collect(trend_res,  'trend_break')], ignore_index=True)
out.to_csv(f'{OUT_DIR}/did_results.csv', index=False)
impact.to_csv(f'{OUT_DIR}/impact_index.csv', index=False)

# Trend break
trend_rows = []
for y, r in trend_res.items():
    for name in ['t','t_war','Treated_t','Treated_t_war']:
        trend_rows.append({
            'outcome': y, 'term': name,
            'coef': r.params[name], 'se': r.std_errors[name],
            't': r.tstats[name], 'p': r.pvalues[name]
        })
pd.DataFrame(trend_rows).to_csv(f'{OUT_DIR}/trend_break_results.csv', index=False)

# Wild bootstrap — safe check
if wild_results:
    wild_df = pd.DataFrame([
        {'outcome': y, 't_obs': v[0], 'p_wild': v[1], 'se_boot': v[2]}
        for y, v in wild_results.items()
    ])
    wild_df.to_csv(f'{OUT_DIR}/wild_bootstrap.csv', index=False)
    print("  wild_bootstrap.csv saved")
else:
    print("  wild_bootstrap.csv — skipped (no results)")

print("\n" + "=" * 80)
print("FILES SAVED:")
print("  did_results.csv, trend_break_results.csv,")
print("  impact_index.csv, descriptive_stats.csv, change_by_group.csv,")
print("  balancing_test.csv")
print("  parallel_trends.png, boxplots_by_group.png, change_by_group.png,")
print("  region_trajectories.png, index_distribution.png, event_study.png,")
print("  slopes_by_group.png, trend_break.png")
print("=" * 80)
print("DONE.")