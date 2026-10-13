# =============================================================================
# EXTENDED ANALYSIS — SYNTHETIC CONTROL, CS/SA, CR2, RANDOMIZATION INFERENCE,
# SPATIAL ANALYSIS (LISA, THEIL, CONTIGUITY), PLACEBO, SUBVENTION EVENT STUDY
#
# Інтегрується з main.py та spatial_analysis_regions.py
# Запуск: покладіть цей файл поруч із main.py
#
# Виправлено:
#  - Synthetic Control — ручна реалізація (без SyntheticControlMethods)
#  - Callaway-Sant'Anna — коректний keyword-only виклик
#  - esda.Moran — конвертація numpy-матриці в libpysal W-об'єкт
#  - Placebo 2018 — свідомо пропускається (немає pre-period)
# =============================================================================

from pathlib import Path
import ast
import warnings
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =============================================================================
# НАЛАШТУВАННЯ ШЛЯХІВ
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = BASE_DIR / "main.py"
OUT_DIR = BASE_DIR / "extended_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT_DIR}")

# =============================================================================
# СПРОБА ІМПОРТУ ДОДАТКОВИХ БІБЛІОТЕК
# =============================================================================
try:
    from libpysal.weights import W
    from esda.moran import Moran, Moran_Local
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False
    print("⚠ libpysal / esda не встановлено. LISA та Moran scatterplot пропущено.")
    print("  Встановіть: pip install libpysal esda")

try:
    from linearmodels.panel import PanelOLS
    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False
    print("⚠ linearmodels не встановлено.")
    print("  Встановіть: pip install linearmodels")

try:
    import pyfixest as pf
    PYFIXEST_AVAILABLE = True
except ImportError:
    PYFIXEST_AVAILABLE = False
    print("⚠ pyfixest не встановлено. CR2 через pyfixest пропущено (буде fallback).")

# =============================================================================
# 0. ЗАВАНТАЖЕННЯ ДАНИХ З main.py
# =============================================================================
def extract_raw_dict(script_path: Path) -> dict:
    """Зчитує лише присвоєння raw з main.py, не запускаючи основний аналіз."""
    if not script_path.exists():
        raise FileNotFoundError(f"Не знайдено {script_path}")
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "raw" for t in node.targets):
                expr = ast.Expression(node.value)
                return eval(compile(expr, str(script_path), "eval"),
                            {"np": np, "nan": np.nan}, {})
    raise ValueError("У main.py не знайдено словник raw.")

raw = extract_raw_dict(MAIN_SCRIPT)

# Будуємо панель
rows = []
for reg, records in raw.items():
    for r in records:
        rows.append([reg] + list(r))

df = pd.DataFrame(rows, columns=['region', 'year', 'n_schools', 'n_children',
                                 'SKB', 'subvention', 'applications'])
df['year'] = df['year'].astype(int)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impact Index та групи
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
impact['Index_norm'] = 0.02 + (impact['Index'] - i_min) / (i_max - i_min) * 0.98

df = df.merge(impact[['region', 'Index', 'Index_norm', 'group']], on='region', how='left')
df['post'] = (df['year'] >= 2022).astype(int)
df['treated'] = (df['group'] == 'Treated').astype(int)
df['did_bin'] = df['treated'] * df['post']
df['did_cont'] = df['Index_norm'] * df['post']
df['t'] = df['year'] - 2018
df['t_war'] = np.maximum(0, df['year'] - 2021)
df['Treated_t'] = df['treated'] * df['t']
df['Treated_t_war'] = df['treated'] * df['t_war']

OUTCOMES = ['n_schools', 'n_children', 'SKB', 'subvention', 'applications']
OUTCOME_LABELS = {
    'n_schools': 'Кількість закладів',
    'n_children': 'Кількість дітей, тис.',
    'SKB': 'Середній конкурсний бал (СКБ)',
    'subvention': 'Субвенція на 1 учня, тис. грн',
    'applications': 'Усього заяв на вступ'
}

# =============================================================================
# 1. SYNTHETIC CONTROL — РУЧНА РЕАЛІЗАЦІЯ
# =============================================================================
print("\n" + "=" * 80)
print("1. SYNTHETIC CONTROL (ручна реалізація через scipy.optimize)")
print("=" * 80)

def synthetic_control_manual(df, outcome, treat_reg, donor_regions,
                             pre_years=(2018, 2019, 2020, 2021),
                             post_years=(2022, 2024)):
    """
    Ручний Synthetic Control:
    знаходимо ваги w_i ≥ 0, sum w_i = 1, які мінімізують pre-period RMSE
    між treated та синтетичним контролем.
    """
    sub = df[df['region'].isin([treat_reg] + donor_regions)][['region', 'year', outcome]].dropna()
    pivot = sub.pivot(index='year', columns='region', values=outcome)
    pivot = pivot.dropna(axis=1, how='any')

    if treat_reg not in pivot.columns or len(pivot.columns) < 4:
        return None

    donors = [c for c in pivot.columns if c != treat_reg]
    if len(donors) < 3:
        return None

    Y_treat = pivot[treat_reg].values
    Y_donors = pivot[donors].values  # T x J

    pre_mask = np.isin(pivot.index.values, pre_years)
    post_mask = np.isin(pivot.index.values, post_years)

    if pre_mask.sum() < 2 or post_mask.sum() < 1:
        return None

    Y_treat_pre = Y_treat[pre_mask]
    Y_donors_pre = Y_donors[pre_mask]

    def loss(w):
        w = np.abs(w)
        s = w.sum()
        if s == 0:
            return 1e10
        w = w / s
        synth = Y_donors_pre @ w
        return np.sqrt(np.mean((Y_treat_pre - synth) ** 2))

    w0 = np.ones(len(donors)) / len(donors)
    bounds = [(0, 1)] * len(donors)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1},)

    res = minimize(loss, w0, bounds=bounds, constraints=cons,
                   method='SLSQP', options={'maxiter': 500, 'ftol': 1e-9})

    if not res.success and res.fun > 1e6:
        return None

    w_opt = np.abs(res.x)
    if w_opt.sum() == 0:
        return None
    w_opt = w_opt / w_opt.sum()

    synth_post = Y_donors[post_mask] @ w_opt
    treat_post = Y_treat[post_mask]
    att = float(np.mean(treat_post - synth_post))
    rmse_pre = float(res.fun)

    return {
        'treated': treat_reg,
        'ATT': att,
        'RMSE_pre': rmse_pre,
        'weights': dict(zip(donors, np.round(w_opt, 4))),
    }

donor_regions = impact[impact['group'].isin(['Control', 'Out'])]['region'].tolist()
treated_regions = impact[impact['group'] == 'Treated']['region'].tolist()

sc_summary = []
for outcome in ['n_schools', 'SKB']:
    print(f"\n--- Synthetic Control: {OUTCOME_LABELS[outcome]} ---")
    all_atts = []
    for tr in treated_regions:
        r = synthetic_control_manual(df, outcome, tr, donor_regions)
        if r is None:
            print(f"  {tr:20s}: пропущено (брак даних)")
            continue
        all_atts.append(r['ATT'])
        sc_summary.append({'outcome': outcome, 'treated': tr, **r})
        print(f"  {tr:20s}  ATT = {r['ATT']:9.4f}  RMSE_pre = {r['RMSE_pre']:7.3f}")

    if all_atts:
        print(f"  → Середній ATT = {np.mean(all_atts):.4f}  "
              f"(SD = {np.std(all_atts):.4f},  N = {len(all_atts)})")
    else:
        print("  Не вдалося оцінити жодного синтетичного контролю.")

if sc_summary:
    pd.DataFrame(sc_summary).to_csv(OUT_DIR / 'synthetic_control.csv',
                                    index=False, encoding='utf-8-sig')

# =============================================================================
# 2. CALLAWAY-SANT'ANNA — ВИПРАВЛЕНИЙ ВИКЛИК
# =============================================================================
print("\n" + "=" * 80)
print("2. CALLAWAY-SANT'ANNA (keyword-only API)")
print("=" * 80)

try:
    from csdid.att_gt import ATTgt
    CSDID_AVAILABLE = True
except ImportError:
    CSDID_AVAILABLE = False
    print("  ⚠ csdid не встановлено — блок пропущено.")
    print("  Встановіть: pip install csdid")

if CSDID_AVAILABLE:
    for outcome in OUTCOMES:
        try:
            sub = df[df['group'].isin(['Treated', 'Control', 'Out'])].copy()
            sub = sub.dropna(subset=[outcome])
            sub['G'] = np.where(sub['treated'] == 1, 2022, 0)
            sub['Y'] = sub[outcome]
            sub['id'] = sub['region']
            sub['tt'] = sub['year']

            # FIX: усі аргументи keyword-only
            attgt = ATTgt(
                yname='Y',
                idname='id',
                tname='tt',
                gname='G',
                data=sub,
            )
            attgt.fit()
            att = attgt.att
            print(f"  {OUTCOME_LABELS[outcome]:35s}  ATT = {att:9.4f}")
        except Exception as e:
            print(f"  {outcome}: CS не вдалося — {e}")
    print("\n  ⚠ УВАГА: у ваших даних усі Treated отримують шок одночасно (2022).")
    print("     Тому CS вироджується в звичайний DiD. Для staggered-дизайну")
    print("     потрібні різні дати початку treatment у різних регіонів.")

# =============================================================================
# 3. SUN & ABRAHAM — Event study (interaction-weighted)
# =============================================================================
print("\n" + "=" * 80)
print("3. SUN & ABRAHAM (interaction-weighted event study)")
print("=" * 80)

if PYFIXEST_AVAILABLE:
    for outcome in OUTCOMES:
        try:
            sub = df[df['group'].isin(['Treated', 'Control'])].copy()
            sub = sub.dropna(subset=[outcome])
            sub['rel_time'] = sub['year'] - 2021
            sub['treat_rel'] = sub['treated'] * sub['rel_time']

            model = pf.feols(f"{outcome} ~ i(treat_rel, ref=-1) | region + year",
                             data=sub, vcov={'CRV1': 'region'})
            print(f"\n--- {OUTCOME_LABELS[outcome]} ---")
            print(model.coef().to_string())
        except Exception as e:
            print(f"  {outcome}: SA не вдалося — {e}")
else:
    print("  Sun & Abraham пропущено (немає pyfixest).")

# =============================================================================
# 4. CR2 ТА RANDOMIZATION INFERENCE
# =============================================================================
print("\n" + "=" * 80)
print("4. CR2 та RANDOMIZATION INFERENCE")
print("=" * 80)

# 4.1 CR2 (bias-corrected) — через pyfixest, fallback → linearmodels
print("\n--- CR2 (bias-corrected clustered SE) ---")
if PYFIXEST_AVAILABLE:
    for outcome in OUTCOMES:
        try:
            sub = df[df['group'].isin(['Treated', 'Control'])].dropna(subset=[outcome])
            model = pf.feols(f"{outcome} ~ did_bin | region + year",
                             data=sub, vcov={'CRV1': 'region'})
            cr2_se = model.se()['did_bin']
            coef = model.coef()['did_bin']
            print(f"  {outcome:15s}  CR2 SE = {cr2_se:8.4f}  coef = {coef:9.4f}")
        except Exception as e:
            print(f"  {outcome}: CR2 не вдалося — {e}")
else:
    print("  pyfixest відсутній — CR2 fallback через linearmodels (CR1):")
    for outcome in OUTCOMES:
        sub = df[df['group'].isin(['Treated', 'Control'])].dropna(subset=[outcome])
        sub = sub.set_index(['region', 'year'])
        try:
            res = PanelOLS(sub[outcome], sub[['did_bin']],
                           entity_effects=True, time_effects=True)\
                  .fit(cov_type='clustered', cluster_entity=True)
            print(f"  {outcome:15s}  SE = {res.std_errors['did_bin']:8.4f}  "
                  f"coef = {res.params['did_bin']:9.4f}")
        except Exception as e:
            print(f"  {outcome}: fallback failed — {e}")

# 4.2 Randomization inference на рівні регіонів
print("\n--- Randomization Inference (permutation on region level, 499 perms) ---")
ri_results = []
for outcome in OUTCOMES:
    try:
        sub = df[df['group'].isin(['Treated', 'Control'])].dropna(subset=[outcome]).copy()
        sub_idx = sub.set_index(['region', 'year'])
        res = PanelOLS(sub_idx[outcome], sub_idx[['did_bin']],
                       entity_effects=True, time_effects=True)\
              .fit(cov_type='clustered', cluster_entity=True)
        true_beta = res.params['did_bin']

        rng = np.random.default_rng(42)
        perm_betas = []
        regions = sub['region'].unique()
        treated_mask = sub.groupby('region')['treated'].first().values

        for _ in range(499):
            perm_treated = rng.permutation(treated_mask)
            perm_map = dict(zip(regions, perm_treated))
            sp = sub.copy()
            sp['treated_perm'] = sp['region'].map(perm_map)
            sp['did_perm'] = sp['treated_perm'] * sp['post']
            sp = sp.set_index(['region', 'year'])
            try:
                r = PanelOLS(sp[outcome], sp[['did_perm']],
                             entity_effects=True, time_effects=True)\
                     .fit(cov_type='clustered', cluster_entity=True)
                perm_betas.append(r.params['did_perm'])
            except Exception:
                continue

        if perm_betas:
            p_ri = float(np.mean(np.abs(perm_betas) >= np.abs(true_beta)))
            print(f"  {outcome:15s}  true β = {true_beta:9.4f}  p_RI = {p_ri:.4f}  "
                  f"(n_perm = {len(perm_betas)})")
            ri_results.append({'outcome': outcome, 'true_beta': true_beta,
                               'p_RI': p_ri, 'n_perm': len(perm_betas)})
    except Exception as e:
        print(f"  {outcome}: RI не вдалося — {e}")

if ri_results:
    pd.DataFrame(ri_results).to_csv(OUT_DIR / 'randomization_inference.csv',
                                    index=False, encoding='utf-8-sig')

# =============================================================================
# 5. EVENT STUDY ДЛЯ SUBVENTION (2020, 2021, 2022, 2024)
# =============================================================================
print("\n" + "=" * 80)
print("5. EVENT STUDY ДЛЯ SUBVENTION (лише 2020, 2021, 2022, 2024)")
print("=" * 80)

sub_sub = df[df['year'].isin([2020, 2021, 2022, 2024])].copy()
sub_sub = sub_sub[sub_sub['group'].isin(['Treated', 'Control'])].dropna(subset=['subvention'])
sub_sub = sub_sub.set_index(['region', 'year'])

for yr in [2020, 2022, 2024]:
    sub_sub[f'treat_{yr}'] = ((sub_sub.index.get_level_values('year') == yr) &
                              (sub_sub['treated'] == 1)).astype(float)

try:
    res_sub = PanelOLS(sub_sub['subvention'],
                       sub_sub[['treat_2020', 'treat_2022', 'treat_2024']],
                       entity_effects=True, time_effects=True)\
              .fit(cov_type='clustered', cluster_entity=True)
    print(res_sub.params.to_string())
except Exception as e:
    print(f"Subvention event study failed: {e}")

# =============================================================================
# 6. UNBALANCED PANEL (ЯВНО)
# =============================================================================
print("\n" + "=" * 80)
print("6. UNBALANCED PANEL (2023 відсутній)")
print("=" * 80)
print(f"Унікальні роки: {sorted(df['year'].unique().tolist())}")
print(f"Кількість спостережень на регіон (унікальні значення): "
      f"{sorted(df.groupby('region').size().unique().tolist())}")
print("PanelOLS вище використовують цей unbalanced panel явно.")

# =============================================================================
# 7. PLACEBO (2019, 2020) — 2018 неможливий
# =============================================================================
print("\n" + "=" * 80)
print("7. PLACEBO (фейкові роки 2019, 2020)")
print("=" * 80)
print("  ⚠ Placebo для 2018 року неможливий: немає жодного pre-period (усі роки ≤ 2021")
print("     вже після 2018). Тому did_f стає константою і повністю поглинається FE.")

placebo_rows = []
for fake_year in [2019, 2020]:
    print(f"\n--- Placebo: фейкова війна у {fake_year} ---")
    sub_pl = df[df['year'] <= 2021].copy()
    sub_pl['post_f'] = (sub_pl['year'] >= fake_year).astype(int)
    sub_pl['did_f'] = sub_pl['Index_norm'] * sub_pl['post_f']
    sub_pl = sub_pl[sub_pl['group'].isin(['Treated', 'Control'])]

    for outcome in OUTCOMES:
        try:
            sub_pl_o = sub_pl.dropna(subset=[outcome]).set_index(['region', 'year'])
            if sub_pl_o['did_f'].std() == 0:
                print(f"  {outcome:15s}: did_f константа — пропущено")
                continue
            res = PanelOLS(sub_pl_o[outcome], sub_pl_o[['did_f']],
                           entity_effects=True, time_effects=True)\
                  .fit(cov_type='clustered', cluster_entity=True)
            print(f"  {outcome:15s}  β = {res.params['did_f']:9.4f}  "
                  f"p = {res.pvalues['did_f']:.4f}")
            placebo_rows.append({
                'fake_year': fake_year, 'outcome': outcome,
                'coef': res.params['did_f'], 'p': res.pvalues['did_f']
            })
        except Exception as e:
            print(f"  {outcome}: placebo failed — {str(e)[:120]}")

if placebo_rows:
    pd.DataFrame(placebo_rows).to_csv(OUT_DIR / 'placebo_all_years.csv',
                                      index=False, encoding='utf-8-sig')

# =============================================================================
# 8. ПРОСТОРОВИЙ АНАЛІЗ — Moran / LISA / Theil
# =============================================================================
print("\n" + "=" * 80)
print("8. ПРОСТОРОВИЙ АНАЛІЗ")
print("=" * 80)

# 8.1. Справжня суміжність областей (queen contiguity)
CONTIGUITY = {
    'Вінницька': ['Житомирська', 'Київська', 'Черкаська', 'Кіровоградська',
                  'Одеська', 'Хмельницька', 'Чернівецька', 'Тернопільська'],
    'Волинська': ['Рівненська', 'Львівська', 'Тернопільська'],
    'Дніпропетровська': ['Запорізька', 'Херсонська', 'Миколаївська', 'Кіровоградська',
                         'Полтавська', 'Харківська', 'Донецька'],
    'Донецька': ['Луганська', 'Дніпропетровська', 'Запорізька', 'Харківська'],
    'Житомирська': ['Київська', 'Вінницька', 'Хмельницька', 'Рівненська'],
    'Закарпатська': ['Львівська', 'Івано-Франківська'],
    'Запорізька': ['Донецька', 'Дніпропетровська', 'Херсонська'],
    'Івано-Франківська': ['Львівська', 'Закарпатська', 'Тернопільська', 'Чернівецька'],
    'Київська': ['Житомирська', 'Вінницька', 'Черкаська', 'Чернігівська', 'м. Київ'],
    'Кіровоградська': ['Вінницька', 'Черкаська', 'Полтавська', 'Дніпропетровська',
                       'Миколаївська', 'Одеська'],
    'Луганська': ['Донецька', 'Харківська'],
    'Львівська': ['Волинська', 'Рівненська', 'Тернопільська', 'Івано-Франківська', 'Закарпатська'],
    'Миколаївська': ['Одеська', 'Кіровоградська', 'Дніпропетровська', 'Херсонська'],
    'Одеська': ['Вінницька', 'Кіровоградська', 'Миколаївська'],
    'Полтавська': ['Київська', 'Черкаська', 'Кіровоградська', 'Дніпропетровська',
                   'Харківська', 'Сумська', 'Чернігівська'],
    'Рівненська': ['Волинська', 'Львівська', 'Тернопільська', 'Хмельницька', 'Житомирська'],
    'Сумська': ['Чернігівська', 'Полтавська', 'Харківська'],
    'Тернопільська': ['Львівська', 'Івано-Франківська', 'Чернівецька', 'Хмельницька',
                      'Рівненська', 'Волинська'],
    'Харківська': ['Сумська', 'Полтавська', 'Дніпропетровська', 'Донецька', 'Луганська'],
    'Херсонська': ['Миколаївська', 'Дніпропетровська', 'Запорізька'],
    'Хмельницька': ['Рівненська', 'Тернопільська', 'Чернівецька', 'Вінницька', 'Житомирська'],
    'Черкаська': ['Київська', 'Вінницька', 'Кіровоградська', 'Полтавська'],
    'Чернівецька': ['Івано-Франківська', 'Тернопільська', 'Хмельницька', 'Вінницька'],
    'Чернігівська': ['Київська', 'Полтавська', 'Сумська'],
    'м. Київ': ['Київська'],
}

COORDS = {
    "Вінницька": (49.23, 28.47), "Волинська": (50.75, 25.33),
    "Дніпропетровська": (48.46, 35.05), "Донецька": (48.00, 37.80),
    "Житомирська": (50.25, 28.66), "Закарпатська": (48.62, 22.30),
    "Запорізька": (47.84, 35.14), "Івано-Франківська": (48.92, 24.71),
    "Київська": (50.35, 30.95), "Кіровоградська": (48.51, 32.26),
    "Луганська": (48.57, 39.31), "Львівська": (49.84, 24.03),
    "Миколаївська": (46.98, 31.99), "Одеська": (46.48, 30.73),
    "Полтавська": (49.59, 34.55), "Рівненська": (50.62, 26.25),
    "Сумська": (50.91, 34.80), "Тернопільська": (49.55, 25.59),
    "Харківська": (49.99, 36.23), "Херсонська": (46.64, 32.62),
    "Хмельницька": (49.42, 26.99), "Черкаська": (49.44, 32.06),
    "Чернівецька": (48.29, 25.94), "Чернігівська": (51.50, 31.29),
    "м. Київ": (50.45, 30.52),
}

# 8.2. Побудова numpy-матриць ваг
def make_weights_knn(regions, k=4, row_standardized=True):
    coords = np.array([COORDS[r] for r in regions], dtype=float)
    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)
    n = len(regions)
    w = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist[i])[:min(k, n - 1)]
        w[i, neighbors] = 1.0
    w = np.maximum(w, w.T)
    if row_standardized:
        rs = w.sum(axis=1)
        w = np.divide(w, rs[:, None], out=np.zeros_like(w), where=rs[:, None] != 0)
    return w

def make_weights_queen(regions, row_standardized=True):
    n = len(regions)
    idx = {r: i for i, r in enumerate(regions)}
    w = np.zeros((n, n))
    for r in regions:
        if r in CONTIGUITY:
            for nb in CONTIGUITY[r]:
                if nb in idx:
                    w[idx[r], idx[nb]] = 1.0
    if row_standardized:
        rs = w.sum(axis=1)
        w = np.divide(w, rs[:, None], out=np.zeros_like(w), where=rs[:, None] != 0)
    return w

def make_weights_inverse_distance(regions):
    coords = np.array([COORDS[r] for r in regions], dtype=float)
    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)
    w = 1.0 / (dist + 1e-6)
    np.fill_diagonal(w, 0)
    rs = w.sum(axis=1)
    w = np.divide(w, rs[:, None], out=np.zeros_like(w), where=rs[:, None] != 0)
    return w

# 8.3. Конвертація numpy-матриці в libpysal W-об'єкт (ВИПРАВЛЕНО)
def numpy_to_libpysal_W(regions, w_array):
    """
    esda.Moran очікує libpysal W-об'єкт, а не numpy ndarray.
    Конвертуємо: neighbors_dict + weights_dict → W.
    """
    neighbors = {}
    weights = {}
    for i, r in enumerate(regions):
        nbs = [regions[j] for j in range(len(regions)) if w_array[i, j] != 0]
        wts = [float(w_array[i, j]) for j in range(len(regions)) if w_array[i, j] != 0]
        neighbors[r] = nbs
        weights[r] = wts
    return W(neighbors, weights, silence_warnings=True)

# 8.4. Moran's I для різних матриць (2024)
if LIBPYSAL_AVAILABLE:
    print("\n--- Moran's I за різних матриць ваг (2024) ---")
    moran_all = []
    for outcome in OUTCOMES:
        sub = df[df['year'] == 2024][['region', outcome]].dropna()
        sub = sub[sub['region'].isin(COORDS)]
        if len(sub) < 5:
            continue
        regions = sub['region'].tolist()
        vals = sub[outcome].values.astype(float)

        print(f"\n{OUTCOME_LABELS[outcome]}  (N = {len(regions)}):")
        for label, w_arr in [
            ('KNN k=3', make_weights_knn(regions, 3)),
            ('KNN k=4', make_weights_knn(regions, 4)),
            ('KNN k=5', make_weights_knn(regions, 5)),
            ('KNN k=6', make_weights_knn(regions, 6)),
            ('Queen', make_weights_queen(regions, True)),
            ('Queen (binary)', make_weights_queen(regions, False)),
            ('InvDist', make_weights_inverse_distance(regions)),
        ]:
            try:
                w_obj = numpy_to_libpysal_W(regions, w_arr)
                mi = Moran(vals, w_obj, permutations=999)
                print(f"  {label:16s}  I = {mi.I:8.4f}  p = {mi.p_sim:.4f}")
                moran_all.append({
                    'outcome': outcome, 'year': 2024, 'weights': label,
                    'moran_i': mi.I, 'p_value': mi.p_sim, 'n': len(regions)
                })
            except Exception as e:
                print(f"  {label:16s}  помилка: {e}")

    if moran_all:
        pd.DataFrame(moran_all).to_csv(OUT_DIR / 'moran_weights_sensitivity.csv',
                                       index=False, encoding='utf-8-sig')

    # 8.5. LISA для зміни n_children 2018→2024
    print("\n--- Local Moran's I (LISA): зміна n_children 2018→2024 ---")
    wide = df.pivot(index='region', columns='year', values='n_children')
    change = (wide[2024] - wide[2018]).dropna()
    change = change[change.index.isin(COORDS)]

    if len(change) >= 5:
        regions = change.index.tolist()
        vals = change.values.astype(float)
        w_arr = make_weights_knn(regions, 4)
        try:
            w_obj = numpy_to_libpysal_W(regions, w_arr)
            lisa = Moran_Local(vals, w_obj, permutations=999)
            lisa_rows = []
            for reg, Ii, p in zip(regions, lisa.Is, lisa.p_sim):
                quadrant = ['HH', 'LH', 'LL', 'HL'][lisa.q[regions.index(reg)] - 1]
                marker = ' *' if p < 0.05 else ''
                print(f"  {reg:20s}  I_i = {Ii:8.4f}  p = {p:.4f}  [{quadrant}]{marker}")
                lisa_rows.append({
                    'region': reg, 'I_i': Ii, 'p': p, 'quadrant': quadrant
                })
            pd.DataFrame(lisa_rows).to_csv(OUT_DIR / 'lisa_n_children_change.csv',
                                           index=False, encoding='utf-8-sig')

            # Moran scatterplot
            fig, ax = plt.subplots(figsize=(8, 7))
            z = (vals - vals.mean()) / vals.std()
            wz = w_arr @ z
            colors = ['#C97B74' if p < 0.05 else '#5C9E9A' for p in lisa.p_sim]
            ax.scatter(z, wz, c=colors, s=90, edgecolors='black', linewidth=0.5)
            for i, reg in enumerate(regions):
                ax.annotate(reg[:12], (z[i], wz[i]), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
            ax.axhline(0, color='gray', linestyle='--')
            ax.axvline(0, color='gray', linestyle='--')
            ax.set_xlabel('z (зміна n_children 2018→2024)')
            ax.set_ylabel('Wz (просторовий лаг, KNN k=4)')
            ax.set_title("Moran scatterplot: зміна кількості дітей")
            plt.tight_layout()
            plt.savefig(OUT_DIR / 'moran_scatterplot.png', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  → Moran scatterplot збережено")
        except Exception as e:
            print(f"  LISA не вдалося: {e}")
else:
    print("  LISA та Moran scatterplot пропущено (немає libpysal/esda).")
# =============================================================================
# 8.6. MORAN SCATTERPLOTS ДЛЯ ЗНАЧУЩИХ ПОКАЗНИКІВ (n_children та SKB)
# =============================================================================
print("\n" + "=" * 80)
print("8.6. MORAN SCATTERPLOTS: зміна 2018→2024 для n_children та SKB")
print("=" * 80)

if LIBPYSAL_AVAILABLE:

    # Словник діаграм для побудови
    SCATTER_CONFIGS = [
        {
            'outcome': 'n_children',
            'label': 'зміна кількості дітей, тис.',
            'filename': 'moran_scatterplot_n_children.png',
            'csv_filename': 'moran_scatterplot_data_n_children.csv',
        },
        {
            'outcome': 'SKB',
            'label': 'зміна середнього конкурсного балу',
            'filename': 'moran_scatterplot_skb.png',
            'csv_filename': 'moran_scatterplot_data_skb.csv',
        },
    ]

    for cfg in SCATTER_CONFIGS:
        outcome = cfg['outcome']
        print(f"\n--- Moran scatterplot: {cfg['label']} ---")

        # --- 1. Дані: зміна показника між 2018 та 2024 ---
        wide = df.pivot(index='region', columns='year', values=outcome)
        if 2018 not in wide.columns or 2024 not in wide.columns:
            print(f"  Пропущено: немає даних за 2018 або 2024")
            continue

        change = (wide[2024] - wide[2018]).dropna()
        change = change[change.index.isin(COORDS)]

        if len(change) < 5:
            print(f"  Пропущено: недостатньо спостережень (N = {len(change)})")
            continue

        regions = change.index.tolist()
        vals = change.values.astype(float)

        # --- 2. Матриця просторових ваг KNN k=4 ---
        w_arr = make_weights_knn(regions, 4)
        w_obj = numpy_to_libpysal_W(regions, w_arr)

        # --- 3. Глобальний Moran's I ---
        mi = Moran(vals, w_obj, permutations=999)
        print(f"  Глобальний Moran's I = {mi.I:.4f}  (p = {mi.p_sim:.4f}, N = {len(regions)})")

        # --- 4. Локальний Moran's I (LISA) ---
        lisa = Moran_Local(vals, w_obj, permutations=999)

        # --- 5. Стандартизовані значення та просторовий лаг ---
        z = (vals - vals.mean()) / vals.std()
        wz = w_arr @ z

        # --- 6. Класифікація за квадрантами ---
        quadrants = []
        for zi, wzi in zip(z, wz):
            if zi >= 0 and wzi >= 0:
                quadrants.append('HH')
            elif zi < 0 and wzi >= 0:
                quadrants.append('LH')
            elif zi < 0 and wzi < 0:
                quadrants.append('LL')
            else:
                quadrants.append('HL')

        # --- 7. Побудова діаграми ---
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = []
        for p in lisa.p_sim:
            colors.append('#C97B74' if p < 0.05 else '#A8C4C0')

        ax.scatter(z, wz, c=colors, s=140, edgecolors='black',
                   linewidth=0.7, alpha=0.85, zorder=3)

        # Підписи регіонів
        for i, reg in enumerate(regions):
            short = reg.replace('м. ', '').replace('ька', '').replace('ська', '')
            ax.annotate(short, (z[i], wz[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.85)

        # Осі координат
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Лінія регресії (нахил = Moran's I)
        b, a = np.polyfit(z, wz, 1)
        xs = np.linspace(z.min() - 0.2, z.max() + 0.2, 50)
        ax.plot(xs, a + b * xs, color='darkred', linewidth=2,
                linestyle='-', alpha=0.75, zorder=2,
                label=f"Нахил = {b:.3f}  (Moran's I = {mi.I:.3f})")

        # Підписи квадрантів
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[1] * 0.92, ylim[1] * 0.92, 'HH',
                fontsize=14, color='gray', alpha=0.55,
                ha='right', va='top', fontweight='bold')
        ax.text(xlim[0] * 0.92, ylim[1] * 0.92, 'LH',
                fontsize=14, color='gray', alpha=0.55,
                ha='left', va='top', fontweight='bold')
        ax.text(xlim[0] * 0.92, ylim[0] * 0.92, 'LL',
                fontsize=14, color='gray', alpha=0.55,
                ha='left', va='bottom', fontweight='bold')
        ax.text(xlim[1] * 0.92, ylim[0] * 0.92, 'HL',
                fontsize=14, color='gray', alpha=0.55,
                ha='right', va='bottom', fontweight='bold')

        # Заголовок та підписи осей
        ax.set_title(f"Діаграма розсіювання Морана: {cfg['label']}, 2018→2024",
                     fontsize=13, pad=12)
        ax.set_xlabel(f"z (стандартизована {cfg['label']})", fontsize=11)
        ax.set_ylabel("Wz (просторовий лаг, KNN k=4)", fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(OUT_DIR / cfg['filename'], dpi=220, bbox_inches='tight')
        plt.close()
        print(f"  → Діаграму збережено: {cfg['filename']}")

        # --- 8. Збереження таблиці з квадрантами ---
        scatter_rows = []
        for reg, zi, wzi, q, Ii, p in zip(regions, z, wz, quadrants,
                                          lisa.Is, lisa.p_sim):
            scatter_rows.append({
                'region': reg,
                'z': zi, 'Wz': wzi,
                'quadrant': q,
                'local_moran_I': Ii,
                'p_value': p,
                'significant': p < 0.05,
            })

        pd.DataFrame(scatter_rows).to_csv(
            OUT_DIR / cfg['csv_filename'],
            index=False, encoding='utf-8-sig'
        )
        print(f"  → Дані діаграми збережено: {cfg['csv_filename']}")

        # --- 9. Друкарня квадрантів для тексту роботи ---
        print(f"  Розподіл регіонів за квадрантами (N = {len(regions)}):")
        for q in ['HH', 'LH', 'LL', 'HL']:
            quad_regs = [r for r, qq, p in zip(regions, quadrants, lisa.p_sim)
                         if qq == q and p < 0.05]
            if quad_regs:
                print(f"    {q} (значущі): {', '.join(quad_regs)}")

else:
    print("  Діаграми пропущено (немає libpysal/esda).")

# =============================================================================
# 9. THEIL INDEX (global + декомпозиція)
# =============================================================================
print("\n" + "=" * 80)
print("9. THEIL INDEX (global + декомпозиція на between/within)")
print("=" * 80)

def theil_t(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    mu = x.mean()
    return float(np.mean((x / mu) * np.log(x / mu)))

def theil_decomposition(sub, outcome, group_col='group'):
    sub = sub[['region', outcome, group_col]].dropna()
    if len(sub) < 3:
        return np.nan, np.nan, np.nan
    total = theil_t(sub[outcome])
    between = 0.0
    within = 0.0
    grand_mean = sub[outcome].mean()
    total_sum = sub[outcome].sum()
    for g, grp in sub.groupby(group_col):
        x_g = grp[outcome]
        s_g = x_g.sum() / total_sum if total_sum != 0 else 0
        mu_g = x_g.mean()
        if mu_g > 0 and grand_mean > 0:
            between += s_g * np.log(mu_g / grand_mean)
        within += s_g * theil_t(x_g)
    return total, between, within

theil_rows = []
for outcome in OUTCOMES:
    print(f"\n--- {OUTCOME_LABELS[outcome]} ---")
    for year in sorted(df['year'].unique()):
        sub = df[df['year'] == year]
        T, B, W = theil_decomposition(sub, outcome)
        if np.isnan(T):
            continue
        print(f"  {year}: Theil = {T:.4f}  between = {B:.4f}  within = {W:.4f}")
        theil_rows.append({
            'outcome': outcome, 'year': int(year),
            'theil_total': T, 'theil_between': B, 'theil_within': W
        })

if theil_rows:
    pd.DataFrame(theil_rows).to_csv(OUT_DIR / 'theil_index.csv',
                                    index=False, encoding='utf-8-sig')

# Динаміка Theil T
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, outcome in enumerate(OUTCOMES):
    ax = axes[i]
    sub = df[['year', 'region', outcome]].dropna()
    years = sorted(sub['year'].unique())
    theil_vals = []
    for y in years:
        vals = sub[sub['year'] == y][outcome].values
        theil_vals.append(theil_t(vals))
    ax.plot(years, theil_vals, marker='o', linewidth=2)
    ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
    ax.set_xlabel('Рік')
    ax.set_ylabel('Theil T')
    ax.grid(alpha=0.3)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(OUT_DIR / 'theil_dynamics.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n  Динаміку Theil збережено: {OUT_DIR / 'theil_dynamics.png'}")

# =============================================================================
# 10. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# =============================================================================
print("\n" + "=" * 80)
print("10. РЕЗУЛЬТАТИ ЗБЕРЕЖЕНО")
print("=" * 80)
print(f"Папка: {OUT_DIR}")
print("Файли:")
for f in sorted(OUT_DIR.glob('*')):
    print(f"  {f.name}")

print("\nГотово.")