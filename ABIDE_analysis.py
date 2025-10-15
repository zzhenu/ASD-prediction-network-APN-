import pandas as pd
import numpy as np

# Read the subject_IDs.txt file
with open('../subject_IDs.txt', 'r') as f:
    subject_ids = [int(line.strip()) for line in f]

# Read the CSV file
df = pd.read_csv('../data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv')

# Filter out SUB_IDs that are in the subject_ids list
filtered_df = df[df['SUB_ID'].isin(subject_ids)].copy()

# Define diagnostic group labels
def map_dx_group(dx):
    if dx == 1:
        return 'ASD'
    elif dx == 2:
        return 'TC'
    return 'Unknown'

# Add a diagnostic group column
filtered_df['DX_LABEL'] = filtered_df['DX_GROUP'].map(map_dx_group)


# Merge specified sites
def merge_sites(site_id):
    if site_id in ['LEUVEN_1', 'LEUVEN_2']:
        return 'LEUVEN'
    elif site_id in ['UCLA_1', 'UCLA_2']:
        return 'UCLA'
    elif site_id in ['UM_1', 'UM_2']:
        return 'UM'
    return site_id


# Apply site merging
filtered_df['SITE_ID'] = filtered_df['SITE_ID'].map(merge_sites)

# Fill missing or -9999 values in the FIQ column with the column's mean
mean_fiq = filtered_df['FIQ'][filtered_df['FIQ'] != -9999].mean()
filtered_df['FIQ'] = filtered_df['FIQ'].apply(lambda x: mean_fiq if x == -9999 or pd.isna(x) else x)

# Group by the merged sites and compute statistics
grouped = filtered_df.groupby('SITE_ID')

# Create the results DataFrame
results = []
for site, group in grouped:
    asd_count = group[group['DX_LABEL'] == 'ASD'].shape[0]
    tc_count = group[group['DX_LABEL'] == 'TC'].shape[0]

    avg_age = group['AGE_AT_SCAN'].mean()
    std_age = group['AGE_AT_SCAN'].std()
    avg_fiq = group['FIQ'].mean()
    std_fiq = group['FIQ'].std()

    age_str = f"{round(avg_age, 2)} ({round(std_age, 2)})" if not np.isnan(avg_age) else 'N/A'
    fiq_str = f"{round(avg_fiq, 2)} ({round(std_fiq, 2)})" if not np.isnan(avg_fiq) else 'N/A'

    results.append({
        'SITE_ID': site,
        'ASD_Count': asd_count,
        'TC_Count': tc_count,
        'Total_Subjects': asd_count + tc_count,
        'Average_AGE_AT_SCAN': age_str,
        'Average_FIQ': fiq_str
    })

results_df = pd.DataFrame(results)

total_asd = filtered_df[filtered_df['DX_LABEL'] == 'ASD'].shape[0]
total_tc = filtered_df[filtered_df['DX_LABEL'] == 'TC'].shape[0]
total_subjects = filtered_df.shape[0]
total_avg_age = filtered_df['AGE_AT_SCAN'].mean()
total_std_age = filtered_df['AGE_AT_SCAN'].std()
total_avg_fiq = filtered_df['FIQ'].mean()
total_std_fiq = filtered_df['FIQ'].std()

total_row = {
    'SITE_ID': 'TOTAL',
    'ASD_Count': total_asd,
    'TC_Count': total_tc,
    'Total_Subjects': total_subjects,
    'Average_AGE_AT_SCAN': f"{round(total_avg_age, 2)} ({round(total_std_age, 2)})" if not np.isnan(
        total_avg_age) else 'N/A',
    'Average_FIQ': f"{round(total_avg_fiq, 2)} ({round(total_std_fiq, 2)})" if not np.isnan(total_avg_fiq) else 'N/A'
}

results_df = pd.concat([results_df, pd.DataFrame([total_row])], ignore_index=True)

results_df = results_df[['SITE_ID', 'ASD_Count', 'TC_Count', 'Total_Subjects',
                         'Average_AGE_AT_SCAN', 'Average_FIQ']]

# Print the results
print("Site statistics results (merged sites including total):")
print(results_df.to_string(index=False))

# Save the results to a CSV file (optional)
results_df.to_csv('site_statistics_with_merge_and_total.csv', index=False)
print("\nThe results have been saved to site_statistics_with_merge_and_total.csv")
