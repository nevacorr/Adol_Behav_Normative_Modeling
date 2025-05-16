
def get_agegroup(row):
    subject = row['subject']
    visit = row['visit']

    if 100 <= subject < 200:
        return 9 if visit == 1 else 12
    elif 200 <= subject < 300:
        return 11 if visit == 1 else 14
    elif 300 <= subject < 400:
        return 13 if visit == 1 else 16
    elif 400 <= subject < 500:
        return 15 if visit == 1 else 18
    elif 500 <= subject < 600:
        return 17 if visit == 1 else 20

def summarize_visit(df, conv_days_to_years):
    # Mean and std age by gender and agegroup
    agegroup_gender_summary = (
        df.groupby(['gender', 'agegroup'])['agedays']
        .agg(N='count', mean_days='mean', std_days='std')
        .assign(
            mean_years=lambda x: x['mean_days'] / conv_days_to_years,
            std_years=lambda x: x['std_days'] / conv_days_to_years
        )
        .reset_index()
    )
    return agegroup_gender_summary

def print_summaries(agegroup_gender_summary, visit_name="Visit"):

    print(f"\n--- {visit_name} Age Summary by Gender and Age Group ---")
    # Format floats to 2 decimals for readability
    formatted = agegroup_gender_summary.copy()
    formatted['mean_days'] = formatted['mean_days'].round(2)
    formatted['std_days'] = formatted['std_days'].round(2)
    formatted['mean_years'] = formatted['mean_years'].round(2)
    formatted['std_years'] = formatted['std_years'].round(2)

    print(formatted.to_string(index=False))