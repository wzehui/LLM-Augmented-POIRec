import gc
import pandas as pd
import os.path as osp
from tqdm import tqdm
from geopy.distance import geodesic
from preprocessing import save_csv


def load_csv(load_path, file_name):
    """Loads a CSV file from the specified directory into a pandas DataFrame."""
    file_path = osp.join(load_path, file_name)
    if not osp.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    try:
        # Attempt to read with 'c' engine for performance
        return pd.read_csv(file_path)
    except pd.errors.ParserError:
        print(
            f"Error using 'c' engine for '{file_name}', retrying with 'python' engine.")
        return pd.read_csv(file_path, engine='python')
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_name}': {e}")
        raise

def analyse_statistics(data, data_filtered):
    """Analyzes and prints statistics for the original and filtered datasets."""
    checkin_count_filtered = len(data_filtered)
    users_count_filtered = data_filtered['user_id'].nunique()
    business_count_filtered = data_filtered['business_id'].nunique()

    checkin_count = len(data)
    users_count = data['user_id'].nunique()
    business_count = data['business_id'].nunique()

    percentage_checkin_filtered = (checkin_count_filtered / checkin_count) * 100
    percentage_users_filtered = (users_count_filtered / users_count) * 100
    percentage_business_filtered = (business_count_filtered / business_count) * 100

    sequence_length = data.groupby('user_id').size()
    sequence_length_filtered = data_filtered.groupby('user_id').size()

    print(f"Check-in counts: {checkin_count:.0f} -->"
          f" {checkin_count_filtered:.0f}")
    print(f"User counts: {users_count:.0f} -->"
          f" {users_count_filtered:.0f}")
    print(f"Business counts: {business_count:.0f} -->"
          f" {business_count_filtered:.0f}")
    print(f"Percentage of check-in retained:"
          f" {percentage_checkin_filtered:.2f}%")
    print(f"Percentage of users retained: {percentage_users_filtered:.2f}%")
    print(f"Percentage of businesses retained: "
          f"{percentage_business_filtered:.2f}%")
    print(f"Average sequential length: "
          f"{sequence_length.mean():.1f} --> "
          f"{sequence_length_filtered.mean():.1f}")
    print(f"max/min sequential length: {sequence_length_filtered.max():.0f} / "
          f"{sequence_length_filtered.min():.0f} \n")

def identify_bot_user(checkin_df, user_id, business_df=None):
    """Identifies potential bot users by analyzing their speed between check-ins."""
    fastest_airplane_speed = 900
    data = checkin_df[checkin_df['user_id'] == user_id]
    data = data.sort_values(by=['date'])

    if business_df is not None:
        data = pd.merge(data, business_df[['business_id', 'latitude',
                                           'longitude']], on='business_id', how='left')

    data['prev_latitude'] = data['latitude'].shift()
    data['prev_longitude'] = data['longitude'].shift()
    data['distance'] = data.apply(
        lambda row: geodesic(
            (row['prev_latitude'], row['prev_longitude']),
            (row['latitude'], row['longitude'])
        ).kilometers if pd.notnull(row['prev_latitude']) else 0,
        axis=1
    )

    data['time_diff'] = (data['date'] - data['date'].shift()).dt.total_seconds() / 3600
    data['speed'] = data['distance'] / data['time_diff']
    ratio_above_airplane_speed = ((data['speed'] > fastest_airplane_speed).sum() /
                            len(data['speed']))
    return ratio_above_airplane_speed

if __name__ == "__main__":
    WORKING_DIR = './yelp'
    load_path = osp.join(WORKING_DIR, 'csv')
    save_path = osp.join(WORKING_DIR, 'csv')

    checkin_df = load_csv(load_path, 'checkin.csv')
    user_df = load_csv(load_path, 'user.csv')
    business_df = load_csv(load_path, 'business.csv')

    # Smartphones became popular in 2007
    checkin_df['date'] = pd.to_datetime(checkin_df['date'])
    checkin_filtered = checkin_df[
        (checkin_df['date'].dt.year >= 2017) &
        (checkin_df['date'].dt.year <= 2019)
        ]
    print('Timespan filtering')
    analyse_statistics(checkin_df, checkin_filtered)
    checkin_df = checkin_filtered

    checkin_counts = checkin_df.groupby('user_id').size()
    suspicious_user = checkin_counts[checkin_counts > 300].index
    results = []
    for user_id in tqdm(suspicious_user, desc="Processing Users"):
        ratio_above_airplane_speed = identify_bot_user(
            checkin_filtered, user_id, business_df)
        results.append(
            {'user_id': user_id, 'above_airplane_speed': ratio_above_airplane_speed})
    filtered_df = pd.DataFrame(results)
    filtered_df = filtered_df[filtered_df['above_airplane_speed'] > 0]
    checkin_filtered = checkin_df[~checkin_df['user_id'].isin(filtered_df['user_id'])]
    print('Remove bot user')
    analyse_statistics(checkin_df, checkin_filtered)
    checkin_df = checkin_filtered

    user_id_filtered = set(checkin_df['user_id'].unique()) - set(user_df['user_id'].unique())
    checkin_filtered = checkin_df[~checkin_df['user_id'].isin(user_id_filtered)]
    print('Remove user which not in user.json')
    analyse_statistics(checkin_df, checkin_filtered)
    checkin_df = checkin_filtered

    MIN_NUM_INTERACTIONS = 10
    pre_len_checkin = len(checkin_df)
    while True:
        print('Iteration')
        business_to_keep = {business for business, value_count in checkin_filtered[
            'business_id'].value_counts().items() if value_count >=
        MIN_NUM_INTERACTIONS}
        checkin_filtered = checkin_filtered[checkin_filtered['business_id'].isin(
            business_to_keep)]
        user_to_keep = {user for user, value_count in checkin_filtered[
            'user_id'].value_counts().items() if value_count >= MIN_NUM_INTERACTIONS}
        checkin_filtered = checkin_filtered[checkin_filtered['user_id'].isin(
            user_to_keep)]
        if pre_len_checkin == len(checkin_filtered):
            break
        pre_len_checkin = len(checkin_filtered)
    print(f'Remove business and user with less than {MIN_NUM_INTERACTIONS} interactions')
    analyse_statistics(checkin_df, checkin_filtered)
    checkin_df = checkin_filtered

    save_csv(checkin_df, save_path, 'checkin_filtered.csv')

    user_df = user_df[user_df['user_id'].isin(checkin_df['user_id'])]
    existing_user_id = set(user_df['user_id'])
    user_df['friends'] = user_df['friends'].fillna('')
    user_df['original_friends_count'] = user_df['friends'].apply(
        lambda x: len(x.split(', ')) if isinstance(x, str) else 0
    )
    user_df['friends'] = user_df['friends'].apply(lambda x: ', '.join(
        user for user in x.split(', ') if user in existing_user_id))
    user_df['filtered_friends_count'] = user_df['friends'].apply(
        lambda x: len(x.split(', ')) if isinstance(x, str) else 0
    )
    print(f"friend number from {user_df['original_friends_count'].mean()} to "
          f"{user_df['filtered_friends_count'].mean()}")
    save_csv(user_df, save_path, 'user_filtered.csv')
    del user_df
    gc.collect()

    business_df = business_df[business_df['business_id'].isin(checkin_df['business_id'])]
    save_csv(business_df, save_path, 'business_filtered.csv')
    del business_df
    gc.collect()

    review_df = load_csv(load_path, 'review.csv')
    review_df = review_df[review_df['business_id'].isin(checkin_df['business_id'])]
    review_df = review_df[review_df['user_id'].isin(checkin_df['user_id'])]
    save_csv(review_df, save_path, 'review_filtered.csv')
    del review_df
    gc.collect()

    photo_df = load_csv(load_path, 'photo.csv')
    photo_df = photo_df[photo_df['business_id'].isin(checkin_df['business_id'])]
    save_csv(photo_df, save_path, 'photo_filtered.csv')
    del photo_df
    gc.collect()

    checkin_distribution_df = load_csv(load_path, 'checkin_distribution.csv')
    checkin_distribution_df = checkin_distribution_df[checkin_distribution_df[
        'business_id'].isin(checkin_df['business_id'])]
    save_csv(checkin_distribution_df, save_path, 'checkin_distribution_filtered.csv')
    del checkin_distribution_df
    gc.collect()

    tip_df = load_csv(load_path, 'tip.csv')
    tip_df = tip_df[tip_df['business_id'].isin(checkin_df['business_id'])]
    save_csv(tip_df, save_path, 'tip_filtered.csv')
    del tip_df
    gc.collect()