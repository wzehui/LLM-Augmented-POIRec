import re
import os
import gc
import json
import pandas as pd
from tqdm import tqdm
import os.path as osp


def load_json(file_path, file_name):
    """Reads a JSON file line by line and returns the data as a list."""
    data = []
    with open(osp.join(file_path, file_name), 'r', encoding='utf-8') as file:
        for line_number, line in tqdm(enumerate(file, start=1),
                                      desc=f"Reading {file_name}"):
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.decoder.JSONDecodeError as e:
                print(f"\nError decoding JSON at line {line_number}: {e}")
    return data

def clean_text(text):
    """Replaces newlines and carriage returns in the text with spaces."""
    if pd.notna(text):
        return text.replace('\n', ' ').replace('\r', ' ')
    return text

def reshape_review(data):
    """Processes review data, adding reactions and cleaning text, and splits it into checkin and review DataFrames."""
    data['reactions'] = data['useful'] + data['funny'] + data['cool']
    data['text'] = data['text'].apply(clean_text)
    checkin_df = data[['review_id', 'user_id', 'business_id', 'stars', 'date']]
    review_df = data[['review_id', 'user_id', 'business_id', 'text', 'stars', 'reactions']]
    return checkin_df, review_df

def reshape_user(data):
    """Processes user data, calculating reactions and compliments, and selecting relevant fields."""
    data['reactions'] = data['useful'] + data['funny'] + data['cool']
    data['compliments'] = (
        data['compliment_hot'] + data['compliment_more'] +
        data['compliment_profile'] + data['compliment_cute'] +
        data['compliment_list'] + data['compliment_note'] +
        data['compliment_plain'] + data['compliment_cool'] +
        data['compliment_funny'] + data['compliment_writer'] +
        data['compliment_photos']
    )
    user_df = data[[
        'user_id', 'name', 'review_count', 'yelping_since',
        'reactions', 'elite', 'friends', 'fans', 'average_stars', 'compliments'
    ]]
    return user_df

def save_csv(data, output_path, file_name):
    """Saves a DataFrame as a CSV file to the specified path."""
    if not osp.exists(output_path):
        os.makedirs(output_path)
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    file_name = re.sub(r'[^\w\s.-_]', '', file_name)
    data.to_csv(osp.join(output_path, file_name), index=False, encoding='utf-8')

if __name__ == "__main__":
    WORKING_DIR = './yelp'
    load_path = osp.join(WORKING_DIR, 'raw')
    save_path = osp.join(WORKING_DIR, 'csv')

    data_interaction = load_json(load_path, 'yelp_academic_dataset_review.json')
    interaction_df = pd.DataFrame(data_interaction)
    checkin_df, review_df = reshape_review(interaction_df)
    save_csv(checkin_df, save_path, 'checkin.csv')
    save_csv(review_df, save_path, 'review.csv')
    del data_interaction, interaction_df, checkin_df, review_df
    gc.collect()

    data_photo = load_json(load_path, 'photos.json')
    photo_df = pd.DataFrame(data_photo)
    photo_ids_grouped = photo_df.groupby('business_id')['photo_id'].agg(
        list).reset_index()
    save_csv(photo_df, save_path, 'photo.csv')
    del data_photo, photo_df
    gc.collect()

    data_business = load_json(load_path, 'yelp_academic_dataset_business.json')
    business_df = pd.DataFrame(data_business)
    business_df = pd.merge(business_df, photo_ids_grouped, on='business_id',
                           how='left')
    save_csv(business_df, save_path, 'business.csv')
    del data_business, business_df, photo_ids_grouped
    gc.collect()

    data_user = load_json(load_path, 'yelp_academic_dataset_user.json')
    user_df = pd.DataFrame(data_user)
    user_df = reshape_user(user_df)
    save_csv(user_df, save_path, 'user.csv')
    del data_user, user_df
    gc.collect()

    data_tips = load_json(load_path, 'yelp_academic_dataset_tip.json')
    tip_df = pd.DataFrame(data_tips)
    tip_df['text'] = tip_df['text'].apply(clean_text)
    save_csv(tip_df, save_path, 'tip.csv')
    del data_tips, tip_df
    gc.collect()

    data_checkin_distribution = load_json(load_path, 'yelp_academic_dataset_checkin.json')
    checkin_distribution_df = pd.DataFrame(data_checkin_distribution)
    save_csv(checkin_distribution_df, save_path, 'checkin_distribution.csv')
    del data_checkin_distribution, checkin_distribution_df
    gc.collect()