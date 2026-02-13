import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import warnings


# Tutte le feature audio confermate come utili
AUDIO_FEATURES = [
    'duration_ms', 'danceability', 'energy', 'key', 'loudness', 
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'time_signature'
]

def clean(INPUT_FILE = 'Dataset.csv', OUTPUT_FILE = 'Dataset_cleaned.csv' ):
    df = pd.read_csv(INPUT_FILE)
    initial_rows = len(df)
    
    duplicates = df[df.duplicated(subset=['spotify_id'], keep=False)]
    n_duplicates = len(duplicates)
    print(f"\nBrani duplicati trovati: {n_duplicates}")
    
    if n_duplicates > 0:
        df_before = len(df)
        df = df.drop_duplicates(subset=['spotify_id'], keep='first')
        removed = df_before - len(df)
        print(f"Rimossi {removed} duplicati (mantenuta prima occorrenza)")
    

    print("\nGestione anni sospetti (year < 1920)...")
    suspicious_years = df[df['year'] < 1920]
    n_suspicious = len(suspicious_years)
    print(f"\nBrani con anno < 1920: {n_suspicious}")
    
    if n_suspicious > 0:
        df_before = len(df)
        df = df[df['year'] >= 1920]
        removed = df_before - len(df)
        print(f"\nRimossi {removed} brani con anno < 1920")
    
    # Analisi singole feature
    zero_features = ['danceability', 'tempo', 'valence']
    for feature in zero_features:
        n_zeros = len(df[df[feature] == 0])
        print(f"\nBrani con {feature}=0: {n_zeros}")
    
    # Identifica brani con 2+ features a zero simultaneamente
    zero_count = df[zero_features].apply(lambda row: (row == 0).sum(), axis=1)
    multi_zero = df[zero_count >= 2]
    n_multi_zero = len(multi_zero)
    
    print(f"\nBrani con 2+ features a zero simultaneamente: {n_multi_zero}")
    
    if n_multi_zero > 0:
        df_before = len(df)
        df = df[zero_count < 2]
        removed = df_before - len(df)
        print(f"\nRimossi {removed} brani con 2+ features anomale")
    
    for feature in zero_features:
        n_zeros = len(df[df[feature] == 0])
        print(f"\n{feature}=0: {n_zeros} brani")
    
    tags_null = df['tags'].isna().sum()
    tags_empty = (df['tags'] == '').sum()
    tags_total_missing = tags_null + tags_empty
    percentage_tags = (tags_total_missing / len(df)) * 100
    
    print(f"\nTags nulli: {tags_null}")
    print(f"\nTags vuoti: {tags_empty}")
    print(f"\nTOTALE tags mancanti: {tags_total_missing} ({percentage_tags:.2f}%)")
    
    if tags_total_missing > 0:
        df_before = len(df)
        df = df[(df['tags'].notna()) & (df['tags'] != '')]
        removed = df_before - len(df)
        print(f"\nRimossi {removed} brani senza tags")

    genre_null = df['genre'].isna().sum()
    genre_empty = (df['genre'] == '').sum()
    genre_total_missing = genre_null + genre_empty
    percentage_genre = (genre_total_missing / len(df)) * 100
    
    print(f"\nGenre nulli: {genre_null}")
    print(f"\nGenre vuoti: {genre_empty}")
    print(f"\nTOTALE genre mancanti: {genre_total_missing} ({percentage_genre:.2f}%)")
    
    # Brani con tags ma senza genre (candidati per imputazione)
    has_tags_no_genre = df[(df['tags'].notna()) & (df['tags'] != '') & 
                            ((df['genre'].isna()) | (df['genre'] == ''))]
    print(f"\nBrani con tags MA senza genre: {len(has_tags_no_genre)} (candidati per imputazione)")

    columns_to_drop = ['spotify_preview_url', 'spotify_id']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"\nRimosse colonne: {', '.join(existing_columns)}")

    final_rows = len(df)
    total_removed = initial_rows - final_rows
    percentage_removed = (total_removed / initial_rows) * 100
    
    print(f"\nRighe iniziali:  {initial_rows:,}")
    print(f"\nRighe finali:    {final_rows:,}")
    print(f"\nRighe rimosse:   {total_removed:,} ({percentage_removed:.2f}%)")
    print(f"\nColonne finali:  {len(df.columns)}")
    
    # Verifica valori nulli
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\nValori nulli rilevati:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"\n{col}: {count}")
    else:
        print(f"\nNessun valore nullo presente")
    
    print("\nSalvataggio dataset pulito...")
    df.to_csv(OUTPUT_FILE, index=False)

def imputation (INPUT_FILE = 'Dataset_cleaned.csv', OUTPUT_FILE = 'Dataset_final_imputed_dataset.csv'):
    df = pd.read_csv(INPUT_FILE)
    
    df_known = df[(df['genre'].notna()) & (df['genre'] != '')].copy()
    df_unknown = df[(df['genre'].isna()) | (df['genre'] == '')].copy()
    
    print(f"\nBrani con genere noto: {len(df_known):,}")
    print(f"\nBrani da imputare: {len(df_unknown):,}")

    X_known_text = df_known['tags'].fillna('')
    X_known_num = df_known[AUDIO_FEATURES].values
    y_known = df_known['genre'].values
    
    X_unknown_text = df_unknown['tags'].fillna('')
    X_unknown_num = df_unknown[AUDIO_FEATURES].values

    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    
    X_known_text_vec = tfidf.fit_transform(X_known_text).toarray()
    X_unknown_text_vec = tfidf.transform(X_unknown_text).toarray()

    scaler = StandardScaler()
    X_known_num_scaled = scaler.fit_transform(X_known_num)
    X_unknown_num_scaled = scaler.transform(X_unknown_num)

    X_known_comb = np.hstack((X_known_num_scaled, X_known_text_vec))
    X_unknown_comb = np.hstack((X_unknown_num_scaled, X_unknown_text_vec))
    print(f"    Feature totali pronte per l'addestramento: {X_known_comb.shape[1]}")

    smote = SMOTE(random_state=42)
    X_known_bal, y_known_bal = smote.fit_resample(X_known_comb, y_known)
    print(f"\nCampioni dopo il bilanciamento: {len(X_known_bal):,}")

    print("\nAddestramento HistGradientBoostingClassifier in corso...")
    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(X_known_bal, y_known_bal)
    print("\nAddestramento HistGradientBoostingClassifier completato")

    predicted_genres = hgb.predict(X_unknown_comb)

    df_unknown['genre_imputed'] = predicted_genres
    df_known['genre_imputed'] = df_known['genre']

    df_final = pd.concat([df_known, df_unknown], ignore_index=True)

    df_final['genre_is_imputed'] = False
    df_final.loc[df_final['genre'].isna() | (df_final['genre'] == ''), 'genre_is_imputed'] = True

    df_final['genre'] = df_final['genre_imputed']
    df_final = df_final.drop(columns=['genre_imputed'])

    df_final.to_csv(OUTPUT_FILE, index=False)

clean()
imputation()
