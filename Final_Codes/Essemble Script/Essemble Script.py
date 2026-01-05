import pandas as pd
import os

# ================= CONFIGURATION =================
# 1. Path to U-Net
FILE_UNET = 'submission_unet.csv'

# 2. Path to ROADMAP
FILE_ROAD = 'submission_roadmap.csv'

# 3. Path to STAE
FILE_STAE = 'submission_STAE.csv' 

# 4. Path to NEW MODEL (Your 4th File)
FILE_NEW  = 'submission_JR.csv' # <--- UPDATE THIS PATH

OUTPUT_FILE = 'submission_quad_ensemble.csv'

# WEIGHTS (Must sum to 1.0)
# Strategy: Strongest model gets the most weight. 
# Example: If STAE is best, give it 0.4. If the new one is weak, give it 0.1.
W_ROAD = 0.10
W_UNET = 0.10
W_STAE = 0.50
W_NEW  = 0.30
# =================================================

def quad_ensemble():
    print("🚀 Initiating The Quad Ensemble (ROADMAP + U-Net + STAE + NEW)...")
    
    try:
        # 1. Load Dataframes
        df_u = pd.read_csv(FILE_UNET)
        df_r = pd.read_csv(FILE_ROAD)
        
        # Check optional files
        df_s = pd.read_csv(FILE_STAE) if os.path.exists(FILE_STAE) else None
        df_n = pd.read_csv(FILE_NEW)  if os.path.exists(FILE_NEW) else None
        
        print("✅ Base files loaded.")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # 2. Rename Columns to avoid collision
    df_u = df_u.rename(columns={'Predicted': 'score_unet'})
    df_r = df_r.rename(columns={'Predicted': 'score_road'})
    
    # 3. Start Merging
    df_merge = pd.merge(df_u, df_r, on='ID', how='inner')
    
    # Merge STAE if exists
    if df_s is not None:
        df_s = df_s.rename(columns={'Predicted': 'score_stae'})
        df_merge = pd.merge(df_merge, df_s, on='ID', how='inner')
    else:
        # Handle missing file by zeroing its weight later or raising error
        print("⚠️ STAE file missing, skipping...")

    # Merge NEW Model if exists
    if df_n is not None:
        df_n = df_n.rename(columns={'Predicted': 'score_new'})
        df_merge = pd.merge(df_merge, df_n, on='ID', how='inner')
    else:
        print("⚠️ New Model file missing, skipping...")

    # 4. Calculate Weighted Average
    # We dynamically handle missing files by checking columns
    
    final_score = (df_merge['score_road'] * W_ROAD) + (df_merge['score_unet'] * W_UNET)
    
    if 'score_stae' in df_merge.columns:
        final_score += (df_merge['score_stae'] * W_STAE)
    
    if 'score_new' in df_merge.columns:
        final_score += (df_merge['score_new'] * W_NEW)

    # Normalize Result (Optional but recommended if weights didn't sum perfectly due to missing files)
    # This ensures the final score is scaled nicely
    
    df_merge['ensemble_score'] = final_score

    # 5. Save Final
    final_df = df_merge[['ID', 'ensemble_score']].rename(columns={'ensemble_score': 'Predicted'})
    
    # Ensure no NaN
    final_df['Predicted'] = final_df['Predicted'].fillna(0.0)
    from scipy.ndimage import gaussian_filter1d

    # Apply this to your final 'Predicted' column
    # sigma=2 is usually the "magic number" for 30fps videos (Avenue)
    final_df['Predicted'] = gaussian_filter1d(final_df['Predicted'], sigma=15)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅✅✅ QUAD ENSEMBLE SAVED: {OUTPUT_FILE}")
    print(final_df.head())

if __name__ == "__main__":
    quad_ensemble()