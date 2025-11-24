# create_evaluation.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("ai_tags.csv")

# Normalization function
def normalize_tag(t):
    return t.strip().lower().replace(" ", "_").replace("-", "_")

def tag_list(s):
    if pd.isna(s) or s=="":
        return []
    return [normalize_tag(x) for x in str(s).split(",") if x.strip()]

# Build list columns
df['tags_manual_list'] = df['tags_manual'].apply(tag_list)
df['tags_ai_list'] = df['tags_ai'].apply(tag_list)

# Scoring function
def score_row(ai_list, manu_list):
    ai = set(ai_list)
    manu = set(manu_list)
    tp = len(ai & manu)
    fp = len(ai - manu)
    fn = len(manu - ai)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1

scores = df.apply(lambda r: score_row(r['tags_ai_list'], r['tags_manual_list']), axis=1)
scores_df = pd.DataFrame(scores.tolist(), columns=['tp','fp','fn','precision','recall','f1'])
df = pd.concat([df, scores_df], axis=1)

# Save detailed results
df.to_csv("events_with_scores.csv", index=False)
print("Saved events_with_scores.csv — contains per-row TP/FP/FN and metrics")

# Aggregate by sport
agg = df.groupby("sport")[['tp','fp','fn','precision','recall','f1']].mean().reset_index()
print("\nMean metrics by sport:")
print(agg.to_string(index=False))

# Save metrics CSV
agg.to_csv("metrics_by_sport.csv", index=False)

# Plot: Precision/Recall/F1 by sport
plt.figure(figsize=(8,4))
agg_plot = agg.set_index('sport')[['precision','recall','f1']]
agg_plot.plot(kind='bar', ylim=(0,1), rot=0)
plt.title("Precision / Recall / F1 by Sport (mean per-row)")
plt.ylabel("score")
plt.tight_layout()
plt.savefig("metrics_by_sport.png")
plt.close()
print("Saved plot: metrics_by_sport.png")

# Plot: TP/FP/FN overall distribution
overall = df[['tp','fp','fn']].sum()
plt.figure(figsize=(5,4))
overall.plot(kind='pie', autopct='%1.0f%%', ylabel='', title='TP / FP / FN distribution (overall)')
plt.tight_layout()
plt.savefig("tp_fp_fn_pie.png")
plt.close()
print("Saved plot: tp_fp_fn_pie.png")

# Print top error examples
print("\nTop rows with highest FP (possible overprediction):")
print(df.sort_values('fp', ascending=False)[['id','sport','event_text','tags_ai','tags_manual','fp']].head(6).to_string(index=False))

print("\nTop rows with highest FN (possible misses):")
print(df.sort_values('fn', ascending=False)[['id','sport','event_text','tags_ai','tags_manual','fn']].head(6).to_string(index=False))

print("\nAll done — files saved:")
print(" - events_with_scores.csv")
print(" - metrics_by_sport.csv")
print(" - metrics_by_sport.png")
print(" - tp_fp_fn_pie.png")
