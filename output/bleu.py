import pandas as pd

from eval.translate_metric import get_nltk33_sent_bleu as bleu

hyp_path= '/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/output/hypotheses.txt'
ref_path= '/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/output/references.txt'

hyp_df = pd.read_csv(hyp_path)
ref_df = pd.read_csv(ref_path)

hyp_list = []
ref_list = []

for text in hyp_df.values:
    hyp_list.append(text[0].split())

for text in ref_df.values:
    ref_list.append(text[0].split())

print(bleu.__name__, ':', bleu(hyp_list, ref_list))
