from nlgeval import compute_metrics


metrics_dict = compute_metrics(hypothesis="/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/output/hypotheses.txt",
                               references=["/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/output/references.txt"], no_skipthoughts=True,
                               no_glove=True)

