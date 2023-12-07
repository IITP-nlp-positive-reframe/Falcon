# Falcon
Falcon for positive reframing project

* Download falcon model folder from [google drive](https://drive.google.com/drive/folders/1_x0kKUmnxPKLda8lnJWbTufb99jvT9rC?usp=sharing)  and unzip under falcon-rw-1b/output folder
* Run falcon_eval.py to get reframed output and performance score
* Reframed result is provided in reframe_falcon_rw_1b_predict.txt file

* Score measured with the generated sentence

```
rouge1 24.581379629703704
rouge2 6.593148985690581
rougeL 20.218399523152513
rougeLsum 20.246356296437394
load_metric('sacrebleu') 3.442923813094571
load_metric('bertscore') 86.3173001612018
Text Blob Avg Sentiment Change:  0.3020066002762485
```
