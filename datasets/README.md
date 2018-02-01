Complex Word Identification (CWI) Shared Task 2018
==================================================

About
-----

See the [shared task website](http://sites.google.com/view/cwisharedtask2018/) for more info.

### Contact

Sanja Štajner - sanja(at)informatik(dot)uni-mannheim(dot)de

### Organizers

- Sanja Štajner (University of Mannheim)
- Chris Biemann (University of Hamburg)
- Shervin Malmasi (Harvard Medical School)
- Gustavo Paetzold (University of Sheffield)
- Lucia Specia (University of Sheffield)
- Anaïs Tack (Université Catholique de Louvain and KU Leuven)
- Seid Muhie Yimam (University of Hamburg)
- Marcos Zampieri (University of Wolverhampton)

### Citation

When using the data in your research or publication, please cite the task paper which can be found in the proceedings of BEA13 (The 13th Workshop on Innovative Use of NLP for Building Educational Applications, NAACL 2018 Workshops).

### References

> Glavaš G. and Štajner S. 2015. Simplifying Lexical Simplification: Do We Need Simplified Corpora? In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (ACL-IJCNLP), pp. 63-68.
> Paetzold, G. and Specia, L. 2016a. Unsupervised lexical simplification for non-native speakers Proceedings of the 30th AAAI.
> Paetzold, G. and Specia, L. 2016b. PLUMBErr: An Automatic Error Identification Framework for Lexical Simplification. Proceedings of the first international workshop on Quality Assessment for Text Simplification (QATS), pp. 1-9.
> Paetzold, G. and Specia, L. 2016c. SemEval 2016 Task 11: Complex Word Identification Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pp. 560-569.
> Shardlow, M. 2013. A Comparison of Techniques to Automatically Identify Complex Words. Proceedings of the Student Research Workshop at the 51st Annual Meeting of the Association for Computational Linguistics (ACL), pp. 103-109.


Datasets
--------

### Description

The English dataset consists of mixture of professionally written news, non-professionally written news (WikiNews), and Wikipedia articles. Spanish and German datasets contain data taken from Spanish and German Wikipedia pages. The French dataset contains data taken from French Wikipedia pages.

Each sentence in the English dataset was annotated by 10 native and 10 non-native speakers. Annotators were provided with the surrounding context of each sentence, i.e. a paragraph, then asked to mark words they think would be difficult to understand for children, non-native speakers, and people with language disabilities.

Each sentence in the German, Spanish and French dataset was annotated by 10 people (a mixture of native and non-native speakers).


### Data Format

#### Training Data

The training data will be provided in the following format:

    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	31	51	flexed their muscles	10	10	3	2	1	0.25
    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	31	37	flexed	10	10	2	6	1	0.4
    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	44	51	muscles	10	10	0	0	0	0.0

Each line represents a sentence with one complex word annotation and relevant information, each separated by a TAB character.
- The first column shows the HIT ID of the sentence. All sentences with the same ID belong to the same HIT.
- The second column shows the actual sentence where there exists a complex phrase annotation.
- The third and fourth columns display the start and end offsets of the target word in this sentence.
- The fifth column represents the target word.
- The sixth and seventh columns show the number of native annotators and the number of non-native annotators who saw the sentence.
- The eighth and ninth columns show the number of native annotators and the number of non-native annotators who marked the target word as difficult.
- The tenth and eleventh columns show the gold-standard label for the binary and probabilistic classification tasks.

The labels in the binary classification task were assigned in the following manner:
- 0: simple word (none of the annotators marked the word as difficult)
- 1: complex word (at least one annotator marked the word as difficult)

The labels in the probabilistic classification task were assigned as `<the number of annotators who marked the word as difficult>`/`<the total number of annotators>`.

#### Test Data

The test data will be in the following format:

    3P7RGTLO6EE07HLUVDKKHS6O7CCKA5	Both China and the Philippines flexed their muscles on Wednesday.	31	37	flexed	10	10

In the test input format, only the first seven columns of the train format are given.

In the test prediction format, the order of the input file should be kept and each line of the file must be a either a binary label or a probability, depending on whether the team participates in the binary/probabilistic classification task.

In the binary classification task, the participating systems will have a task of predicting the right label (0 or 1) for the test data.

In the probabilistic classification task, the participating systems will have a task of giving a probability of the target word being complex.


### Data Instances

The number of instances for each training, development and test set is:

- English: 27,299 training; 3,328 dev; 4,252 test
- Spanish: 13,750 training; 1,622 dev; 2,233 test
- German: 6,151 training; 795 dev; 959 test
- French: 2,251 test
