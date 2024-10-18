# Aromanian Translation Project

This repository contains our journey in attempting to translate a low-resource language, Aromanian. There are ongoing debates about whether Aromanian is a dialect of Romanian or a language of its own. Aromanian is a Romance language closely related to Romanian, originating from the Latin spoken south of the Danube during the Romanization of the Balkans. It features distinct dialects and preserves several archaic elements that reflect its unique historical development. Aromanian serves as both a spoken and cultural identifier for the Aromanian people, who number between 400,000 and 500,000, primarily inhabiting regions of the Balkan Peninsula, including Greece, Albania, North Macedonia, Bulgaria, Serbia, and Romania. [^1]

## Technical Overview

### Part-of-Speech Tagging Attempts
We began this project by attempting to label parts of speech in Aromanian using existing models for Romanian. The results turned out to be inefficient in practice due to the differences between the two languages.

### Dataset
We used a parallel dataset of fairy tales in both languages and an Aromanian-Romanian dictionary as sources of data. While the dataset consists of parallel sentences in both languages, a significant number of them are a combination of literal and free translation. The meaning is retained, even though the words or expressions might not be the best matches between the two languages. Additionally, there are grammatical differences, such as the positioning of parts of speech.

The dataset is structured so that one line in a file corresponds to the same line in the file with stories in the other language. A single row in the file can have multiple sentences.

### Dataset Processing
Since Aromanian is not an official language, there is no standardized form of writing. There are divergences in character representation between the fairy tale dataset and the dictionary. Some characters differ entirely, and there are also differences in diacritics. Different characters were replaced with their Romanian equivalents (as in some cases, Greek letters appeared in the script). However, the diacritics were removed, relying on experience from Romanian, where most of the time, the meaning of a sentence can be understood without them. An exception was made for the GRAG variant, where diacritics were retained. This was because LLMs often returned responses with diacritics, using them incorrectly if they did not appear in the provided prompt words.

### Dictionary Creation
We then created a dictionary of words to facilitate future work. The dictionary was built based on a CSV file. Interestingly, some Romanian words are represented by the same word in Aromanian, resulting in homonyms. In such cases, we retained all possible translations. 

### Similarity Measures
In addition to direct translations, we added methods to the dictionary for obtaining similar words based on Levenshtein distance or other metrics such as Euclidean and Cosine distances between the Romanian word and its Aromanian counterpart.

### Graph-Based Translation
Later, we constructed a graph-based database using this dictionary. The database contains three types of nodes: Romanian sentences, their Aromanian translations, and words that appear in the sentences, along with their translations and parts of speech.

When a new sentence is proposed for translation, it is split into words. The graph database searches for these words or, if they don't exist, for similar words. The corresponding sentence nodes are retrieved, along with other words in those sentences. These nodes and the information they contain are assembled into a prompt that can be well-understood by LLMs to translate new input.

While lexically the results were similar to the correct translation, they often differed fundamentally in meaning. (To do: Add examples from notebooks).

### Tested Models
We tested several models, including GPT-3.5 Turbo, GPT-4, GPT-4o, and GPT-4o-mini. GPT-4o yielded the best results, but it seems that the LLM had some minimal prior knowledge about Aromanian.

### NLLB Approach
Next, we used NLLB models, which offered the best results overall. The dataset was modified to be more granular, with sentences extracted from the parallel dataset. The model was trained in both directions (ro→rup and rup→ro). Sentences exceeding 1000 tokens were excluded from training. The model was trained for 20,000 steps on a dataset consisting of 27,000 training examples, 95% of which were dictionary words, while the remaining 5% were sentences. 

For testing, we used a dataset with a higher proportion of sentences (about 10%). 

The results for NLLB are as follows:

| Direction  | BLEU | chrF2 |
|------------|------|-------|
| rup → ro   | 5.62 | 21.91 |
| ro → rup   | 5.18 | 22.08 |

### Fine-Tuning LLMs
Several sources inspired this attempt to fine-tune LLMs on our dataset. One of them is the English-Kalamang translation project (insert reference). Unlike them, we did not have access to a structured textbook aimed at language learning. Additionally, there were limitations regarding processing power.

We primarily relied on Meta's LLaMA models (LLaMA2, LLaMA3, LLaMA3.1, and LLaMA3.2), as well as their Romanian variants, RoLLaMA. The fine-tuning was parameter-efficient, where we trained only the linear layers of the models. Using LoRA and PEFT, we trained quantized versions (4-bit) of the models.

The best results were obtained from OpenLLM-Ro/RoLLaMA3-8b-Instruct with a BLEU score of 1.61. All models were trained for just one epoch but with different learning rates. For this model, we used r=32 and lora_alpha=32. For LLaMA2, we used r=16 and lora_alpha=32. The BLEU score for LLaMA2 was around 1.4.

As the large parameter models trained with LoRA and PEFT did not perform well, we also tried smaller models. However, the dataset was not sufficient even for the smaller models to learn using parameter-efficient methods. For LLaMA3.2 1B, the BLEU score was 0.59 and chrF2 was 8.83. 

It was observed that even before the end of one epoch, the loss began fluctuating significantly and did not decrease meaningfully. We also tested smaller models like BART-LARGE, which achieved:

| Model        | BLEU | chrF2 |
|--------------|------|-------|
| BART-LARGE   | 1.28 | 19.90 |
| LLaMA3.2 1B  | 0.59 | 8.83  |

Despite some promising results, the models still have room for improvement, especially regarding maintaining sentence meaning.


## References

[^1]: Nicolae Saramandu, *Aromânii. Istorie, Literatură. Scrieri despre Dialectul Aromân*, [link to the article](https://lingv.ro/wp-content/uploads/2024/06/Art_10_NICOLAE-SARAMANDU-Aromanii_177-206.pdf).
