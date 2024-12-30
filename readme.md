# Aromanian Translation Project

This repository contains our journey in attempting to translate a low-resource language, Aromanian. There are ongoing debates about whether Aromanian is a dialect of Romanian or a language of its own. Aromanian is a Romance language closely related to Romanian, originating from the Latin spoken south of the Danube during the Romanization of the Balkans. It features distinct dialects and preserves several archaic elements that reflect its unique historical development. Aromanian serves as both a spoken and cultural identifier for the Aromanian people, who number between 400,000 and 500,000, primarily inhabiting regions of the Balkan Peninsula, including Greece, Albania, North Macedonia, Bulgaria, Serbia, and Romania. [^1]

## Technical Overview

### Part-of-Speech Tagging Attempts
We began this project by attempting to label parts of speech in Aromanian using existing models for Romanian. The results turned out to be inefficient in practice due to the differences between the two languages.

In our effort to create a Part-of-Speech (PoS) tagging system for Aromanian using Romanian models, we encountered significant challenges due to the linguistic differences between the two languages. Although closely related, Aromanian and Romanian have distinct vocabulary, syntax, and false friends—similar-looking words with different meanings—making a direct transfer of PoS tags inefficient.

We began by aligning Aromanian and Romanian texts using techniques like Levenshtein distance and word embeddings. However, these models struggled to accurately map words and tags due to the variations in sentence structure and expression between the two languages. Statistical models like Giza++ and eflomal improved alignment but still fell short in precision.

To address this, we used resources like the Aromanian dictionary by Tache Papahagi to refine PoS tagging where automatic alignment failed. Despite the limited data, we trained a Hidden Markov Model (HMM) to generalize PoS tagging for new Aromanian text. While the model is far from perfect, it provided a foundation for further development and demonstrated that adapting Romanian PoS models for Aromanian requires significant linguistic adjustments.


### Dataset
We used a parallel dataset of fairy tales in both languages and an Aromanian-Romanian dictionary as sources of data. While the dataset consists of parallel sentences in both languages, a significant number of them are a combination of literal and free translation. The meaning is retained, even though the words or expressions might not be the best matches between the two languages. Additionally, there are grammatical differences, such as the positioning of parts of speech.

The dataset is structured so that one line in a file corresponds to the same line in the file with stories in the other language. A single row in the file can have multiple sentences.

### Dataset Processing
Since Aromanian is not an official language, there is no standardized form of writing. There are divergences in character representation between the fairy tale dataset and the dictionary. Some characters differ entirely, and there are also differences in diacritics. Different characters were replaced with their Romanian equivalents (as in some cases, Greek letters appeared in the script). However, the diacritics were removed, relying on experience from Romanian, where most of the time, the meaning of a sentence can be understood without them. An exception was made for the GRAG variant, where diacritics were retained. This was because LLMs often returned responses with diacritics, using them incorrectly if they did not appear in the provided prompt words.

### Dictionary Creation
We then created a dictionary of words to facilitate future work. The dictionary was built based on a CSV file. Interestingly, some Romanian words are represented by the same word in Aromanian, resulting in homonyms. In such cases, we retained all possible translations. 

### Similarity Measures
In addition to direct translations, we added methods to the dictionary for obtaining similar words based on Levenshtein distance or other metrics such as Euclidean and Cosine distances between the Romanian word and its Aromanian counterpart.

When querying the dictionary, there are multiple translation methods available. If a direct translation exists in the Cunia dictionary, it will be provided. Alternatively, the dictionary can offer a translation based on a predefined similarity threshold. This threshold ensures that the returned word has a high degree of similarity to the queried word, minimizing the chance of returning dissimilar words. Users can also customize the similarity threshold to better suit their specific needs, allowing for more flexible or stricter translation results depending on the context.

### Graph-Based RAG Translation [^6]
Retrieval-Augmented Generation (RAG) is a technique that enhances language models by combining them with a retrieval mechanism. This approach uses external data sources, such as databases or graphs, to generate more accurate responses by fetching relevant information based on a query. For our project, we applied a graph-based database to assist with the translation of Aromanian to Romanian.

We constructed a graph with three types of nodes: Romanian sentences, Aromanian translations, and words appearing in both languages. The relationship between Romanian and Aromanian sentences is represented by a "translates_to" link. Each Aromanian word in a sentence has a relationship of "appear_in" that connects it to the corresponding sentence node.

Each Aromanian word node is enriched with several attributes: its part of speech, its translation into Romanian (from a dictionary), and a similarity score if the translation is not an exact match. The similarity score helps measure how closely the Aromanian word corresponds to a Romanian word, accounting for potential variations that are not directly found in the dictionary. This graph-based approach allows for more flexible and accurate translations by considering sentence structure, word alignment, and linguistic similarities across both languages.

When a new sentence is proposed for translation, it is split into words. The graph database searches for these words or, if they don't exist, for similar words. The corresponding sentence nodes are retrieved, along with other words in those sentences. These nodes and the information they contain are assembled into a prompt that can be well-understood by LLMs to translate new input.

While lexically the results were similar to the correct translation, they often differed fundamentally in meaning. (To do: Add examples from notebooks).

![RAG_image](https://github.com/user-attachments/assets/7d813110-a683-40a6-a46d-3cfe1bed86cf)


### Tested Models
We tested several models, including GPT-3.5 Turbo, GPT-4, GPT-4o, and GPT-4o-mini. GPT-4o yielded the best results, but it seems that the LLM had some minimal prior knowledge about Aromanian.

### NLLB Approach [^3]
Next, we used NLLB models, which offered the best results overall. The dataset was modified to be more granular, with sentences extracted from the parallel dataset. The model was trained in both directions (ro→rup and rup→ro) [^4]. Sentences exceeding 1000 tokens were excluded from training. The model was trained for 20,000 steps on a dataset consisting of 27,000 training examples, 92% of which were dictionary words, while the remaining 8% were sentences. 

For testing, we used a dataset with a higher proportion of sentences (about 10%). 

The results for NLLB are as follows:

| Direction  | BLEU | chrF2 |
|------------|------|-------|
| rup → ro   | 5.62 | 21.91 |
| ro → rup   | 5.18 | 22.08 |

### Fine-Tuning LLMs
Several sources inspired this attempt to fine-tune LLMs on our dataset. One of them is the English-Kalamang translation project [^5]. Unlike them, we did not have access to a structured textbook aimed at language learning. Additionally, there were limitations regarding processing power.

We primarily relied on Meta's LLaMA models (LLaMA2, LLaMA3, LLaMA3.1, and LLaMA3.2), as well as their Romanian variants, RoLLaMA. The fine-tuning was parameter-efficient, where we trained only the linear layers of the models. Using LoRA and PEFT, we trained quantized versions (4-bit) of the models.

The best results were obtained from OpenLLM-Ro/RoLLaMA3-8b-Instruct [^2] with a BLEU score of 1.61. All models were trained for just one epoch but with different learning rates. For this model, we used r=32 and lora_alpha=32. For LLaMA2, we used r=16 and lora_alpha=32. The BLEU score for LLaMA2 was around 1.4.

As the large parameter models trained with LoRA and PEFT did not perform well, we also tried smaller models. However, the dataset was not sufficient even for the smaller models to learn using parameter-efficient methods. For LLaMA3.2 1B, the BLEU score was 0.59 and chrF2 was 8.83. 

It was observed that even before the end of one epoch, the loss began fluctuating significantly and did not decrease meaningfully. We also tested smaller models like BART-LARGE, which achieved:

| Model        | BLEU | chrF2 |
|--------------|------|-------|
| BART-LARGE   | 1.28 | 19.90 |
| LLaMA3.2 1B  | 0.59 | 8.83  |

Despite some promising results, the models still have room for improvement, especially regarding maintaining sentence meaning.


## References

[^1]: Nicolae Saramandu, *Aromânii. Istorie, Literatură. Scrieri despre Dialectul Aromân*, [link to the article](https://lingv.ro/wp-content/uploads/2024/06/Art_10_NICOLAE-SARAMANDU-Aromanii_177-206.pdf).

[^2]: Mihai Masala, Denis C. Ilie-Ablachim, Alexandru Dima, Dragos Corlatescu, Miruna Zavelca, Ovio Olaru, Simina Terian-Dan, Andrei Terian-Dan, Marius Leordeanu, Horia Velicu, Marius Popescu, Mihai Dascalu, and Traian Rebedea. *"Vorbești Românește? A Recipe to Train Powerful Romanian LLMs with English Instructions"*. 2024. [arXiv:2406.18266](https://arxiv.org/abs/2406.18266).

[^3]: Koishekenov, Yeskendir, Alexandre Berard, and Vassilina Nikoulina. *Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model*. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, edited by Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, 3567-3585. Toronto, Canada: Association for Computational Linguistics, 2023. [Link to paper](https://aclanthology.org/2023.acl-long.198), doi: [10.18653/v1/2023.acl-long.198](https://doi.org/10.18653/v1/2023.acl-long.198).

[^4]: David Dale. *How to Fine-Tune a NLLB-200 Model for Translating a New Language*. Medium, July 6, 2023. [Link to article](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865).

[^5]: Tanzer, Garrett, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. *A Benchmark for Learning to Translate a New Language from One Grammar Book*. 2024. [arXiv:2309.16575](https://arxiv.org/abs/2309.16575).

[^6]: NVIDIA. What Is Retrieval-Augmented Generation (RAG). NVIDIA Blog, October 20, 2023. [Link to article](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/). 

