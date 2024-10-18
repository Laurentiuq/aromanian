import pandas as pd
import unicodedata
from collections import defaultdict
from tqdm import tqdm
import spacy
from rapidfuzz import process, fuzz, utils
import re
nlp = spacy.load('ro_core_news_sm')



class Dictionary:
    def __init__(self, file, column_names, unused_columns=[]):
    # A dictionary that has as keys the Aromanian words from the transformed dataframe and as values a list with a tuple containing the original aromanian
    # word(use the df with the index) and its translation and part of speech
        self.file = file
        self.column_names = column_names
        self.unused_columns = unused_columns
        self.dict = defaultdict(list)
    # Create an iterator
    def __iter__(self):
        for key in self.dict.keys():
            yield key

    def process_file(self, file="", df=pd.DataFrame()):
        """
        Process the file and return the dataframe and the transformed dataframe. Transformations imply removing diacritics and eliminating -mi.
        If the dataframe is empty, it will read the file from a given path(must be an excel file)
        Parameters
        ----------
        file : str
            An excel file containing the words
        df : pd.DataFrame
            A dataframe containing the words
        Returns
        -------
        pd.DataFrame
            The original dataframe
        pd.DataFrame
            The transformed dataframe
        """
        if df.empty:        
            df = pd.read_excel(file, header=None, names=self.column_names)
        df.drop(columns=self.unused_columns, inplace=True)
        # Eliminate diacritics from the words
        df_transformed = df.applymap(lambda x: ''.join([c for c in unicodedata.normalize('NFKD', x)  if unicodedata.category(c) != 'Mn']) if type(x) == str else x)
        # There are some words that have a -mi at the end, we will eliminate them also
        df_transformed.replace(r'\s*-\s*mi\b', '', regex=True, inplace=True)
        df_transformed.replace(r'\(i\)', 'i', regex=True, inplace=True)
        df_transformed.to_csv(file[:-4] + '.csv', index=False)
        df.to_csv(file[:-4] + '_original.csv', index=False)
        return df, df_transformed
    
    
    def create_dictionary(self, original_csv_file="", transformed_csv_file="", source_lang="aromanian", target_lang="romanian", extra_info="pos"):
        """
        Create a dictionary from the given file. If the transformed file is not given,
        it will process the original file and create the transformed file
        Parameters
        ----------
        original_csv_file : str
            The path to the original csv file
        transformed_csv_file : str -- optional
            The path to the transformed csv file
        source_lang : str
            The column name of the source language
        target_lang : str
            The column name of the target language
        extra_info : str
            The column name of the extra information, like part of speech
        """
        if original_csv_file != "" and transformed_csv_file == "":
            df, df_transformed = self.process_file(original_csv_file)
        else:
            df_transformed = pd.read_csv(transformed_csv_file)
            df = pd.read_csv(original_csv_file)
        for index, row in df_transformed.iterrows():
            self.dict[row[source_lang]].append((df.iloc[index][source_lang], row[target_lang], row[extra_info]))

    def find_word(self, s):
        """
        Find the word in the dictionary
        Parameters
        ----------
        word : str
            The word to be found
        """
        # Apply the same transformations to the word as we did to the words in the dictionary
        s = ''.join([c for c in unicodedata.normalize('NFKD', s) if unicodedata.category(c) != 'Mn'])
        s = re.sub(r'\s*-\s*mi\b', '', s)
        s = re.sub(r'\(i\)', 'i', s)
        s = s.lower()
        return self.dict[s] if s in self.dict else None
    
    def find_similar_word(self, s, similarity_threshold=100, verbose=False, return_similarity_score=False):
        """
        Find a similar word in the dictionary
        Parameters
        ----------
        word : str
            The word to be found
        similarity_threshold : int
            The similarity threshold
        verbose : bool
            If True, it will print the word and the similar word
        """
        s = ''.join([c for c in unicodedata.normalize('NFKD', s) if unicodedata.category(c) != 'Mn'])
        s = re.sub(r'\s*-\s*mi\b', '', s)
        s = re.sub(r'\(i\)', 'i', s)
        s = s.lower()
    
        if len(s) >= 5:
            word = process.extractOne(s, self.dict.keys(), scorer=fuzz.ratio, score_cutoff=similarity_threshold)
            if verbose and word:
                print(s, word)
                # Show the similarity between the word and the similar word
                print(fuzz.ratio(s, word[0]) if word else None)
        else:
            if verbose:
                print(s, self.dict[s] if s in self.dict else None)
            if return_similarity_score:
                return self.dict[s], 100 if s in self.dict else None
            return self.dict[s] if s in self.dict else None
        if return_similarity_score:
            if word:
                return self.dict[word[0]], fuzz.ratio(s, word[0]) if word else None
            else:
                return None, None
        return self.dict[word[0]] if word else None
    
    def translate(self, sentence, similarity_threshold=100, split_translation=False):
        """
        Translate a sentence using the dictionary and a similarity threshold
        Parameters
        ----------
        sentence : str
            The sentence to be translated
        similarity_threshold : int
            The similarity threshold
        split_translation : bool
            If True, it will split the translation of a word if it has multiple translations and return only the first one
        """

        # Remove punctuation
        sentence = nlp(sentence)
        sentence = ' '.join([token.text for token in sentence if not token.is_punct])
        
        words = sentence.split()
        translated_sentence = ""
        for word in words:
            similar_word = self.find_similar_word(word, similarity_threshold)
            if similar_word:
                if split_translation:
                    splited_translation = re.split(r',\s*', similar_word[0][1])
                    translated_sentence += splited_translation[0] + " "
                else:
                    translated_sentence += similar_word[0][1] + " "
            else:
                translated_sentence += "|__| "
        return translated_sentence




