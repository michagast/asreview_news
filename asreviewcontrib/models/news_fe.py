from asreview.models.feature_extraction.base import BaseFeatureExtraction
import numpy as np

import pandas as pd                 #For data science purposes
import re                           #For performing regex
import torch                        #For running models with cude
import nltk.data                    #For various things
from nltk.tokenize import word_tokenize
from collections import Counter
from tensorflow.keras.preprocessing.text import hashing_trick
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from num2words import num2words
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline

class News(BaseFeatureExtraction):
    """Custom feature extraction

    Feature extraction that generates features based on sentiment values and named entity recogntion among other things.
    """

    name = "news"
    label = "Dutch news feature extractor"

    def __init__(self, *args, **kwargs):
        self._model, self._tokenizernlp = None,None
        #Load in all required thing from packages.
        #self._modelner = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english", return_dict=True)
        #self._modelner.eval() # make sure model is not in training mode
        #self._tokenizerner = AutoTokenizer.from_pretrained('xlm-roberta-large-finetuned-conll03-english')
        #self.vectorizer = CountVectorizer()
        spacy_model = "en_core_web_lg"
        #For use in the split into sentences function



        #nltk.download('punkt')
        #nltk.download('stopwords')
        #nltk.download('averaged_perceptron_tagger')


        super(News, self).__init__(*args, **kwargs)
    #Todo refactor this so that no for loop is used
    #Todo remove hardcoded array reduction
    def transform(self, texts):
        # Create numpy array of features since asreview model only works with np arrays
        resultsentiment = np.empty([0])
        resulttextlen = np.empty([0])
        resultspecificwords = np.empty([0])
        resultner = np.array([])
        resultstddevsentence = np.empty([0])
        resultstddevwords = np.empty([0])
        resultreadability = np.empty([0])
        resulttypetoken = np.empty([0])
        resultpropernouns = np.empty([0])
        resultpassivevoice = np.empty([0])
        resultactivevoice = np.empty([0])

        counter1 = 0
        for text in texts:
            texts[counter1] = ' '.join(text.split()[1:len(text.split())])
            counter1 += 1


        counter = 0 #for keeping track of progress
        for text in texts:
            #text = ' '.join(text.split()[1:len(text.split())])

            counter = counter+1
            #resultsentiment = np.append(resultsentiment, self.generatesentimentvalues(text))
            #resulttextlen = np.append(resulttextlen, self.gettextlength(text))
            #resultspecificwords = np.append(resultspecificwords, self.specific_words_check(text))
            #resultner = np.append(resultner, self.generate_named_entities(text), axis = 0)
            #resultstddevsentence = np.append(resultstddevsentence, self.standard_dev_sentence_length(text))
            resultstddevwords = np.append(resultstddevwords, self.standard_dev_word_length(text))
            resultreadability = np.append(resultreadability, self.readability_index(text))
            resulttypetoken = np.append(resulttypetoken, self.type_token_ratio(text))
            #resultpropernouns = np.append(resultpropernouns, self.proper_nouns(text))
            #resultpassivevoice = np.append(resultpassivevoice, self.percentage_passive_voice(text))
            #resultactivevoice = np.append(resultactivevoice, self.percentage_active_voice(text))
            print('Currently at instance:', counter, '/', len(texts))

        # load in bag of words data
        resultbow = pd.read_csv(r'C:\Users\MichaG\Documents\Scriptie\Data-main\bowdf1001_news_adjusted_train.csv', index_col=[0])
        resultbow = resultbow.to_numpy()


        # Turn arrays into 2d Arrays
        resultner = resultner.reshape(int(len(resultner)/4),4)
        resultsentiment = resultsentiment.reshape(-1, 1)
        resulttextlen = resulttextlen.reshape(-1,1)
        resultspecificwords = resultspecificwords.reshape(-1,1)
        resultstddevsentence = resultstddevsentence.reshape(-1,1)
        resultstddevwords = resultstddevwords.reshape(-1,1)
        resultreadability = resultreadability.reshape(-1,1)
        resulttypetoken = resulttypetoken.reshape(-1,1)
        #resultpropernouns = resultpropernouns.reshape(-1,1)
        resultpassivevoice = resultpassivevoice.reshape(-1,1)
        resultactivevoice = resultactivevoice.reshape(-1,1)


        #Concatenate all arrays into one final array
        #result = np.hstack((resultsentiment, resulttextlen, resultspecificwords, resultstddevsentence, resultstddevwords[0:1596], resultreadability, resultpassivevoice, resultactivevoice, resulttypetoken, result_bow, resultner))
        result = np.hstack((resultreadability, resultstddevwords, resulttypetoken, resultbow))
        print(result.shape)
        return result

    def generatesentimentvalues(self, text):
        if not self._model:
            self._model = AutoModelForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base")
            self._model.eval()  # make sure model is not in training mode
        if not self._tokenizernlp:
            self._tokenizernlp = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

        ''' Function that generates the sentiment value for the specific text
        '''
        sentiment_analysis = pipeline("sentiment-analysis", model=self._model, tokenizer=self._tokenizernlp,
                                      max_length=512,
                                      truncation=True, device=0)
        sentiment_result = sentiment_analysis(text)
        if sentiment_result[0]['label'] == 'NEGATIVE':
            result = 0 - sentiment_result[0]['score']
        else:
            result = sentiment_result[0]['score']
        return result

    def gettextlength(self, text):
        return len(text)

    def specific_words_check(self, text):
        ''' Function that searches for specific words and sums the total occurences
        By using regex, this function looks for the words Office, Policy, CAISO, Sales and Ligitiation and counts the amount of times it finds these words and adds it all up
        '''
        amount_of_words = len(re.findall(
            r'(\b[Mm]+eet?(ing)?s?\b|\b[Pp]+lane\b|\bexpense report\b|\b[Cc]+all\b|\b[Vv]+oicemail\b|\b[Ee]+?[Mm]+ail(ing)?\b|\b[Ww]+eeks\b|\b([Ss]+chedul(e)?(ing)?)|\b[Tt]+ime|\b[Ww]+eek\b|\b[Ii]+nvite?d?(ing)?\b|\b([0-1]?[0-9]):[0-5][0-9]\b)', text))
        if amount_of_words:
            return (amount_of_words)
        else:
            return 0

    def split_into_sentences(self, text):
        '''Function that splits a specific text into sentences. Used for generating named entities
        '''
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(self.prefixes, "\\1<prd>", text)
        text = re.sub(self.websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + self.alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(self.acronyms + " " + self.starters, "\\1<stop> \\2", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + self.suffixes + "[.] " + self.starters, " \\1<stop> \\2", text)
        text = re.sub(" " + self.suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + self.alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def remove_short_tokens(self, tokens):
        '''Functions that removes tokens that are shorter than the average of all token lengths.
        '''
        average = 0
        for token in tokens:
            average += len(token)
        try:
            average = average / len(tokens)
            return ([x for x in tokens if len(x) >= average])
        except:
            return (tokens)
    #TO0DO:Remove All B-entities if they do not add anything useful to the model
    def generate_entity_list(self, entities):
        '''Function used for generating a list of entity numbers based on the amount of times they appear in the entity list.
        '''
        B_LOC, B_MISC, B_ORG, I_LOC, I_MISC, I_ORG, I_PER = 0, 0, 0, 0, 0, 0, 0
        for entity in entities:
            if entity == 'B-LOC':
                B_LOC += 1
            elif entity == 'B-MISC':
                B_MISC += 1
            elif entity == 'B-ORG':
                B_ORG += 1
            elif entity == 'I-LOC':
                I_LOC += 1
            elif entity == 'I-MISC':
                I_MISC += 1
            elif entity == 'I-ORG':
                I_ORG += 1
            elif entity == 'I-PER':
                I_PER += 1
        return ([I_LOC, I_MISC, I_ORG, I_PER])

    def generate_named_entities(self, text):
        ''' Function that generates named entity values for the inputted text.
        This Method does a few things. First it splits the text into single sentences(split_into_senteces). The short sentences are then removed based on the average length of the sentences in the text(remove_short_tokens)
        The tokens are then fed into a tokenizer and the generated tokens are fed into a model that generates named entities based on the tokens. The result of this is returned as a dict which can then be appended to the dataframe.
        them to the dataframe and removing the old one.
        '''
        tokens = [x for x in self.split_into_sentences(text) if not any(y in x for y in ['/','+'])]  # split text into sentences and remove any sentence that contains / or + as a character
        tokens = self.remove_short_tokens(tokens)
        if tokens:
            inputs = self._tokenizerner.batch_encode_plus(tokens, return_tensors="pt", padding=True, max_length=512, truncation=True)  # tokenize sentences, max_length is 512 for if cuda is enabled to speed the model up
            with torch.no_grad():
                results = self._modelner(**inputs)
                for i, _input in enumerate(inputs['input_ids']):
                    namedentities = [self._modelner.config.id2label[item.item()] for item in results.logits[i].argmax(axis=1)]  # for every probability for a named entity for a word, turn the probabilities into their associated labels
            entitynumberslist = self.generate_entity_list(namedentities)  # Based on the array of entity names that is generated, count each entity and make a dict of this
        else:
            entitynumberslist = [0,0,0,0]
        return entitynumberslist

    def remove_numbers_phonenumbers(self,text):
        ''' Function that remove specific sequences of numbers from a text so that they are not seen as words by the BagofWords feature.
        '''
        text = re.sub(r'\b([0-9]{3}-[0-9]{3}-[0-9]{4})\b', '', str(text)) #Remove US phone numbers
        text = re.sub(r'\b([0-1][0-9]\/[0-3][0-9]\/[0-9]{4})\b', '', text) #Removes dates
        text = re.sub(r'\b([0-1]?[0-9]):[0-5][0-9]\b', '', text) #Removes timestamps
        text = re.sub(r'\b\w*\d\w*\b', '', text) #Removes single whitespaces
        return text

    def doc_freq(self, word, DF):
        c = 0
        try:
            c = DF[word]
        except:
            pass
        return c
    #TODO refactor this function have it use all texts at once since otherwise it will not work
    def tf_idf(self, texts):
        N = len(texts)


        #Calculate DF for all words
        DF = {}

        for i in range(0, N):
            tokens = texts[i]
            for w in tokens:
                try:
                    DF[w].add(i)
                except:
                    DF[w] = {i}

        for i in DF:
            DF[i] = len(DF[i])

        doc = 0

        tf_idf = {}

        for i in range(0, N):

            tokens = texts[i]

            counter = Counter(tokens)
            words_count = len(tokens)

            for token in np.unique(tokens):
                tf = counter[token] / words_count
                df = self.doc_freq(token, DF)
                idf = np.log((N + 1) / (df + 1))

                tf_idf[doc, token] = tf * idf

            doc += 1
        return tf_idf

    def bag_of_words(self, texts, hash_length):
        processed_text_tf_idf = []
        processed_text_bow = []
        for text in texts:
            processed_text_tf_idf.append(word_tokenize(str(self.preprocess(text))))
            processed_text_bow.append(str(self.preprocess(text)))


        tf_idf_values = self.tf_idf(processed_text_tf_idf)

        BOW_df = pd.DataFrame(columns=[str(i) for i in range(0, hash_length)])

        for i in range(0,len(processed_text_bow)):
            BOW_values = self.Hashed_BOW(processed_text_bow[i], hash_length)
            BOW_df.loc[i] = BOW_values

        for key in tf_idf_values:
            BOW_df[str(hashing_trick(key[1], hash_length, lower=True)[0])].iloc[key[0]] = BOW_df[str(hashing_trick(key[1], hash_length, lower=True)[0])].iloc[key[0]] * tf_idf_values[key]

        return BOW_df.to_numpy()

    def Hashed_BOW(self, text, length):
        bow_array = [0] * length
        hashes = hashing_trick(
            text,
            length,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
            analyzer=None)
        for hash_ in hashes:
            bow_array[hash_] = bow_array[hash_] + 1

        return bow_array[0:length]

    def standard_dev_sentence_length(self,text):
        ''' Function that calulates the standard deviation of the length of all the sentences in a text.
        '''
        sentences = nltk.tokenize.sent_tokenize(text)
        sentence_length = []
        for item in sentences:
            sentence_length.append(len(item))
        return (np.std(sentence_length))

    def readability_index(self, text):
        ''' Function that calculates the automated readability index of a text.
        '''
        sentences = nltk.tokenize.sent_tokenize(text)
        words = text.count(' ')
        characters = len(text) - words
        try:
            return (4.71 * (characters / words) + 0.5 * (words / len(sentences)) - 21.43)
        except:
            return (0)

    def standard_dev_word_length(self,text):
        ''' Function that calculates the standard deviation of the word lengths in a text.
        '''
        words = text.split()
        words_length = []
        for word in words:
            words_length.append(len(word))
        return (np.std(words_length))

    def type_token_ratio(self,text):
        ''' Function that calculates the type token ratio of the text. The type token ratio is the ratio between the toal amount of words and the amount of unique words in a text.
        '''
        unique = set(text.split())
        return len(unique) / len(text.split())

    def proper_nouns(self,text):
        ''' Function that caclulates how many proper nouns there are in a text. This is done with the help of the NLTK perceptron tagger.
        '''
        tagged_sent = pos_tag(text.split())
        propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
        return len(propernouns)

    def percentage_passive_voice(self, text):
        ''' Function that caclulates what percentage of sentences in a text are written in passive voice. This is done using the passivePy package.
        '''
        passive_amount = self.passivepy.match_text(text, full_passive=True, truncated_passive=True).passive_count.iloc[0]
        sentence_amount = nltk.tokenize.sent_tokenize(text)
        return passive_amount / len(sentence_amount)

    def percentage_active_voice(self, text):
        ''' Function that caclulates what percentage of sentences in a text are written in active voice. This is done using the passivePy package.
        '''
        return (1 - self.percentage_passive_voice(text))

    def convert_lower_case(self,text):
        return np.char.lower(text)

    def remove_email_adresses(self, text):
        text = re.sub(r'\S*@\S*\s?', '', str(text))
        return text

    def remove_stop_words(self,text):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(text))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    def remove_punctuation(self,text):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            text = np.char.replace(text, symbols[i], ' ')
            text = np.char.replace(text, "  ", " ")
        text = np.char.replace(text, ',', '')
        return text

    def remove_apostrophe(self,text):
        return np.char.replace(text, "'", "")

    def stemming(self,text):
        stemmer = PorterStemmer()

        tokens = word_tokenize(str(text))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    def convert_numbers(self,text):
        tokens = word_tokenize(str(text))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    def remove_random_words(self, data):
        data = str(data)
        data = re.sub(r'\b(USL)?(usl)?\b', '', data)
        data = re.sub(r'\b(\.((DOC)?(doc)?))\b', '', data)
        data = re.sub(r'\b(e-mail)\b', '', data)
        data = re.sub(r'(\<div\>)', '', data)
        data = re.sub(r'(\<br\>)', '', data)
        data = re.sub(r'(.*?\.[\w:]+)', '', data)
        data = re.sub(r'\[image\]', '', data)
        return data

    def preprocess(self,text):
        text = self.convert_lower_case(text)
        text = self.remove_email_adresses(text)
        text = self.remove_random_words(text)
        text = self.remove_punctuation(text)  # remove comma seperately
        text = self.remove_apostrophe(text)
        text = self.remove_stop_words(text)
        text = self.remove_numbers_phonenumbers(text)
        text = self.stemming(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers_phonenumbers(text)
        text = self.stemming(text)  # needed again as we need to stem the words
        text = self.remove_punctuation(text)  # needed again as num2word is giving few hypens and commas fourty-one
        text = self.remove_stop_words(text)  # needed again as num2word is giving stop words 101 - one hundred and one
        return text