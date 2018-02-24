from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import kenlm


class FeatureExtractor():
    def __init__(self, psycho_path=None, lm_books_path=None, lm_news_path=None, embedding_model_path=None):
        if psycho_path:
            self.df = pd.read_csv(psycho_path, sep='\t')
            self.df_mean = self.df.mean(axis=0)
        if lm_books_path:
            self.lm_books = kenlm.LanguageModel(lm_books_path)
        if lm_news_path:
            self.lm_news = kenlm.LanguageModel(lm_news_path)
        if embedding_model_path:
            self.embeddings = KeyedVectors.load_word2vec_format(embedding_model_path, binary=True)

    def psycholinguistics(self, words):
        """Extract psycholinguistic features."""

        psychos = []
        for word in words:
            psycho = {'Familiarity': 0, 'Age_of_Acquisition':0, 'Concreteness':0,'Imagery': 0}
            infos = self.df[self.df.Word == word]
            if not infos.empty:
                for key in psycho.keys():
                    psycho[key] += infos[key].values[0]
            else:
                for key in psycho.keys():
                    psycho[key] += self.df_mean[key]
            psychos.append(pd.Series(psycho, index=psycho.keys()))
        return pd.DataFrame(psychos).mean().to_dict()

    def language_model(self, tokens):
        """Extract language model features."""

        model = {'LM-Book_log10':0, 'LM-News_log10':0}    
        model['LM-Book_log10'] = self.lm_books.score(' '.join(tokens), bos=False, eos=False)
        model['LM-News_log10'] = self.lm_news.score(' '.join(tokens), bos=False, eos=False)
        return model

    def predict_average_embeddings(self, instances):
        """Extract average value of target words embeddings."""
        
        data_embeddings = []
        for index, instance in enumerate(instances):
            words = []
            for i in instance.target:
                if instance.tokens[i] in self.embeddings:
                    words.append(self.embeddings[instance.tokens[i]])
            if len(words) == 0:
                words.append(self.embeddings['unk'])
            data_embeddings.append(np.average(words, axis=0))
        return np.asarray(data_embeddings, )

    def predict_linguistics(self, instances):
        """Extract features for every instance."""

        x = pd.DataFrame()
        for instance in instances:
            features = dict()
            tokens = [instance.tokens[i] for i in instance.target]

            features.update(self.psycholinguistics(tokens))
            features.update(self.language_model(tokens))
            x = x.append(pd.DataFrame([features]))
        return x.reset_index().drop('index', axis=1)