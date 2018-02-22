import pandas as pd
import kenlm

class FeatureExtractor():
    def __init__(self, psycho_path, lm_books_path, lm_news_path):
        if psycho_path:
            self.df = pd.read_csv(psycho_path, sep='\t')
            self.df_mean = self.df.mean(axis=0)
        if lm_books_path:
            self.lm_books = kenlm.LanguageModel(lm_books_path)
        if lm_news_path:
            self.lm_news = kenlm.LanguageModel(lm_news_path)

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

    def predict(self, instances):
        """Extract features for every instance."""

        x = pd.DataFrame()
        for instance in instances:
            features = dict()
            tokens = [instance.tokens[i] for i in instance.target]

            features.update(self.psycholinguistics(tokens))
            features.update(self.language_model(tokens))
            x = x.append(pd.DataFrame([features]))
        return x.reset_index().drop('index', axis=1)