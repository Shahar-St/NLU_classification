import itertools
import os
import time
import xml.etree.ElementTree as ET
from enum import Enum
from sys import argv

import gender_guesser.detector as gender
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Token:
    def __init__(self, value):
        self.value = value


class Gender(Enum):
    Male = 1
    Female = 2
    Unknown = 3

    @staticmethod
    def get_gender(name):
        detected_gender = gender.Detector().get_gender(name)
        genders_map = {
            'male': Gender.Male,
            'female': Gender.Female,
        }
        return genders_map.get(detected_gender, Gender.Unknown)


class Sentence:
    def __init__(self, tokens_array, authors_list, author_gender: Gender):
        self.tokens = tokens_array
        self.authors_list = authors_list
        self.author_gender = author_gender


class Chunk:
    def __init__(self, sentences):
        self.sentences = sentences
        all_genders = [sen.author_gender for sen in sentences]
        self.overall_gender = all_genders[0] if len(set(all_genders)) == 1 else Gender.Unknown

    def get_all_words(self):
        tokens_lists = [sen.tokens for sen in self.sentences]
        tokens = list(itertools.chain.from_iterable(tokens_lists))
        words = [tok.value for tok in tokens]
        return words


class Chunks:
    def __init__(self, chunks):
        self.chunks = chunks

    def __getitem__(self, item):
        sentences = self.chunks[item].sentences
        tokens_lists = [sen.tokens for sen in sentences]
        tokens = np.concatenate(tokens_lists)
        words = np.array([tok.value for tok in tokens])
        return ' '.join(words)


class Corpus:
    def __init__(self):
        self.sentences = []
        self.chunks = []

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus.
        :param: file_name: The name of the XML file that will be read
        :return: None
        """
        tree = ET.parse(file_name)
        # get author/s
        authors = []
        for author in tree.iter(tag='author'):
            authors.append(author.text)

        if len(authors) == 1 and len(authors[0].split(',')) == 2:
            author_gender = Gender.get_gender(authors[0].split(',')[1].strip())
        else:
            author_gender = Gender.Unknown

        # iterate over all sentences in the file and extract the tokens
        for sentence in tree.iter(tag='s'):
            tokens = []
            for word in sentence:
                if word.tag in ('w', 'c') and isinstance(word.text, str):
                    new_token = Token(value=word.text.strip())
                    tokens.append(new_token)

            tokens = np.array(tokens)
            new_sentence = Sentence(tokens, authors, author_gender)
            self.sentences.append(new_sentence)

    def set_sentences_to_np(self):
        self.sentences = np.array(self.sentences)

    def calculate_chunks(self):
        chunks = []
        sentences_counter = 0
        while sentences_counter <= len(self.sentences) + 10:
            sentences = self.sentences[sentences_counter: sentences_counter + 10]
            chunks.append(Chunk(sentences))
            sentences_counter += 10
        self.chunks = np.array(chunks)


class Classify:

    def __init__(self, corpus):
        self.corpus = corpus
        self.male_chunks = np.array([chunk for chunk in self.corpus.chunks if chunk.overall_gender == Gender.Male])
        self.male_chunks_size = self.male_chunks.size
        self.female_chunks = np.array([chunk for chunk in self.corpus.chunks if chunk.overall_gender == Gender.Female])
        self.female_chunks_size = self.female_chunks.size

    def get_male_female_chunks_count(self):
        return self.male_chunks_size, self.female_chunks_size

    def even_out_classes(self):
        target_num_of_samples = min(self.male_chunks_size, self.female_chunks_size)
        if self.male_chunks_size > self.female_chunks_size:
            self.male_chunks = np.random.choice(self.male_chunks, size=target_num_of_samples, replace=False)
            self.male_chunks_size = target_num_of_samples
        else:
            self.female_chunks = np.random.choice(self.female_chunks, size=target_num_of_samples, replace=False)
            self.female_chunks_size = target_num_of_samples

    def classify(self, vector_method):
        # get train data and labels
        chunks = Chunks(np.concatenate([self.male_chunks, self.female_chunks]))
        train_data = self.get_data_by_BoW(chunks) if vector_method == 'BoW' else \
            self.get_data_by_personal_vector(chunks)

        train_labels = np.array([c.overall_gender.value for c in chunks.chunks])
        target_names = ['Male', 'Female']

        # model w 10-fold cross-validation
        neigh_cross_val = KNeighborsClassifier()
        neigh_cross_val.fit(train_data, train_labels)
        cross_val_score_list = cross_val_score(neigh_cross_val, train_data, train_labels, cv=10)
        # todo all data?
        cross_val_predicted = neigh_cross_val.predict(train_data)
        cross_val_report = classification_report(train_labels, cross_val_predicted, target_names=target_names)

        # model w 70:30 split validation
        neigh_split_val = KNeighborsClassifier()
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3)
        neigh_split_val.fit(X_train, y_train)
        reg_val_score = neigh_split_val.score(X_test, y_test)

        reg_val_predicted = neigh_split_val.predict(train_data)
        reg_val_report = classification_report(train_labels, reg_val_predicted, target_names=target_names)

        return cross_val_score_list, cross_val_report, reg_val_score, reg_val_report

    @staticmethod
    def get_data_by_BoW(chunks):
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(chunks)
        return data

    def get_data_by_personal_vector(self, chunks):
        # get top occur words for each gender
        male_words = []
        female_words = []
        for male_chunk, female_chunk in zip(self.male_chunks, self.female_chunks):
            male_words.extend(male_chunk.get_all_words())
            female_words.extend(female_chunk.get_all_words())

        male_words = np.array(male_words)
        female_words = np.array(female_words)

        male_unique, male_counts = np.unique(male_words, return_counts=True)
        female_unique, female_counts = np.unique(female_words, return_counts=True)
        counter_size = min(male_unique.size, female_unique.size)

        male_sorted_counter = \
            {k: v for k, v in
             sorted(dict(zip(male_unique, male_counts)).items(), key=lambda item: item[1], reverse=True)}
        female_sorted_counter = \
            {k: v for k, v in
             sorted(dict(zip(female_unique, female_counts)).items(), key=lambda item: item[1], reverse=True)}
        male_sorted_counter = dict(itertools.islice(male_sorted_counter.items(), counter_size))
        female_sorted_counter = dict(itertools.islice(female_sorted_counter.items(), counter_size))

        # give each word a score
        scores_dict = {}
        for word in np.unique(np.concatenate([male_words, female_words])):
            scores_dict[word] = abs(male_sorted_counter.get(word, 0) - female_sorted_counter.get(word, 0))

        # pick the top words
        words_sorted_by_importance = np.array(
            [k for k, v in sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)])
        voc_size = min(words_sorted_by_importance.size, 1000)
        vocab = np.array(words_sorted_by_importance[:voc_size])

        vectorizer = CountVectorizer()
        vectorizer.fit_transform(vocab)
        data = vectorizer.transform(chunks)
        return data


def main():
    print('Program started')
    start_time = time.time()
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    # 1. Create a corpus from the file in the given directory (up to 1000 XML files from the BNC)
    print('Corpus Building - In Progress...')
    corpus = Corpus()
    xml_files_names = os.listdir(xml_dir)
    for file in xml_files_names[:min(len(xml_files_names), 1000)]:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    corpus.set_sentences_to_np()
    corpus.calculate_chunks()
    print('Corpus Building - Done!')

    # 2. Create a classification object based on the class implemented above.
    print('classify Building - In Progress...')
    classify = Classify(corpus)
    male_count, female_count = classify.get_male_female_chunks_count()
    output_str = f'Before Down-sampling:\nFemale: {female_count} Male {male_count}\n\n'
    classify.even_out_classes()
    male_count, female_count = classify.get_male_female_chunks_count()
    output_str += f'After Down-sampling:\nFemale: {female_count} Male {male_count}\n\n'
    print('classify Building - Done!')

    # 3. Classify the chunks of text from the corpus as described in the instructions.
    print('Classification - In Progress...')
    vectors_methods = ['BoW', 'Custom Feature Vector']
    for vector_method in vectors_methods:
        output_str += f'== {vector_method} Classification ==\n'
        cross_val_score_list, cross_val_report, reg_val_score, reg_val_report = classify.classify(vector_method)
        # todo which one to take?
        # todo what to do with the val score
        output_str += f'Cross Validation Accuracy: {cross_val_score_list[-1] * 100:.3f}% \n{cross_val_report}\n\n'
    print('Classification - Done!')

    # 4. Print onto the output file the results from the second task in the wanted format.
    print(f'Writing output to {output_file}')
    with open(output_file, 'w', encoding='utf8') as output_file:
        output_file.write(output_str)
    print(f'Program ended.')
    print(f'Run time: {(time.time() - start_time):.2f} seconds.')


if __name__ == '__main__':
    main()
