import itertools
import os
import xml.etree.ElementTree as ET
from enum import Enum
from sys import argv
from threading import Thread

import gender_guesser.detector as gender
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.size = chunks.size

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

    def add_xml_file_to_corpus(self, file_name: str, return_list=False):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus.
        :param: file_name: The name of the XML file that will be read
        :return: None
        """
        tree = ET.parse(file_name)
        # get author/s
        authors = [author.text for author in tree.iter(tag='author')]
        if len(authors) == 1 and len(authors[0].split(',')) == 2:
            author_gender = Gender.get_gender(authors[0].split(',')[1].strip())
        else:
            author_gender = Gender.Unknown

        # iterate over all sentences in the file and extract the tokens
        sentences = []
        for sentence in tree.iter(tag='s'):
            tokens = np.array([Token(value=word.text.strip()) for word in sentence if
                               word.tag in ('w', 'c') and isinstance(word.text, str)])

            new_sentence = Sentence(tokens, authors, author_gender)
            if return_list:
                sentences.append(new_sentence)
            else:
                self.sentences.append(new_sentence)

        if return_list:
            return sentences

    def bulk_add_xml_to_corpus(self, xml_dir, xml_files_names):
        all_sentences = []
        for file in xml_files_names:
            all_sentences.extend(self.add_xml_file_to_corpus(os.path.join(xml_dir, file), return_list=True))

        self.sentences.extend(all_sentences)

    def calculate_chunks(self):
        self.sentences = np.array(self.sentences)
        chunks = []
        sentences_counter = 0
        while sentences_counter <= self.sentences.size + 10:
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
        print(f'Running classify with method = {vector_method}')
        print(f'Building input data')
        chunks = np.concatenate([self.male_chunks, self.female_chunks])
        np.random.shuffle(chunks)
        chunks = Chunks(chunks)
        train_data = self.get_data_by_BoW(chunks) if vector_method == 'BoW' else \
            self.get_data_by_personal_vector(chunks)
        train_labels = np.array([c.overall_gender.value for c in chunks.chunks])

        print('Calculating cross validation score')
        n_jobs = 10
        neigh = KNeighborsClassifier(n_jobs=n_jobs)
        cross_val_scores_list = cross_val_score(neigh, train_data, train_labels, cv=10, n_jobs=n_jobs)

        print('Train phase')
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, shuffle=True)
        neigh.fit(X_train, y_train)

        print('Validation phase')
        predicted = neigh.predict(X_test)
        target_names = ['Male', 'Female']
        report = classification_report(y_test, predicted, target_names=target_names)

        return cross_val_scores_list, report

    @staticmethod
    def get_data_by_BoW(chunks):
        vectorizer = TfidfVectorizer(max_features=1000)
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

        data = []
        chunk_num = 0
        while chunk_num < chunks.size:
            chunk_str = chunks[chunk_num]
            chunk_vec = []
            for i, word in np.ndenumerate(vocab):
                chunk_vec.append(chunk_str.count(word))
            data.append(np.array(chunk_vec))
            chunk_num += 1
        data = csr_matrix(data)

        return data


def main():
    print('Program started')
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    # 1. Create a corpus from the file in the given directory (up to 1000 XML files from the BNC)
    print('Corpus Building - In Progress...')
    corpus = Corpus()
    xml_files_names = os.listdir(xml_dir)
    # limit files to 1000
    xml_files_names = np.array(xml_files_names[:min(len(xml_files_names), 1000)])

    # The corpus building is a bottleneck, so I used threads to accelerate the process
    num_of_threads = 5 if xml_files_names.size >= 100 else 1
    thread_work_size = int(xml_files_names.size / num_of_threads)
    threads = []
    i = 0
    while i < xml_files_names.size:
        thread = Thread(target=corpus.bulk_add_xml_to_corpus,
                        args=(xml_dir, xml_files_names[i:min(i + thread_work_size, xml_files_names.size)]))
        thread.start()
        threads.append(thread)
        i += thread_work_size

    for thread in threads:
        thread.join()

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
        cross_val_scores_list, report = classify.classify(vector_method)
        output_str += f'Cross Validation Accuracy: {np.mean(cross_val_scores_list) * 100:.3f}% \n{report}\n\n'
    print('Classification - Done!')

    # 4. Print onto the output file the results from the second task in the wanted format.
    print(f'Writing output to {output_file}')
    with open(output_file, 'w', encoding='utf8') as output_file:
        output_file.write(output_str)
    print(f'Program ended.')


if __name__ == '__main__':
    main()
