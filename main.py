import os
from sys import argv
import xml.etree.ElementTree as ET
import gender_guesser.detector as gender
from enum import Enum


class Token:
    SENTENCE_START_TOK = '<s>'
    SENTENCE_END_TOK = '</s>'

    def __init__(self, token_type, value, c5=None, hw=None, pos=None):
        self.token_type = token_type  # w (word), c (punctuation marks), s (beg/end of sentence)
        self.value = value
        self.c5 = c5  # The C5 tag
        self.hw = hw  # headword
        self.pos = pos  # part of speech, derived from the c5 tag


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
    def __init__(self, tokens_array, index: int, authors_list, author_gender: Gender):
        self.tokens = tokens_array
        self.tokens_num = len(tokens_array)
        self.index = index  # starts with 1
        self.authors_list = authors_list
        self.author_gender = author_gender


class Corpus:
    def __init__(self):
        self.sentences = []
        self.num_of_words = 0
        self.sentences_lengths = []
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

        author_gender = Gender.get_gender(authors[0].split(',')[1].strip()) if len(authors) == 1 else Gender.Unknown

        # iterate over all sentences in the file and extract the tokens
        for sentence in tree.iter(tag='s'):
            # Adding sentence start token at the beginning of the sentence
            tokens = [Token('s', Token.SENTENCE_START_TOK)]
            # tokens = []
            for word in sentence:
                if word.tag in ('w', 'c'):
                    att = word.attrib
                    new_token = Token(
                        token_type=word.tag,
                        c5=att['c5'],
                        hw=att.get('hw'),
                        pos=att.get('pos'),
                        value=word.text.strip()
                    )
                    tokens.append(new_token)
                    self.num_of_words += 1

            # Adding sentence end token at the end of the sentence
            tokens.append(Token('s', Token.SENTENCE_END_TOK))
            new_sentence = Sentence(tokens, int(sentence.attrib['n']), authors, author_gender)
            self.sentences.append(new_sentence)
            # Saving the sentence length. Will be used in the random sentence generation
            self.sentences_lengths.append(len(sentence))

    def get_tokens(self):
        # get a list of all tokens in their lower case form
        tokens_list = []
        for sen in self.sentences:
            tokens_list.extend([tok.value.lower() for tok in sen.tokens])
        return tokens_list

    def calculate_chunks(self):
        chunks = []
        sentences_counter = 0
        while sentences_counter <= len(self.sentences) + 10:
            chunks.append(self.sentences[sentences_counter: sentences_counter + 10])
            sentences_counter += 10
        self.chunks = chunks


# Implement a "Classify" class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class). Make sure that the class contains the relevant fields for
# classification, and the methods in order to complete the tasks:


class Classify:

    def __init__(self):
        return


def main():
    print('Program started')
    # xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    xml_dir = os.path.join(os.getcwd(), 'XML_files')
    # output_file = argv[2]  # output file name, full path
    output_file = os.path.join(os.getcwd(), 'output.txt')

    # 1. Create a corpus from the file in the given directory (up to 1000 XML files from the BNC)
    print('Initializing Corpus')
    c = Corpus()
    print('XML files Additions - In Progress...')
    xml_files_names = os.listdir(xml_dir)
    for file in xml_files_names[:min(len(xml_files_names), 1000)]:
        c.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    c.calculate_chunks()
    print('XML files Additions - Done!')

    # 2. Create a classification object based on the class implemented above.

    # 3. Classify the chunks of text from the corpus as described in the instructions.
    # 4. Print onto the output file the results from the second task in the wanted format.


if __name__ == '__main__':
    main()
