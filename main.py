from sys import argv


# Your implemented classes from Ex1, you may change them here according to your needs:


class Token:

    def __init__(self):
        return


class Sentence:

    def __init__(self):
        return


class Corpus:

    def __init__(self):
        return


# Implement a "Classify" class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class). Make sure that the class contains the relevant fields for
# classification, and the methods in order to complete the tasks:


class Classify:

    def __init__(self):
        return


def main():
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path

    # Implement here your program:
    # 1. Create a corpus from the file in the given directory (up to 1000 XML files from the BNC)
    # 2. Create a classification object based on the class implemented above.
    # 3. Classify the chunks of text from the corpus as described in the instructions.
    # 4. Print onto the output file the results from the second task in the wanted format.


if __name__ == '__main__':
    main()
