import argparse
import os
import sys

import nltk
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from maestro import __description__
from maestro.config import Mode
from maestro.parser import Parser


class Maestro:
    def __init__(self, filename, parser, options):
        self.filename = filename
        self.parser = parser
        self.options = options

        self.model_filename = f"data/{os.path.basename(self.filename)}.hdf5"

    def run(self):
        self.file = self.parser(self.filename)
        self.processed_input = self.tokenize_words(self.file)
        self.transform_chars_to_numbers()
        self.create_dataset()
        self.convert_to_np_array()
        self.define_lstm_model()

        if self.options["mode"] == Mode.learn:
            self.create_lstm_model()
            self.learn()

        if self.options["mode"] == Mode.generate:
            self.num_to_char = dict((i, c) for i, c in enumerate(self.chars))
            self.load_lstm_model()
            self.get_random_seed()
            self.generate()

    def exit(self):
        sys.exit(0)

    def tokenize_words(self, text):
        text = text.lower()

        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)

        filtered = filter(lambda token: token not in stopwords.words("russian"), tokens)
        return " ".join(filtered)

    def transform_chars_to_numbers(self):
        self.chars = sorted(list(set(self.processed_input)))
        self.char_to_num = dict((c, i) for i, c in enumerate(self.chars))

        self.input_len = len(self.processed_input)
        self.vocab_len = len(self.chars)

        print("Total number of characters:", self.input_len)
        print("Total vocab:", self.vocab_len)

    def create_dataset(self):
        self.seq_length = 100
        self.x_data = []
        self.y_data = []

        for i in range(0, self.input_len - self.seq_length, 1):
            in_seq = self.processed_input[i : i + self.seq_length]
            out_seq = self.processed_input[i + self.seq_length]

            self.x_data.append([self.char_to_num[char] for char in in_seq])
            self.y_data.append(self.char_to_num[out_seq])

        self.n_patterns = len(self.x_data)
        print("Total Patterns:", self.n_patterns)

    def convert_to_np_array(self):
        X = numpy.reshape(self.x_data, (self.n_patterns, self.seq_length, 1))
        self.X = X / float(self.vocab_len)
        self.y = np_utils.to_categorical(self.y_data)

    def define_lstm_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                256,
                input_shape=(self.X.shape[1], self.X.shape[2]),
                return_sequences=True,
            )
        )
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.y.shape[1], activation="softmax"))

    def create_lstm_model(self):
        checkpoint = ModelCheckpoint(
            self.model_filename,
            monitor="loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )
        self.desired_callbacks = [checkpoint]

    def learn(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        self.model.fit(
            self.X,
            self.y,
            epochs=self.options["epoch_count"],
            batch_size=256,
            callbacks=self.desired_callbacks,
        )

    def load_lstm_model(self):
        self.model.load_weights(self.model_filename)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")

    def get_random_seed(self):
        self.pattern = self.x_data[numpy.random.randint(0, len(self.x_data) - 1)]

        print("Random Seed:")
        print('"', "".join([self.num_to_char[value] for value in self.pattern]), '"')

    def generate(self):
        for i in range(1000):
            x = numpy.reshape(self.pattern, (1, len(self.pattern), 1))
            x = x / float(self.vocab_len)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = self.num_to_char[index]

            sys.stdout.write(result)

            self.pattern.append(index)
            self.pattern = self.pattern[1 : len(self.pattern)]


def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("--learn", action="store_true", help="learn mode")
    parser.add_argument("--generate", action="store_true", help="generate mode")
    parser.add_argument(
        "-f",
        dest="textfile_filename",
        action="store",
        help="get sample from plain text file",
    )
    parser.add_argument(
        "-t",
        dest="telegram_chat_filename",
        action="store",
        help="get sample from telegram chat HTML file",
    )
    parser.add_argument(
        "-e",
        dest="epoch_count",
        action="store",
        default=4,
        help="number of epochs to learn",
    )
    args = parser.parse_args()

    options = {}

    if args.learn:
        options["mode"] = Mode.learn
        options["epoch_count"] = int(args.epoch_count)
    elif args.generate:
        options["mode"] = Mode.generate

    if args.textfile_filename is not None:
        filename = args.textfile_filename
        parser = Parser.parse_textfile_file
    elif args.telegram_chat_filename is not None:
        filename = args.telegram_chat_filename
        parser = Parser.parse_telegram_chat_file
    else:
        print("Nothing to do")
        sys.exit(0)

    app = Maestro(filename, parser, options)
    app.run()
    app.exit()


if __name__ == "__main__":
    main()
