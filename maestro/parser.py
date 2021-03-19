class Parser:
    @staticmethod
    def parse_textfile_file(filename):
        return open(filename, "r").read()

    @staticmethod
    def parse_telegram_chat_file(filename):
        pass
