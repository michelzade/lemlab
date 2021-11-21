import urllib
import requests
import logging
import yaml


class SenderTelegram:
    __log__ = logging.getLogger(__name__)

    def __init__(self, cfg):
        self.cfg = cfg
        self.bot_token = self.cfg.get('bot_token')
        self.chat_ids = self.cfg.get('chat_ids')

    def send_msg(self, message):
        for chat_id in self.chat_ids:
            url = 'https://api.telegram.org/bot%s/sendMessage?chat_id=%i&text=%s'
            text = urllib.parse.quote_plus(message.encode('utf-8'))
            qry = url % (self.bot_token, chat_id, text)
            self.__log__.debug("Retrieving URL %s" % qry)
            resp = requests.get(qry)
            self.__log__.debug("Got response (%i): %s" % (resp.status_code, resp.content))
            data = resp.json()

            # handle error
            if resp.status_code != 200:
                sc = resp.status_code
                self.__log__.error("When sending bot message, we got status %i with message: %s" % (sc, data))


if __name__ == '__main__':
    # load telegram configuration
    with open(f"telegram_config.yaml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    sender = SenderTelegram({'bot_token': config['telegram']['bot_token'], 'chat_ids': config['telegram']['chat_ids']})

    sender.send_msg(message='Das ist unser neuer Plattform-Bot, der uns Fehlernachrichten direkt aufs Handy schickt!')
