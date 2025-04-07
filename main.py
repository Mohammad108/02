import sys
from PyQt5 import QtWidgets, uic
from rl_agent import RLAgent
from rl_env import TradingEnv
import MetaTrader5 as mt5
from utils import get_symbol_data

class TraderBotApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(TraderBotApp, self).__init__()
        uic.loadUi("ui.ui", self)
        self.agent = RLAgent()
        self.env = None

        self.trainButton.clicked.connect(self.train_model)
        self.loadButton.clicked.connect(self.load_model)
        self.tradeButton.clicked.connect(self.trade_live)

    def train_model(self):
        symbol = self.symbolInput.text()
        data = get_symbol_data(symbol, 500)
        self.env = TradingEnv(data)
        self.agent.train(self.env)
        self.log.append("✅ آموزش مدل تمام شد.")

    def load_model(self):
        self.agent.load("models/rl_model.zip")
        self.log.append("✅ مدل بارگذاری شد.")

    def trade_live(self):
        if not mt5.initialize():
            self.log.append("❌ اتصال به متاتریدر برقرار نشد.")
            return
        symbol = self.symbolInput.text()
        data = get_symbol_data(symbol, 100)
        self.env = TradingEnv(data)
        obs = self.env.reset()
        done = False
        while not done:
            action, _ = self.agent.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
        self.log.append(f"📈 ترید زنده انجام شد: {info}")

app = QtWidgets.QApplication(sys.argv)
window = TraderBotApp()
window.show()
sys.exit(app.exec_())