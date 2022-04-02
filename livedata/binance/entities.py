class LiveData:
    def __init__(self, data):
        self.symbol = data["s"]
        self.price_change = float(data["p"])
        self.price_change_percent = float(data["P"])
        self.weighted_average_price = float(data["w"])
        self.last_price = float(data["c"])
        self.last_quantity = float(data["Q"])
        self.bid_price = float(data["b"])
        self.bid_quantity = float(data["B"])
        self.ask_price = float(data["a"])
        self.ask_quantity = float(data["A"])
        self.open = float(data["o"])
        self.high = float(data["h"])
        self.low = float(data["l"])
        self.volume = float(data["v"])