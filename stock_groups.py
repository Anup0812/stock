class StockGroupManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.initialize_default_groups()
    
    def initialize_default_groups(self):
        """Initialize default stock groups if they don't exist"""
        # This is where you could pre-populate groups with default stocks
        # For now, we'll just ensure the groups exist
        pass
    
    def add_stock_to_group(self, group_name, symbol):
        """Add a stock to a specific group"""
        return self.db.add_stock_to_group(group_name, symbol)
    
    def remove_stock_from_group(self, group_name, symbol):
        """Remove a stock from a specific group"""
        self.db.remove_stock_from_group(group_name, symbol)
    
    def get_group_stocks(self, group_name):
        """Get all stocks in a specific group"""
        return self.db.get_group_stocks(group_name)
    
    def get_all_groups(self):
        """Get all available groups"""
        return ["V40", "V40 Next", "V200"]
    
    def get_group_summary(self, group_name):
        """Get summary statistics for a group"""
        stocks = self.get_group_stocks(group_name)
        
        if not stocks:
            return {
                'total_stocks': 0,
                'avg_fundamental_score': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'watch_signals': 0
            }
        
        total_stocks = len(stocks)
        total_fundamental_score = 0
        buy_signals = 0
        sell_signals = 0
        watch_signals = 0
        
        for stock in stocks:
            signals = self.db.get_signals(stock['symbol'])
            if signals:
                total_fundamental_score += signals['fundamental_score']
                
                # Count signals
                signal_list = [
                    signals['sma_signal'],
                    signals['green_candle_signal'],
                    signals['range_bound_signal'],
                    signals['rhs_signal'],
                    signals['cup_handle_signal']
                ]
                
                for signal in signal_list:
                    if signal == 'BUY':
                        buy_signals += 1
                    elif signal == 'SELL':
                        sell_signals += 1
                    elif signal == 'WATCH':
                        watch_signals += 1
        
        return {
            'total_stocks': total_stocks,
            'avg_fundamental_score': total_fundamental_score / total_stocks if total_stocks > 0 else 0,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'watch_signals': watch_signals
        }
