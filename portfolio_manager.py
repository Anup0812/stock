class PortfolioManager:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def add_stock(self, symbol, quantity, buy_price):
        """Add a stock to the portfolio"""
        self.db.add_portfolio_stock(symbol, quantity, buy_price)
    
    def remove_stock(self, symbol):
        """Remove a stock from the portfolio"""
        self.db.remove_portfolio_stock(symbol)
    
    def get_portfolio_stocks(self):
        """Get all stocks in the portfolio"""
        return self.db.get_portfolio_stocks()
    
    def calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        portfolio_stocks = self.get_portfolio_stocks()
        total_investment = 0
        total_current_value = 0
        
        for stock in portfolio_stocks:
            stock_info = self.db.get_stock_info(stock['symbol'])
            if stock_info:
                current_price = stock_info['current_price']
                investment = stock['quantity'] * stock['avg_buy_price']
                current_value = stock['quantity'] * current_price
                
                total_investment += investment
                total_current_value += current_value
        
        return {
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_pnl': total_current_value - total_investment,
            'total_pnl_percent': ((total_current_value - total_investment) / total_investment) * 100 if total_investment > 0 else 0
        }
    
    def get_portfolio_performance(self):
        """Get detailed portfolio performance"""
        portfolio_stocks = self.get_portfolio_stocks()
        performance_data = []
        
        for stock in portfolio_stocks:
            stock_info = self.db.get_stock_info(stock['symbol'])
            if stock_info:
                current_price = stock_info['current_price']
                investment = stock['quantity'] * stock['avg_buy_price']
                current_value = stock['quantity'] * current_price
                pnl = current_value - investment
                pnl_percent = (pnl / investment) * 100 if investment > 0 else 0
                
                performance_data.append({
                    'symbol': stock['symbol'],
                    'company_name': stock_info['company_name'],
                    'quantity': stock['quantity'],
                    'avg_buy_price': stock['avg_buy_price'],
                    'current_price': current_price,
                    'investment': investment,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent
                })
        
        return performance_data
