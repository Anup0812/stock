import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_path="stock_analysis.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Stock groups table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_name, symbol)
                )
            """)
            
            # Stock info table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    current_price REAL,
                    market_cap REAL,
                    pe_ratio REAL,
                    debt_to_equity REAL,
                    roe REAL,
                    sector TEXT,
                    industry TEXT,
                    fifty_two_week_high REAL,
                    fifty_two_week_low REAL,
                    lifetime_high REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Historical data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Technical signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_signals (
                    symbol TEXT PRIMARY KEY,
                    sma_signal TEXT,
                    green_candle_signal TEXT,
                    range_bound_signal TEXT,
                    rhs_signal TEXT,
                    cup_handle_signal TEXT,
                    fundamental_score REAL,
                    average_rating TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER,
                    avg_buy_price REAL,
                    total_investment REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def add_stock_to_group(self, group_name, symbol):
        """Add a stock to a specific group"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO stock_groups (group_name, symbol) 
                    VALUES (?, ?)
                """, (group_name, symbol))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # Stock already exists in group
    
    def get_group_stocks(self, group_name):
        """Get all stocks in a specific group"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM stock_groups 
                WHERE group_name = ? 
                ORDER BY symbol
            """, (group_name,))
            
            return [{'symbol': row[0]} for row in cursor.fetchall()]
    
    def remove_stock_from_group(self, group_name, symbol):
        """Remove a stock from a specific group"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM stock_groups 
                WHERE group_name = ? AND symbol = ?
            """, (group_name, symbol))
            conn.commit()
    
    def store_stock_info(self, stock_info):
        """Store or update stock information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO stock_info (
                    symbol, company_name, current_price, market_cap, pe_ratio,
                    debt_to_equity, roe, sector, industry, fifty_two_week_high,
                    fifty_two_week_low, lifetime_high, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stock_info['symbol'],
                stock_info['company_name'],
                stock_info['current_price'],
                stock_info['market_cap'],
                stock_info['pe_ratio'],
                stock_info['debt_to_equity'],
                stock_info['roe'],
                stock_info['sector'],
                stock_info['industry'],
                stock_info['fifty_two_week_high'],
                stock_info['fifty_two_week_low'],
                stock_info['lifetime_high'],
                stock_info['updated_at']
            ))
            conn.commit()
    
    def get_stock_info(self, symbol):
        """Get stock information for a specific symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stock_info WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def store_historical_data(self, symbol, hist_data):
        """Store historical data for a stock"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing data for this symbol
            cursor.execute("DELETE FROM historical_data WHERE symbol = ?", (symbol,))
            
            # Insert new data
            for date, row in hist_data.iterrows():
                cursor.execute("""
                    INSERT INTO historical_data (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ))
            
            conn.commit()
    
    def get_historical_data(self, symbol):
        """Get historical data for a specific symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, open, high, low, close, volume 
                FROM historical_data 
                WHERE symbol = ? 
                ORDER BY date
            """, (symbol,))
            
            rows = cursor.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            return pd.DataFrame()
    
    def store_signals(self, signals):
        """Store technical analysis signals"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO technical_signals (
                    symbol, sma_signal, green_candle_signal, range_bound_signal,
                    rhs_signal, cup_handle_signal, fundamental_score, average_rating,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signals['symbol'],
                signals['sma_signal'],
                signals['green_candle_signal'],
                signals['range_bound_signal'],
                signals['rhs_signal'],
                signals['cup_handle_signal'],
                signals['fundamental_score'],
                signals['average_rating'],
                signals['updated_at']
            ))
            conn.commit()
    
    def get_signals(self, symbol):
        """Get technical analysis signals for a specific symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM technical_signals WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def add_portfolio_stock(self, symbol, quantity, buy_price):
        """Add or update a stock in the portfolio"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if stock already exists
            cursor.execute("SELECT * FROM portfolio WHERE symbol = ?", (symbol,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing stock - average the price
                existing_qty = existing[1]
                existing_avg_price = existing[2]
                existing_investment = existing[3]
                
                new_investment = quantity * buy_price
                total_investment = existing_investment + new_investment
                total_quantity = existing_qty + quantity
                new_avg_price = total_investment / total_quantity
                
                cursor.execute("""
                    UPDATE portfolio 
                    SET quantity = ?, avg_buy_price = ?, total_investment = ?, updated_at = ?
                    WHERE symbol = ?
                """, (total_quantity, new_avg_price, total_investment, datetime.now(), symbol))
            else:
                # Add new stock
                total_investment = quantity * buy_price
                cursor.execute("""
                    INSERT INTO portfolio (symbol, quantity, avg_buy_price, total_investment)
                    VALUES (?, ?, ?, ?)
                """, (symbol, quantity, buy_price, total_investment))
            
            conn.commit()
    
    def get_portfolio_stocks(self):
        """Get all stocks in the portfolio"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, quantity, avg_buy_price, total_investment 
                FROM portfolio 
                ORDER BY symbol
            """)
            
            return [
                {
                    'symbol': row[0],
                    'quantity': row[1],
                    'avg_buy_price': row[2],
                    'total_investment': row[3]
                }
                for row in cursor.fetchall()
            ]
    
    def remove_portfolio_stock(self, symbol):
        """Remove a stock from the portfolio"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
            conn.commit()
