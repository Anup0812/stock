import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysis:
    def __init__(self):
        pass

    def calculate_sma_signal(self, hist_data):
        """Calculate Simple Moving Average signals as per defined strategy rules"""
        if len(hist_data) < 200:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'sma_20': pd.Series([0]),
                'sma_50': pd.Series([0]),
                'sma_200': pd.Series([0]),
                'reasoning': 'Insufficient data for SMA calculation'
            }

        # Calculate SMAs
        sma_20 = hist_data['Close'].rolling(window=20).mean()
        sma_50 = hist_data['Close'].rolling(window=50).mean()
        sma_200 = hist_data['Close'].rolling(window=200).mean()

        current_price = hist_data['Close'].iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        current_sma_200 = sma_200.iloc[-1]

        # Check bullish and bearish conditions count
        bullish_conditions = [
            current_sma_20 > current_sma_50,
            current_sma_50 > current_sma_200,
            current_price > current_sma_20
        ]
        bearish_conditions = [
            current_sma_20 < current_sma_50,
            current_sma_50 < current_sma_200,
            current_price < current_sma_20
        ]

        bullish_count = sum(bullish_conditions)
        bearish_count = sum(bearish_conditions)

        # Determine signal based on counts
        if bullish_count == 3:
            signal = 'SELL'
            reasoning = 'Strong bullish alignment: Price above all SMAs in correct order - SELL signal'
        elif bearish_count == 3:
            signal = 'BUY'
            reasoning = 'Strong bearish alignment: Price below all SMAs in correct order - BUY signal'
        elif bullish_count >= 2:
            signal = 'WATCH'
            reasoning = f'Partial bullish alignment ({bullish_count}/3 conditions met) - WATCH signal'
        elif bearish_count >= 2:
            signal = 'WATCH'
            reasoning = f'Partial bearish alignment ({bearish_count}/3 conditions met) - WATCH signal'
        else:
            signal = 'NEUTRAL'
            reasoning = 'SMAs are not aligned for clear signal'

        return {
            'signal': signal,
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'reasoning': reasoning
        }

    def calculate_green_candle_signal(self, hist_data, current_position=None):
        """
        Calculate Green Candle (V20) strategy signals with proper rule validation

        V20 Strategy Rules:
        1. Identify single or consecutive green candles (no red candles in between)
        2. Calculate 20% minimum movement from lowest to highest point of the sequence
        3. Define range: Lower line = lowest point, Upper line = highest point
        4. Buy when price touches lower line
        5. Sell when price touches upper line
        6. No stop-loss, hold until target hit

        Args:
            hist_data: Historical price data DataFrame with OHLC columns
            current_position: Dict indicating if currently holding position
                             {'holding': True/False, 'entry_price': float, 'range_id': str}
        """
        if len(hist_data) < 2:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'reasoning': 'Insufficient data for Green Candle analysis'
            }

        current_price = hist_data['Close'].iloc[-1]

        # Initialize position tracking
        if current_position is None:
            current_position = {'holding': False, 'entry_price': None, 'range_id': None}

        # Step 1: Identify green candles and consecutive sequences
        green_candles = hist_data['Close'] > hist_data['Open']

        # Find all green candle sequences (single or consecutive)
        valid_ranges = []
        i = 0

        while i < len(green_candles):
            if green_candles.iloc[i]:  # Found a green candle
                # Start of a sequence
                sequence_start = i
                sequence_end = i

                # Find consecutive green candles
                while (sequence_end + 1 < len(green_candles) and
                       green_candles.iloc[sequence_end + 1]):
                    sequence_end += 1

                # Get the sequence data
                sequence_indices = list(range(sequence_start, sequence_end + 1))
                sequence_data = hist_data.iloc[sequence_start:sequence_end + 1]

                # Calculate lowest and highest points of the sequence
                lowest_point = sequence_data['Low'].min()
                highest_point = sequence_data['High'].max()

                # Calculate price movement percentage
                if lowest_point > 0:
                    price_movement = (highest_point - lowest_point) / lowest_point

                    # Check if movement is >= 20%
                    if price_movement >= 0.20:
                        range_id = f"green_range_{sequence_start}_{sequence_end}"

                        # Find the exact dates where lowest and highest occurred
                        lowest_date = sequence_data[sequence_data['Low'] == lowest_point].index[0]
                        highest_date = sequence_data[sequence_data['High'] == highest_point].index[0]

                        valid_ranges.append({
                            'range_id': range_id,
                            'lower_line': lowest_point,
                            'upper_line': highest_point,
                            'start_idx': sequence_start,
                            'end_idx': sequence_end,
                            'movement_percent': price_movement * 100,
                            'candle_count': len(sequence_indices),
                            'start_date': hist_data.index[sequence_start],
                            'end_date': hist_data.index[sequence_end],
                            'lowest_date': lowest_date,
                            'highest_date': highest_date,
                            'sequence_data': sequence_data.copy()
                        })

                # Move to next candle after the sequence
                i = sequence_end + 1
            else:
                # Not a green candle, move to next
                i += 1

        # If no valid ranges found
        if not valid_ranges:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'No valid green candle sequences with 20% minimum movement found',
                'all_valid_ranges': [],
                'range_details': None,
                'position_status': {
                    'holding': current_position['holding'],
                    'entry_price': current_position['entry_price'],
                    'current_range': current_position['range_id'],
                    'profit_loss_percent': None
                }
            }

        # Get the most recent valid range for trading
        latest_range = valid_ranges[-1]
        lower_line = latest_range['lower_line']
        upper_line = latest_range['upper_line']
        range_id = latest_range['range_id']

        # Step 3 & 4: Apply V20 trading rules
        # Define tolerance for "touching" the lines (within 1%)
        lower_line_tolerance = lower_line * 1.01  # 1% above lower line
        upper_line_tolerance = upper_line * 0.99  # 1% below upper line

        # Check if we're in a new range (reset position if range changed)
        if current_position['range_id'] != range_id:
            current_position = {'holding': False, 'entry_price': None, 'range_id': range_id}

        # Determine signal based on V20 rules
        if not current_position['holding']:
            # Not holding position - look for buy signal
            if current_price <= lower_line_tolerance:
                signal = 'BUY'
                reasoning = f'Buy signal: Price ({current_price:.2f}) touched lower line ({lower_line:.2f})'
            elif lower_line < current_price < upper_line:
                signal = 'WATCH'
                reasoning = f'Price within range ({lower_line:.2f} - {upper_line:.2f}). Wait for lower line touch.'
            elif current_price >= upper_line_tolerance:
                signal = 'NEUTRAL'
                reasoning = f'Price at upper line but no position to sell. Wait for retrace to lower line.'
            else:
                signal = 'NEUTRAL'
                reasoning = 'Price outside current range. Wait for new setup.'
        else:
            # Holding position - look for sell signal
            if current_price >= upper_line_tolerance:
                signal = 'SELL'
                reasoning = f'Sell signal: Price ({current_price:.2f}) touched upper line ({upper_line:.2f})'
            elif lower_line < current_price < upper_line:
                signal = 'HOLD'
                reasoning = f'Hold position. Price moving towards target at upper line ({upper_line:.2f})'
            elif current_price <= lower_line_tolerance:
                signal = 'HOLD'
                reasoning = f'Hold position despite retrace. No stop-loss in V20 strategy.'
            else:
                signal = 'HOLD'
                reasoning = 'Hold position and wait for price to reach upper line target.'

        # Calculate potential profit/loss if holding
        profit_loss_percent = None
        if current_position['holding'] and current_position['entry_price']:
            profit_loss_percent = ((current_price - current_position['entry_price']) /
                                   current_position['entry_price']) * 100

        # Calculate target profit potential
        target_profit_percent = ((upper_line - lower_line) / lower_line) * 100

        return {
            'signal': signal,
            'current_price': current_price,
            'range_details': {
                'range_id': range_id,
                'lower_line': lower_line,
                'upper_line': upper_line,
                'movement_percent': latest_range['movement_percent'],
                'candle_count': latest_range['candle_count'],
                'start_date': latest_range['start_date'].strftime('%Y-%m-%d'),
                'end_date': latest_range['end_date'].strftime('%Y-%m-%d'),
                'lowest_date': latest_range['lowest_date'].strftime('%Y-%m-%d'),
                'highest_date': latest_range['highest_date'].strftime('%Y-%m-%d'),
                'target_profit_percent': target_profit_percent
            },
            'position_status': {
                'holding': current_position['holding'],
                'entry_price': current_position['entry_price'],
                'current_range': current_position['range_id'],
                'profit_loss_percent': profit_loss_percent
            },
            'trading_levels': {
                'buy_level': lower_line,
                'sell_level': upper_line,
                'buy_tolerance': lower_line_tolerance,
                'sell_tolerance': upper_line_tolerance
            },
            'all_valid_ranges': valid_ranges,  # For analysis purposes
            'reasoning': reasoning
        }

    def find_pivot_points(self, hist_data, window=5):
        """
        Find pivot points (swing highs and lows) for better chart visualization

        Args:
            hist_data: Historical price data DataFrame with OHLC columns
            window: Number of periods to look back and forward for pivot identification

        Returns:
            Dictionary with pivot highs and lows
        """
        if len(hist_data) < window * 2 + 1:
            return {'pivot_highs': [], 'pivot_lows': []}

        pivot_highs = []
        pivot_lows = []

        # Find pivot points
        for i in range(window, len(hist_data) - window):
            # Check for pivot high
            high_values = hist_data['High'].iloc[i - window:i + window + 1]
            if hist_data['High'].iloc[i] == high_values.max():
                pivot_highs.append({
                    'date': hist_data.index[i],
                    'price': hist_data['High'].iloc[i],
                    'index': i
                })

            # Check for pivot low
            low_values = hist_data['Low'].iloc[i - window:i + window + 1]
            if hist_data['Low'].iloc[i] == low_values.min():
                pivot_lows.append({
                    'date': hist_data.index[i],
                    'price': hist_data['Low'].iloc[i],
                    'index': i
                })

        return {
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows
        }

    def calculate_range_bound_signal(self, hist_data, min_touches=2, min_range_pct=14.0, preferred_range_pct=20.0):
        """
        Detects Range Bound zones and generates trading signals with improved pivot detection.

        Strategy Rules:
        - Support & Resistance levels are identified from recent pivot points.
        - Each level must be touched at least `min_touches` times.
        - The touches must follow a specific alternating pattern: Support → Resistance → Support → Resistance
        - The range between Support and Resistance must meet the minimum percentage requirement.
        - Signals are generated based on the current price's position within the validated range.
        """
        # 1. --- Initial Data Validation ---
        if len(hist_data) < 60:
            return {
                'signal': 'NEUTRAL',
                'reasoning': 'Insufficient data for Range Bound analysis (minimum 60 days required).'
            }

        current_price = hist_data['Close'].iloc[-1]
        # Use more data for better range detection (last 12 months)
        data = hist_data.tail(252) if len(hist_data) >= 252 else hist_data

        # 2. --- Enhanced Pivot Point Detection ---
        def find_pivot_highs(prices, window=5):
            """Find pivot highs with smaller window for better sensitivity"""
            pivots = []
            for i in range(window, len(prices) - window):
                if prices.iloc[i] == prices.iloc[i - window:i + window + 1].max():
                    pivots.append(i)
            return pivots

        def find_pivot_lows(prices, window=5):
            """Find pivot lows with smaller window for better sensitivity"""
            pivots = []
            for i in range(window, len(prices) - window):
                if prices.iloc[i] == prices.iloc[i - window:i + window + 1].min():
                    pivots.append(i)
            return pivots

        # Find resistance and support pivots with multiple window sizes
        resistance_indices = []
        support_indices = []

        # Use multiple window sizes to catch different timeframe pivots
        for window in [5, 8, 12]:
            resistance_indices.extend(find_pivot_highs(data['High'], window=window))
            support_indices.extend(find_pivot_lows(data['Low'], window=window))

        # Remove duplicates and sort
        resistance_indices = sorted(list(set(resistance_indices)))
        support_indices = sorted(list(set(support_indices)))

        if len(support_indices) < min_touches or len(resistance_indices) < min_touches:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': f'Not enough pivot points found (S:{len(support_indices)}, R:{len(resistance_indices)}). Minimum {min_touches} each required.'
            }

        # 3. --- Find Multiple Valid Range Combinations ---
        def cluster_levels(prices, indices, tolerance=0.025):
            """Cluster nearby price levels together with tighter tolerance"""
            if not indices:
                return []

            price_levels = [prices.iloc[i] for i in indices]
            clusters = []

            for price in price_levels:
                added_to_cluster = False
                for cluster in clusters:
                    if abs(price - cluster['center']) / cluster['center'] <= tolerance:
                        cluster['prices'].append(price)
                        cluster['center'] = np.mean(cluster['prices'])
                        cluster['count'] += 1
                        added_to_cluster = True
                        break

                if not added_to_cluster:
                    clusters.append({
                        'center': price,
                        'prices': [price],
                        'count': 1
                    })

            # Sort by count (most touched levels first)
            clusters.sort(key=lambda x: x['count'], reverse=True)
            return clusters

        # Cluster support and resistance levels
        support_clusters = cluster_levels(data['Low'], support_indices)
        resistance_clusters = cluster_levels(data['High'], resistance_indices)

        if not support_clusters or not resistance_clusters:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'Could not identify clustered support/resistance levels.'
            }

        # 4. --- Find Best Range Combination ---
        best_range = None
        best_score = 0

        # Try different combinations of support and resistance levels
        for s_cluster in support_clusters[:3]:  # Top 3 support clusters
            for r_cluster in resistance_clusters[:3]:  # Top 3 resistance clusters
                support_level = s_cluster['center']
                resistance_level = r_cluster['center']

                # Ensure proper order
                if resistance_level <= support_level:
                    continue

                # Calculate range percentage
                range_percent = ((resistance_level - support_level) / support_level) * 100

                # Skip if range is too small
                if range_percent < min_range_pct:
                    continue

                # Validate specific alternating pattern (Support → Resistance → Support → Resistance)
                alternating_touches = self.validate_specific_alternating_pattern(data, support_level, resistance_level)

                support_touch_count = sum(1 for touch in alternating_touches if touch[0] == 'Support')
                resistance_touch_count = sum(1 for touch in alternating_touches if touch[0] == 'Resistance')

                # Check if we have enough touches and valid pattern
                if support_touch_count < min_touches or resistance_touch_count < min_touches:
                    continue

                # Calculate score (higher is better)
                score = (
                        range_percent * 0.4 +  # Prefer larger ranges
                        (support_touch_count + resistance_touch_count) * 10 +  # More touches = better
                        (s_cluster['count'] + r_cluster['count']) * 5  # More clustered pivots = better
                )

                if score > best_score:
                    best_score = score
                    best_range = {
                        'support_level': support_level,
                        'resistance_level': resistance_level,
                        'range_percent': range_percent,
                        'support_touches': support_touch_count,
                        'resistance_touches': resistance_touch_count,
                        'alternating_touches': alternating_touches,
                        'support_cluster_strength': s_cluster['count'],
                        'resistance_cluster_strength': r_cluster['count']
                    }

        if not best_range:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': f'No valid range found meeting criteria (min {min_range_pct}% range, {min_touches} touches each level, Support→Resistance→Support→Resistance pattern).'
            }

        # Extract best range details
        support_level = best_range['support_level']
        resistance_level = best_range['resistance_level']
        range_percent = best_range['range_percent']
        support_touch_count = best_range['support_touches']
        resistance_touch_count = best_range['resistance_touches']
        alternating_touches = best_range['alternating_touches']

        range_quality = "STRONG" if range_percent >= preferred_range_pct else "ACCEPTABLE"

        # 5. --- Generate Trading Signal ---
        # Define tolerance zones for entry signals
        support_tolerance = support_level * 0.02  # 2% tolerance
        resistance_tolerance = resistance_level * 0.02  # 2% tolerance

        buy_zone_upper = support_level + support_tolerance
        sell_zone_lower = resistance_level - resistance_tolerance

        # Determine signal
        if current_price <= buy_zone_upper:
            signal = 'BUY'
            reasoning = f'Price near Support Level ({support_level:.2f}). Buy opportunity.'
        elif current_price >= sell_zone_lower:
            signal = 'SELL'
            reasoning = f'Price near Resistance Level ({resistance_level:.2f}). Sell opportunity.'
        else:
            signal = 'WATCH'
            reasoning = f'Price within range ({support_level:.2f} - {resistance_level:.2f}). Watch for levels.'

        # 6. --- Return Comprehensive Result ---
        return {
            'signal': signal,
            'current_price': round(current_price, 2),
            'support_level': round(support_level, 2),
            'resistance_level': round(resistance_level, 2),
            'range_percent': round(range_percent, 2),
            'range_quality': range_quality,
            'support_touches': support_touch_count,
            'resistance_touches': resistance_touch_count,
            'alternating_pattern': ' → '.join([touch[0] for touch in alternating_touches]),
            'recent_touches': [
                f"{touch[0]}: {touch[1].strftime('%Y-%m-%d')} at {touch[2]:.2f}"
                for touch in alternating_touches[-6:]  # Show last 6 touches
            ],
            'support_cluster_strength': best_range['support_cluster_strength'],
            'resistance_cluster_strength': best_range['resistance_cluster_strength'],
            'trading_levels': {
                'buy_zone_upper': round(buy_zone_upper, 2),
                'sell_zone_lower': round(sell_zone_lower, 2),
                'support_tolerance': round(support_tolerance, 2),
                'resistance_tolerance': round(resistance_tolerance, 2)
            },
            'reasoning': reasoning,
            'pattern_validation': f'Valid Support→Resistance→Support→Resistance pattern with {len(alternating_touches)} total touches'
        }

    def validate_specific_alternating_pattern(self, data, support_level, resistance_level, tolerance=0.03):
        """
        Check for specific alternating pattern: Support → Resistance → Support → Resistance
        Only accepts patterns that start with Support and follow the exact sequence

        Threshold Logic:
        - Support: Price can touch AT or BELOW the support level (support_level * (1 + tolerance) and below)
        - Resistance: Price can touch AT or ABOVE the resistance level (resistance_level * (1 - tolerance) and above)
        """
        touches = []

        # Define threshold levels (allowing plotting outside the range)
        support_threshold = support_level * (1 + tolerance)  # Support can be touched at or below this level
        resistance_threshold = resistance_level * (1 - tolerance)  # Resistance can be touched at or above this level

        for i in range(len(data)):
            low = data['Low'].iloc[i]
            high = data['High'].iloc[i]

            # Check if price touched support (at or below support threshold)
            if low <= support_threshold:
                touches.append(('Support', data.index[i], low))

            # Check if price touched resistance (at or above resistance threshold)
            if high >= resistance_threshold:
                touches.append(('Resistance', data.index[i], high))

        # Remove consecutive touches of the same type
        filtered_touches = []
        last_type = None

        for touch in touches:
            if touch[0] != last_type:
                filtered_touches.append(touch)
                last_type = touch[0]

        # Validate the specific pattern: Must start with Support
        if not filtered_touches or filtered_touches[0][0] != 'Support':
            return []

        # Check if pattern follows Support → Resistance → Support → Resistance sequence
        valid_pattern = []
        expected_sequence = ['Support', 'Resistance', 'Support', 'Resistance']
        sequence_index = 0

        for touch in filtered_touches:
            if sequence_index < len(expected_sequence) and touch[0] == expected_sequence[sequence_index]:
                valid_pattern.append(touch)
                sequence_index += 1

                # If we've completed one full cycle, reset to continue the pattern
                if sequence_index == len(expected_sequence):
                    sequence_index = 0  # Reset to start new cycle with Support
            else:
                # If the pattern is broken, check if we can start a new valid sequence
                if touch[0] == 'Support':
                    # Start new sequence from Support
                    valid_pattern.append(touch)
                    sequence_index = 1  # Next should be Resistance
                else:
                    # Pattern is broken and we can't start fresh, skip this touch
                    continue

        # Filter to ensure we have at least complete alternating touches
        # and the pattern maintains the Support → Resistance → Support → Resistance flow
        final_pattern = []
        for i, touch in enumerate(valid_pattern):
            if i == 0:
                # First touch must be Support
                if touch[0] == 'Support':
                    final_pattern.append(touch)
            else:
                # Subsequent touches must alternate properly
                prev_touch = final_pattern[-1] if final_pattern else None
                if prev_touch and touch[0] != prev_touch[0]:
                    # Ensure we maintain the correct sequence
                    if (prev_touch[0] == 'Support' and touch[0] == 'Resistance') or \
                            (prev_touch[0] == 'Resistance' and touch[0] == 'Support'):
                        final_pattern.append(touch)

        return final_pattern

    def calculate_rhs_signal(self, hist_data):
        """
        Detect Reverse Head and Shoulders pattern (RHS) and generate signals.
        CORRECTED VERSION with improved pattern detection for real market conditions.
        """
        if len(hist_data) < 100:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'reasoning': 'Insufficient data for RHS pattern analysis'
            }

        current_price = hist_data['Close'].iloc[-1]

        # Use more data for better pattern detection
        lookback_period = min(300, len(hist_data))
        recent_data = hist_data.tail(lookback_period).copy()

        # Step 1: Find pivot points using pure numpy/pandas approach
        # Find local minima (lows) and maxima (highs)
        low_order = 10  # Look for lows over 10-period window
        high_order = 8  # Look for highs over 8-period window

        # Find local minima manually
        low_indices = []
        for i in range(low_order, len(recent_data) - low_order):
            current_low = recent_data['Low'].iloc[i]
            left_window = recent_data['Low'].iloc[i - low_order:i]
            right_window = recent_data['Low'].iloc[i + 1:i + low_order + 1]

            if (current_low <= left_window.min() and
                    current_low <= right_window.min()):
                low_indices.append(i)

        # Find local maxima manually
        high_indices = []
        for i in range(high_order, len(recent_data) - high_order):
            current_high = recent_data['High'].iloc[i]
            left_window = recent_data['High'].iloc[i - high_order:i]
            right_window = recent_data['High'].iloc[i + 1:i + high_order + 1]

            if (current_high >= left_window.max() and
                    current_high >= right_window.max()):
                high_indices.append(i)

        # Create pivot points list
        pivot_lows = []
        pivot_highs = []

        for idx in low_indices:
            pivot_lows.append({
                'index': idx,
                'date': recent_data.index[idx],
                'price': recent_data['Low'].iloc[idx]
            })

        for idx in high_indices:
            pivot_highs.append({
                'index': idx,
                'date': recent_data.index[idx],
                'price': recent_data['High'].iloc[idx]
            })

        # Sort by index
        pivot_lows.sort(key=lambda x: x['index'])
        pivot_highs.sort(key=lambda x: x['index'])

        if len(pivot_lows) < 3:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'Insufficient pivot lows for RHS pattern detection'
            }

        # Step 2: Look for RHS pattern in recent pivots
        # Check multiple combinations of the last several lows
        best_pattern = None
        best_score = 0

        # Look at last 6 lows to find the best RHS pattern
        recent_lows = pivot_lows[-6:] if len(pivot_lows) >= 6 else pivot_lows

        for i in range(len(recent_lows) - 2):
            for j in range(i + 1, len(recent_lows) - 1):
                for k in range(j + 1, len(recent_lows)):
                    left_shoulder = recent_lows[i]
                    head = recent_lows[j]
                    right_shoulder = recent_lows[k]

                    # Check RHS pattern rules with more flexibility
                    pattern_score = self._evaluate_rhs_pattern(
                        left_shoulder, head, right_shoulder, pivot_highs
                    )

                    if pattern_score > best_score:
                        best_score = pattern_score
                        best_pattern = {
                            'left_shoulder': left_shoulder,
                            'head': head,
                            'right_shoulder': right_shoulder,
                            'score': pattern_score
                        }

        if best_pattern is None or best_score < 0.5:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'No valid Reverse Head and Shoulders pattern found'
            }

        # Step 3: Find neckline points
        left_shoulder = best_pattern['left_shoulder']
        head = best_pattern['head']
        right_shoulder = best_pattern['right_shoulder']

        # Find highs between the pattern points
        left_neckline = self._find_neckline_high(pivot_highs, left_shoulder['index'], head['index'])
        right_neckline = self._find_neckline_high(pivot_highs, head['index'], right_shoulder['index'])
        current_neckline = self._find_neckline_high(pivot_highs, right_shoulder['index'], len(recent_data))

        # Calculate neckline level with 3% threshold as requested
        neckline_points = []
        if left_neckline:
            neckline_points.append(left_neckline['price'])
        if right_neckline:
            neckline_points.append(right_neckline['price'])
        if current_neckline:
            neckline_points.append(current_neckline['price'])

        if not neckline_points:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'Cannot establish neckline for RHS pattern'
            }

        neckline = np.mean(neckline_points)

        # Step 4: Base formation analysis (right shoulder area)
        rs_start_idx = right_shoulder['index']
        base_data = recent_data.iloc[rs_start_idx:]

        if len(base_data) < 5:
            base_formed = False
            base_high = right_shoulder['price']
            base_low = right_shoulder['price']
        else:
            base_window = min(20, len(base_data))
            base_analysis_data = base_data.tail(base_window)
            base_high = base_analysis_data['High'].max()
            base_low = base_analysis_data['Low'].min()
            base_range = (base_high - base_low) / base_low
            base_formed = base_range < 0.08  # 8% range for base formation

        # Step 5: Breakout analysis
        breakout_above_base = current_price > base_high
        breakout_above_neckline = current_price > neckline

        # Check if latest candle is green (bullish)
        latest_candle = recent_data.iloc[-1]
        is_green = latest_candle['Close'] > latest_candle['Open']

        # Step 6: Target calculation
        depth = neckline - head['price']
        target_price = neckline + depth

        # Consider lifetime high
        lifetime_high = hist_data['High'].max()
        final_target = max(target_price, lifetime_high)

        # Step 7: Signal generation
        if current_price >= final_target:
            signal = 'SELL'
            status = 'Target achieved - Time to sell'
        elif (breakout_above_neckline and is_green and breakout_above_base):
            signal = 'BUY'
            status = 'RHS pattern confirmed - Breakout above neckline with green candle'
        elif breakout_above_neckline and breakout_above_base:
            signal = 'WATCH'
            status = 'Breakout confirmed - Waiting for green candle confirmation'
        elif breakout_above_neckline:
            signal = 'WATCH'
            status = 'Neckline breakout in progress'
        elif base_formed and current_price > (neckline * 0.97):  # Within 3% of neckline
            signal = 'WATCH'
            status = 'Base formation complete - Near neckline breakout'
        elif base_formed:
            signal = 'WATCH'
            status = 'Base formation complete - Waiting for neckline approach'
        else:
            signal = 'NEUTRAL'
            status = 'RHS pattern identified - Base consolidation in progress'

        return {
            'signal': signal,
            'current_price': current_price,
            'status': status,
            'pattern_points': {
                'left_shoulder': {
                    'date': left_shoulder['date'].strftime('%Y-%m-%d'),
                    'price': left_shoulder['price']
                },
                'head': {
                    'date': head['date'].strftime('%Y-%m-%d'),
                    'price': head['price']
                },
                'right_shoulder': {
                    'date': right_shoulder['date'].strftime('%Y-%m-%d'),
                    'price': right_shoulder['price']
                },
                'left_neckline': {
                    'date': left_neckline['date'].strftime('%Y-%m-%d'),
                    'price': left_neckline['price']
                } if left_neckline else None,
                'right_neckline': {
                    'date': right_neckline['date'].strftime('%Y-%m-%d'),
                    'price': right_neckline['price']
                } if right_neckline else None,
                'current_neckline': {
                    'date': current_neckline['date'].strftime('%Y-%m-%d'),
                    'price': current_neckline['price']
                } if current_neckline else None,
                'neckline': neckline,
                'target_price': target_price,
                'final_target': final_target,
                'lifetime_high': lifetime_high,
                'depth': depth
            },
            'base_analysis': {
                'base_formed': base_formed,
                'base_high': base_high,
                'base_low': base_low,
                'base_range_percent': ((base_high - base_low) / base_low * 100) if base_low > 0 else 0,
                'breakout_above_base': breakout_above_base,
                'breakout_above_neckline': breakout_above_neckline,
                'is_green_candle': is_green
            },
            'pattern_quality': {
                'pattern_score': best_score,
                'neckline_points_count': len(neckline_points),
                'distance_to_neckline': ((current_price - neckline) / neckline * 100) if neckline > 0 else 0
            },
            'reasoning': f'Reverse Head and Shoulders pattern detected with score {best_score:.2f}. {status}'
        }

    def _evaluate_rhs_pattern(self, left_shoulder, head, right_shoulder, pivot_highs):
        """
        Evaluate how well the three lows form an RHS pattern.
        Returns a score between 0 and 1.
        """
        score = 0.0

        # Rule 1: Head must be lower than both shoulders (essential)
        if not (head['price'] < left_shoulder['price'] and head['price'] < right_shoulder['price']):
            return 0.0

        score += 0.3  # Base score for valid head position

        # Rule 2: Right shoulder cannot be deeper than head
        if right_shoulder['price'] <= head['price']:
            return 0.0

        score += 0.2  # Additional score for valid right shoulder

        # Rule 3: Shoulders should be roughly similar (within 15% tolerance for real market)
        shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
        avg_shoulder_price = (left_shoulder['price'] + right_shoulder['price']) / 2
        shoulder_similarity = 1 - min(shoulder_diff / avg_shoulder_price, 0.15) / 0.15
        score += 0.2 * shoulder_similarity

        # Rule 4: Time spacing should be reasonable
        total_time_span = right_shoulder['index'] - left_shoulder['index']
        if total_time_span > 10:  # At least 10 periods for pattern formation
            score += 0.1

        # Rule 5: Pattern should be relatively recent (last 60% of data)
        data_length = len(pivot_highs) if pivot_highs else 100
        if left_shoulder['index'] > data_length * 0.4:
            score += 0.1

        # Rule 6: Depth significance (head should be meaningfully lower)
        head_depth = min(left_shoulder['price'], right_shoulder['price']) - head['price']
        avg_price = (left_shoulder['price'] + right_shoulder['price']) / 2
        if head_depth / avg_price > 0.05:  # At least 5% depth
            score += 0.1

        return min(score, 1.0)

    def _find_neckline_high(self, pivot_highs, start_idx, end_idx):
        """
        Find the highest point between two indices for neckline formation.
        """
        candidates = [h for h in pivot_highs if start_idx < h['index'] < end_idx]
        if not candidates:
            return None

        # Return the highest point in the range
        return max(candidates, key=lambda x: x['price'])

    def calculate_cup_handle_signal(self, hist_data):
        """
        Enhanced Cup with Handle Pattern Detection following exact pattern rules.
        """
        if len(hist_data) < 100:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'reasoning': 'Insufficient data for Cup Handle pattern analysis'
            }

        current_price = hist_data['Close'].iloc[-1]

        # Use recent data for analysis
        recent_data = hist_data.tail(200)  # Look at last 200 periods

        # Find the pattern using a more systematic approach
        pattern_result = self._detect_cup_handle_systematic(recent_data, current_price)

        if pattern_result:
            return pattern_result

        return {
            'signal': 'NEUTRAL',
            'current_price': current_price,
            'reasoning': 'No valid Cup with Handle pattern identified.'
        }

    def _detect_cup_handle_systematic(self, data, current_price):
        """
        Systematic approach to detect cup and handle pattern.
        """
        # Step 1: Find potential cup formations
        cup_candidates = self._find_cup_candidates(data)

        for cup in cup_candidates:
            # Step 2: Validate the cup formation
            if self._validate_cup_systematic(cup, data):
                # Step 3: Look for handle after the cup
                handle_result = self._find_handle_after_cup(cup, data)
                if handle_result:
                    # Step 4: Validate complete pattern
                    return self._generate_signal_systematic(cup, handle_result, data, current_price)

        return None

    def _find_cup_candidates(self, data):
        """
        Find potential cup formations by looking for U-shaped or V-shaped patterns.
        """
        cup_candidates = []

        # Look for significant highs that could be cup rims
        for i in range(20, len(data) - 50):  # Leave room for handle
            # Potential left rim
            left_high_idx = i
            left_high_price = data['High'].iloc[i]

            # Look for a significant decline from this high
            for j in range(i + 10, min(i + 80, len(data) - 20)):
                cup_low_idx = j
                cup_low_price = data['Low'].iloc[j]

                # Check if decline is significant enough (at least 15%)
                decline_percent = (left_high_price - cup_low_price) / left_high_price
                if decline_percent < 0.15:
                    continue

                # Look for recovery forming right rim
                for k in range(j + 5, min(j + 50, len(data))):
                    right_high_idx = k
                    right_high_price = data['High'].iloc[k]

                    # Check if recovery is substantial (at least 80% of decline)
                    recovery_percent = (right_high_price - cup_low_price) / (left_high_price - cup_low_price)
                    if recovery_percent < 0.80:
                        continue

                    # Check if the rims are roughly at the same level (within 5%)
                    rim_difference = abs(left_high_price - right_high_price) / max(left_high_price, right_high_price)
                    if rim_difference > 0.05:
                        continue

                    cup_candidates.append({
                        'left_rim_idx': left_high_idx,
                        'left_rim_price': left_high_price,
                        'cup_low_idx': cup_low_idx,
                        'cup_low_price': cup_low_price,
                        'right_rim_idx': right_high_idx,
                        'right_rim_price': right_high_price,
                        'decline_percent': decline_percent,
                        'recovery_percent': recovery_percent,
                        'rim_difference': rim_difference
                    })

        # Sort by quality (deeper cups with better recovery)
        cup_candidates.sort(key=lambda x: x['decline_percent'] * x['recovery_percent'], reverse=True)
        return cup_candidates

    def _validate_cup_systematic(self, cup, data):
        """
        Validate cup formation with systematic checks.
        """
        # Check minimum cup duration
        cup_duration = cup['right_rim_idx'] - cup['left_rim_idx']
        if cup_duration < 30:  # Cup should span at least 30 periods
            return False

        # Check that the cup low is the lowest point in the cup period
        cup_section = data.iloc[cup['left_rim_idx']:cup['right_rim_idx'] + 1]
        actual_low = cup_section['Low'].min()
        if abs(actual_low - cup['cup_low_price']) > 0.01:  # Allow small tolerance
            return False

        # Check U-shape or V-shape (gradual decline and recovery)
        left_section = data.iloc[cup['left_rim_idx']:cup['cup_low_idx']]
        right_section = data.iloc[cup['cup_low_idx']:cup['right_rim_idx'] + 1]

        # Both sections should have reasonable length
        if len(left_section) < 10 or len(right_section) < 10:
            return False

        return True

    def _find_handle_after_cup(self, cup, data):
        """
        Find handle formation after the cup.
        """
        # Handle should start after the right rim
        handle_start_idx = cup['right_rim_idx']

        # Look for handle in the next 30 periods
        handle_end_idx = min(handle_start_idx + 30, len(data) - 1)

        if handle_end_idx <= handle_start_idx + 5:
            return None

        handle_section = data.iloc[handle_start_idx:handle_end_idx + 1]

        # Find handle low
        handle_low_price = handle_section['Low'].min()
        handle_low_idx = handle_section['Low'].idxmin()
        handle_low_idx_relative = handle_section.index.get_loc(handle_low_idx)

        # Handle depth should not exceed 50% of cup depth
        cup_depth = cup['left_rim_price'] - cup['cup_low_price']
        handle_depth = cup['right_rim_price'] - handle_low_price

        if handle_depth > 0.5 * cup_depth:
            return None

        # Handle low should be above cup low
        if handle_low_price <= cup['cup_low_price']:
            return None

        # Check for consolidation/base formation in recent periods
        recent_periods = 10
        if len(handle_section) >= recent_periods:
            recent_section = handle_section.tail(recent_periods)
            base_high = recent_section['High'].max()
            base_low = recent_section['Low'].min()
            base_range = (base_high - base_low) / base_low if base_low > 0 else 0
            base_formed = base_range < 0.05  # Less than 5% range
        else:
            base_formed = False
            base_high = handle_section['High'].max()
            base_low = handle_section['Low'].min()

        return {
            'handle_start_idx': handle_start_idx,
            'handle_low_idx': handle_low_idx_relative + handle_start_idx,
            'handle_low_price': handle_low_price,
            'handle_depth': handle_depth,
            'base_formed': base_formed,
            'base_high': base_high,
            'base_low': base_low
        }

    def _generate_signal_systematic(self, cup, handle_result, data, current_price):
        """
        Generate trading signal based on the validated pattern.
        """
        # Calculate neckline (average of cup rims)
        neckline = (cup['left_rim_price'] + cup['right_rim_price']) / 2

        # Calculate target price
        cup_depth = cup['left_rim_price'] - cup['cup_low_price']
        target_price = neckline + cup_depth

        # Get latest candle info
        latest_candle = data.iloc[-1]
        is_green_candle = latest_candle['Close'] > latest_candle['Open']

        # Check breakout conditions
        breakout_above_base = current_price > handle_result['base_high']
        breakout_above_neckline = current_price > neckline

        # Determine signal
        status = "Pattern identified. Watching for breakout."
        signal = "NEUTRAL"

        if breakout_above_neckline and breakout_above_base and is_green_candle:
            signal = 'BUY'
            status = 'Pattern confirmed: Breakout above neckline and base with a green candle.'
        elif breakout_above_neckline and breakout_above_base:
            signal = 'WATCH'
            status = 'Breakout above neckline occurred. Awaiting a green candle for confirmation.'
        elif handle_result['base_formed']:
            signal = 'WATCH'
            status = 'Handle base has formed. Awaiting breakout above base and neckline.'
        elif breakout_above_neckline:
            signal = 'WATCH'
            status = 'Breakout above neckline. Watching for base breakout.'

        if current_price >= target_price:
            signal = 'SELL'
            status = 'Target price has been reached. Consider selling.'

        return {
            'signal': signal,
            'current_price': current_price,
            'status': status,
            'reasoning': f'Cup with Handle pattern identified. {status}',
            'pattern_points': {
                'cup_start': {
                    'date': data.index[cup['left_rim_idx']].strftime('%Y-%m-%d'),
                    'price': cup['left_rim_price']
                },
                'cup_low': {
                    'date': data.index[cup['cup_low_idx']].strftime('%Y-%m-%d'),
                    'price': cup['cup_low_price']
                },
                'cup_end': {
                    'date': data.index[cup['right_rim_idx']].strftime('%Y-%m-%d'),
                    'price': cup['right_rim_price']
                },
                'handle_low': {
                    'date': data.index[handle_result['handle_low_idx']].strftime('%Y-%m-%d'),
                    'price': handle_result['handle_low_price']
                },
                'neckline': neckline,
                'target_price': target_price
            },
            'pattern_metrics': {
                'cup_depth_percent': cup['decline_percent'] * 100,
                'handle_to_cup_ratio': handle_result['handle_depth'] / cup_depth if cup_depth > 0 else 0,
                'cup_duration_days': cup['right_rim_idx'] - cup['left_rim_idx']
            }
        }
