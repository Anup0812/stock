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
        1. Identify consecutive green candles (any length, minimum 1)
        2. Calculate 20% minimum movement from lowest to highest point
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

        # Step 1: Identify consecutive green candle sequences
        green_candles = hist_data['Close'] > hist_data['Open']

        # Find all consecutive green candle sequences
        valid_ranges = []
        current_sequence = []

        for i in range(len(green_candles)):
            if green_candles.iloc[i]:  # Green candle
                current_sequence.append(i)
            else:  # Red candle - end current sequence
                if current_sequence:  # Process completed sequence
                    # Step 2: Validate 20% minimum movement requirement
                    sequence_data = hist_data.iloc[current_sequence[0]:current_sequence[-1] + 1]
                    lowest_point = sequence_data['Low'].min()
                    highest_point = sequence_data['High'].max()

                    # Calculate price movement percentage
                    price_movement = (highest_point - lowest_point) / lowest_point

                    if price_movement >= 0.20:  # 20% minimum movement
                        range_id = f"green_range_{current_sequence[0]}_{current_sequence[-1]}"
                        valid_ranges.append({
                            'range_id': range_id,
                            'lower_line': lowest_point,
                            'upper_line': highest_point,
                            'start_idx': current_sequence[0],
                            'end_idx': current_sequence[-1],
                            'movement_percent': price_movement * 100,
                            'candle_count': len(current_sequence),
                            'start_date': hist_data.index[current_sequence[0]],
                            'end_date': hist_data.index[current_sequence[-1]]
                        })

                    current_sequence = []  # Reset for next sequence

        # Check final sequence (if data ends with green candles)
        if current_sequence:
            sequence_data = hist_data.iloc[current_sequence[0]:current_sequence[-1] + 1]
            lowest_point = sequence_data['Low'].min()
            highest_point = sequence_data['High'].max()
            price_movement = (highest_point - lowest_point) / lowest_point

            if price_movement >= 0.20:
                range_id = f"green_range_{current_sequence[0]}_{current_sequence[-1]}"
                valid_ranges.append({
                    'range_id': range_id,
                    'lower_line': lowest_point,
                    'upper_line': highest_point,
                    'start_idx': current_sequence[0],
                    'end_idx': current_sequence[-1],
                    'movement_percent': price_movement * 100,
                    'candle_count': len(current_sequence),
                    'start_date': hist_data.index[current_sequence[0]],
                    'end_date': hist_data.index[current_sequence[-1]]
                })

        # If no valid ranges found
        if not valid_ranges:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'No valid green candle ranges with 20% minimum movement found'
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
                reasoning = f'Buy signal: Price touched lower line ({lower_line:.2f})'
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
                reasoning = f'Sell signal: Price touched upper line ({upper_line:.2f})'
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

    def calculate_range_bound_signal(self, hist_data):
        """
        Detect Range Bound zones and generate trading signals as per strategy rules:
        - Support & Resistance must be touched at least twice (with alternating pattern).
        - Range must be at least 15%.
        - Signals:
          - BUY near Support
          - SELL near Resistance
          - WATCH inside range
        Returns:
            Dict containing signals, support/resistance levels, and touch points for plotting.
        """
        if len(hist_data) < 60:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'reasoning': 'Insufficient data for Range Bound analysis'
            }

        current_price = hist_data['Close'].iloc[-1]
        recent_data = hist_data.tail(120)

        # Step 1: Find pivot points (highs & lows)
        pivot_points = []
        window = 5
        for i in range(window, len(recent_data) - window):
            low = recent_data['Low'].iloc[i]
            high = recent_data['High'].iloc[i]

            if (low <= recent_data['Low'].iloc[i - window:i].min() and
                    low <= recent_data['Low'].iloc[i + 1:i + window + 1].min()):
                pivot_points.append({
                    'type': 'support',
                    'date': recent_data.index[i],
                    'price': low,
                    'index': i
                })
            if (high >= recent_data['High'].iloc[i - window:i].max() and
                    high >= recent_data['High'].iloc[i + 1:i + window + 1].max()):
                pivot_points.append({
                    'type': 'resistance',
                    'date': recent_data.index[i],
                    'price': high,
                    'index': i
                })

        pivot_points.sort(key=lambda x: x['index'])

        if len(pivot_points) < 4:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'Insufficient pivot points for Range Bound analysis'
            }

        # Step 2: Group similar levels (within 5% tolerance)
        def cluster_levels(points, tolerance=0.05):
            clusters = []
            for point in points:
                added = False
                for cluster in clusters:
                    avg_price = sum(p['price'] for p in cluster) / len(cluster)
                    if abs(point['price'] - avg_price) / avg_price <= tolerance:
                        cluster.append(point)
                        added = True
                        break
                if not added:
                    clusters.append([point])
            return clusters

        supports = cluster_levels([p for p in pivot_points if p['type'] == 'support'])
        resistances = cluster_levels([p for p in pivot_points if p['type'] == 'resistance'])

        valid_supports = [c for c in supports if len(c) >= 2]
        valid_resistances = [c for c in resistances if len(c) >= 2]

        if not valid_supports or not valid_resistances:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'No valid support/resistance levels with at least 2 touches found'
            }

        # Step 3: Select best support/resistance (most touches, latest)
        def score_cluster(cluster):
            touches = len(cluster)
            latest_touch = max(p['index'] for p in cluster)
            return touches * 10 + latest_touch

        best_support = max(valid_supports, key=score_cluster)
        best_resistance = max(valid_resistances, key=score_cluster)

        support_level = sum(p['price'] for p in best_support) / len(best_support)
        resistance_level = sum(p['price'] for p in best_resistance) / len(best_resistance)

        if resistance_level <= support_level:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'Invalid range: resistance not above support'
            }

        # Step 4: Check range size
        range_percent = ((resistance_level - support_level) / support_level) * 100
        if range_percent < 15:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': f'Range too narrow ({range_percent:.1f}%). Minimum 15% required.'
            }

        # Step 5: Check zig-zag (alternating pattern)
        all_touches = best_support + best_resistance
        all_touches.sort(key=lambda x: x['index'])

        alternating = True
        for i in range(1, len(all_touches)):
            if all_touches[i]['type'] == all_touches[i - 1]['type']:
                if all_touches[i]['index'] - all_touches[i - 1]['index'] < 10:
                    alternating = False
                    break

        if not alternating:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'No alternating pattern between support and resistance'
            }

        # Step 6: Signal generation (with 3% tolerance)
        support_tolerance = support_level * 1.03
        resistance_tolerance = resistance_level * 0.97

        if current_price <= support_tolerance:
            signal = 'BUY'
            reasoning = f'Price near support level ({support_level:.2f})'
        elif current_price >= resistance_tolerance:
            signal = 'SELL'
            reasoning = f'Price near resistance level ({resistance_level:.2f})'
        else:
            if abs(current_price - support_level) < abs(current_price - resistance_level):
                signal = 'WATCH'
                reasoning = f'Price closer to support. Watch for bounce near {support_level:.2f}'
            else:
                signal = 'WATCH'
                reasoning = f'Price closer to resistance. Watch for rejection near {resistance_level:.2f}'

        return {
            'signal': signal,
            'current_price': current_price,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'range_percent': range_percent,
            'support_touches': len(best_support),
            'resistance_touches': len(best_resistance),
            'alternating_pattern': alternating,
            'all_touches': all_touches,
            'trading_levels': {
                'buy_level': support_level,
                'sell_level': resistance_level,
                'buy_tolerance': support_tolerance,
                'sell_tolerance': resistance_tolerance
            },
            'reasoning': reasoning
        }

    def calculate_rhs_signal(self, hist_data):
        """
        Detect Reverse Head and Shoulders pattern (RHS) and generate signals as per rules.
        MODIFIED VERSION with fixes applied
        """
        if len(hist_data) < 100:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'reasoning': 'Insufficient data for RHS pattern analysis'
            }

        current_price = hist_data['Close'].iloc[-1]
        recent_data = hist_data.tail(200)

        # Step 1: Find pivot points (highs & lows)
        pivot_points = []
        window = 10
        for i in range(window, len(recent_data) - window):
            low = recent_data['Low'].iloc[i]
            high = recent_data['High'].iloc[i]
            if (low <= recent_data['Low'].iloc[i - window:i].min() and
                    low <= recent_data['Low'].iloc[i + 1:i + window + 1].min()):
                pivot_points.append({
                    'type': 'low',
                    'date': recent_data.index[i],
                    'price': low,
                    'index': i
                })
            if (high >= recent_data['High'].iloc[i - window:i].max() and
                    high >= recent_data['High'].iloc[i + 1:i + window + 1].max()):
                pivot_points.append({
                    'type': 'high',
                    'date': recent_data.index[i],
                    'price': high,
                    'index': i
                })

        pivot_points.sort(key=lambda x: x['index'])

        # Step 2: Identify RHS pattern sequence: H-L-H-L-H-L-H
        for i in range(6, len(pivot_points)):
            pattern = pivot_points[i - 6:i + 1]
            types = [p['type'] for p in pattern]
            if types == ['high', 'low', 'high', 'low', 'high', 'low', 'high']:
                # Assign pattern points
                ls_start, ls_low, ls_end, head_low, head_end, rs_low, rs_end = pattern

                # Step 3: Validate RHS shape
                # Head must be the deepest point
                if not (head_low['price'] < ls_low['price'] and head_low['price'] < rs_low['price']):
                    continue

                # FIX 1: CORRECTED - Right shoulder must be HIGHER than head (not deeper)
                # ORIGINAL: if rs_low['price'] < head_low['price']: continue
                if rs_low['price'] <= head_low['price']:  # Right shoulder cannot be deeper than head
                    continue

                # FIX 2: CORRECTED - Proper neckline calculation (horizontal line through three peaks)
                # ORIGINAL: neckline = min(ls_end['price'], head_end['price'])
                left_peak = ls_start['price']
                head_peak = head_end['price']
                right_peak = rs_end['price']

                # Calculate average neckline and validate it's horizontal
                neckline = (left_peak + head_peak + right_peak) / 3

                # FIX 3: NEW - Validate neckline is approximately horizontal (within 3% tolerance)
                peak_prices = [left_peak, head_peak, right_peak]
                max_deviation = max(abs(p - neckline) for p in peak_prices)
                if max_deviation / neckline > 0.03:  # 3% tolerance for horizontal line
                    continue

                # ORIGINAL neckline difference check - keep for compatibility but use new neckline
                # neckline_diff = abs(ls_end['price'] - head_end['price'])
                # if neckline_diff / neckline > 0.05:
                #     continue

                # Step 5: Base formation at right shoulder
                rs_data = recent_data.iloc[rs_low['index']:]
                if len(rs_data) < 10:
                    continue
                base_data = rs_data.tail(min(15, len(rs_data)))
                base_high = base_data['High'].max()
                base_low = base_data['Low'].min()
                base_range = (base_high - base_low) / base_low
                base_formed = base_range < 0.05

                # Step 6: Breakout checks
                breakout_above_base = current_price > base_high
                breakout_above_neckline = current_price > neckline
                latest_candle = recent_data.iloc[-1]
                is_green = latest_candle['Close'] > latest_candle['Open']

                # Step 7: Target & Sell Level
                depth = neckline - head_low['price']
                target_price = neckline + depth
                lifetime_high = hist_data['High'].max()
                final_target = max(target_price, lifetime_high)

                # Step 8: Generate signal
                if current_price >= final_target:
                    signal = 'SELL'
                    status = 'Target reached, Sell as per strategy'
                elif base_formed and breakout_above_base and breakout_above_neckline and is_green:
                    signal = 'BUY'
                    status = 'RHS confirmed with breakout and green candle'
                elif base_formed and breakout_above_neckline:
                    signal = 'WATCH'
                    status = 'Breakout above neckline, waiting for green candle confirmation'
                elif base_formed:
                    signal = 'WATCH'
                    status = 'Base formed at right shoulder, watching for breakout'
                else:
                    signal = 'NEUTRAL'
                    status = 'Pattern identified but base not yet formed'

                return {
                    'signal': signal,
                    'current_price': current_price,
                    'status': status,
                    'pattern_points': {
                        'left_shoulder_start': {'date': ls_start['date'].strftime('%Y-%m-%d'),
                                                'price': ls_start['price']},
                        'left_shoulder_low': {'date': ls_low['date'].strftime('%Y-%m-%d'), 'price': ls_low['price']},
                        'left_shoulder_end': {'date': ls_end['date'].strftime('%Y-%m-%d'), 'price': ls_end['price']},
                        'head_low': {'date': head_low['date'].strftime('%Y-%m-%d'), 'price': head_low['price']},
                        'head_end': {'date': head_end['date'].strftime('%Y-%m-%d'), 'price': head_end['price']},
                        'right_shoulder_low': {'date': rs_low['date'].strftime('%Y-%m-%d'), 'price': rs_low['price']},
                        'right_shoulder_end': {'date': rs_end['date'].strftime('%Y-%m-%d'), 'price': rs_end['price']},
                        'neckline': neckline,
                        'technical_target': target_price,
                        'final_target': final_target,
                        'lifetime_high': lifetime_high,
                        # NEW: Additional pattern validation info
                        'left_peak': left_peak,
                        'head_peak': head_peak,
                        'right_peak': right_peak,
                        'neckline_deviation': max_deviation
                    },
                    'base_analysis': {
                        'base_formed': base_formed,
                        'base_high': base_high,
                        'base_low': base_low,
                        'base_range_percent': base_range * 100,
                        'breakout_above_base': breakout_above_base,
                        'breakout_above_neckline': breakout_above_neckline,
                        'is_green_candle': is_green
                    },
                    'reasoning': f'Reverse Head and Shoulders pattern identified. {status}'
                }

        # If no valid pattern found
        return {
            'signal': 'NEUTRAL',
            'current_price': current_price,
            'reasoning': 'No valid Reverse Head and Shoulders pattern detected'
        }

    def calculate_cup_handle_signal(self, hist_data):
        """
        Enhanced Cup with Handle Pattern Detection with SELL at Target
        Includes all strategy rules and charting points.
        """
        if len(hist_data) < 100:
            return {
                'signal': 'NEUTRAL',
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else 0,
                'reasoning': 'Insufficient data for Cup Handle pattern analysis'
            }

        current_price = hist_data['Close'].iloc[-1]
        recent_data = hist_data.tail(200)

        # Find pivot points
        pivot_points = []
        window = 15
        for i in range(window, len(recent_data) - window):
            if (recent_data['High'].iloc[i] >= recent_data['High'].iloc[i - window:i].max() and
                    recent_data['High'].iloc[i] >= recent_data['High'].iloc[i + 1:i + window + 1].max()):
                pivot_points.append({
                    'type': 'high',
                    'date': recent_data.index[i],
                    'price': recent_data['High'].iloc[i],
                    'index': i
                })
            if (recent_data['Low'].iloc[i] <= recent_data['Low'].iloc[i - window:i].min() and
                    recent_data['Low'].iloc[i] <= recent_data['Low'].iloc[i + 1:i + window + 1].min()):
                pivot_points.append({
                    'type': 'low',
                    'date': recent_data.index[i],
                    'price': recent_data['Low'].iloc[i],
                    'index': i
                })

        pivot_points.sort(key=lambda x: x['index'])

        if len(pivot_points) < 5:
            return {
                'signal': 'NEUTRAL',
                'current_price': current_price,
                'reasoning': 'Insufficient pivot points for Cup Handle pattern'
            }

        # Detect cup + handle pattern
        for i in range(len(pivot_points) - 4):
            for j in range(i, len(pivot_points) - 2):
                if (pivot_points[j]['type'] == 'high' and
                        pivot_points[j + 1]['type'] == 'low'):
                    for k in range(j + 2, len(pivot_points)):
                        if pivot_points[k]['type'] == 'high':
                            cup_left_high = pivot_points[j]
                            cup_low = pivot_points[j + 1]
                            cup_right_high = pivot_points[k]
                            cup_depth_percent = (cup_left_high['price'] - cup_low['price']) / cup_left_high['price']
                            if cup_depth_percent < 0.15:
                                continue
                            recovery_percent = (cup_right_high['price'] - cup_low['price']) / (
                                    cup_left_high['price'] - cup_low['price'])
                            if recovery_percent < 0.80:
                                continue

                            neckline = max(cup_left_high['price'], cup_right_high['price'])
                            height_difference = abs(cup_left_high['price'] - cup_right_high['price'])
                            if height_difference / neckline > 0.05:
                                continue

                            # Handle analysis
                            handle_data = recent_data.iloc[cup_right_high['index']:]
                            if len(handle_data) < 10:
                                continue

                            handle_low_idx = handle_data['Low'].idxmin()
                            handle_low_price = handle_data['Low'].min()
                            handle_low_position = handle_data.index.get_loc(handle_low_idx)

                            handle_depth = cup_right_high['price'] - handle_low_price
                            cup_depth = cup_left_high['price'] - cup_low['price']
                            if handle_depth >= 0.5 * cup_depth:
                                continue
                            if handle_low_price < cup_low['price']:
                                continue

                            handle_base_periods = min(15, len(handle_data) - handle_low_position)
                            if handle_base_periods < 5:
                                continue

                            base_data = handle_data.iloc[-handle_base_periods:]
                            base_high = base_data['High'].max()
                            base_low = base_data['Low'].min()
                            base_range_percent = (base_high - base_low) / base_low
                            base_formed = base_range_percent < 0.05

                            depth_to_neckline = neckline - cup_low['price']
                            target_price = neckline + depth_to_neckline

                            breakout_above_base = current_price > base_high
                            breakout_above_neckline = current_price > neckline
                            latest_candle = recent_data.iloc[-1]
                            is_green_candle = latest_candle['Close'] > latest_candle['Open']

                            # SELL at target check (NEW)
                            if current_price >= target_price:
                                signal = 'SELL'
                                status = 'Target price reached — SELL as per strategy rule'
                            else:
                                if (
                                        base_formed and breakout_above_base and breakout_above_neckline and is_green_candle):
                                    signal = 'BUY'
                                    status = 'Pattern confirmed: Handle base formed, breakout above neckline with green candle'
                                elif base_formed and breakout_above_neckline:
                                    signal = 'WATCH'
                                    status = 'Breakout above neckline but waiting for green candle confirmation'
                                elif base_formed and breakout_above_base:
                                    signal = 'WATCH'
                                    status = 'Breakout above base, watching for neckline breakout'
                                elif base_formed:
                                    signal = 'WATCH'
                                    status = 'Handle base formed, watching for breakout'
                                else:
                                    signal = 'NEUTRAL'
                                    status = 'Cup pattern identified but handle base not yet formed'

                            return {
                                'signal': signal,
                                'current_price': current_price,
                                'status': status,
                                'pattern_points': {
                                    'cup_left_high': {
                                        'date': cup_left_high['date'].strftime('%Y-%m-%d'),
                                        'price': cup_left_high['price']
                                    },
                                    'cup_low': {
                                        'date': cup_low['date'].strftime('%Y-%m-%d'),
                                        'price': cup_low['price']
                                    },
                                    'cup_right_high': {
                                        'date': cup_right_high['date'].strftime('%Y-%m-%d'),
                                        'price': cup_right_high['price']
                                    },
                                    'handle_low': {
                                        'date': handle_low_idx,
                                        'price': handle_low_price
                                    },
                                    'neckline': neckline,
                                    'target_price': target_price
                                },
                                'base_analysis': {
                                    'base_formed': base_formed,
                                    'base_high': base_high,
                                    'base_low': base_low,
                                    'base_range_percent': base_range_percent * 100,
                                    'breakout_above_base': breakout_above_base,
                                    'breakout_above_neckline': breakout_above_neckline,
                                    'is_green_candle': is_green_candle
                                },
                                'reasoning': f'Cup with Handle pattern identified. {status}'
                            }

        return {
            'signal': 'NEUTRAL',
            'current_price': current_price,
            'reasoning': 'No valid Cup with Handle pattern identified with proper sequence and validation'
        }
