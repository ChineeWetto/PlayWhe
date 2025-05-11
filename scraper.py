#!/usr/bin/env python3
"""
Play Whe Lottery Data Scraper (Updated)

This script scrapes historical Play Whe lottery data from nlcbgames.com/play-whe-past-results/
by systematically querying month by month from September 2016 to present.

The script handles form submissions, error recovery, and rate limiting to ensure reliable data collection.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import os
import re
from datetime import datetime, timedelta
import random
import json
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/scraper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("play_whe_scraper")

class PlayWheScraper:
    """
    A class to scrape Play Whe lottery results from nlcbgames.com
    """
    
    def __init__(self, base_url="https://nlcbgames.com/play-whe-past-results/", 
                 output_dir="data", 
                 delay_min=1, 
                 delay_max=3):
        """
        Initialize the scraper with configuration parameters
        
        Args:
            base_url (str): The URL to scrape data from
            output_dir (str): Directory to save the scraped data
            delay_min (int): Minimum delay between requests in seconds
            delay_max (int): Maximum delay between requests in seconds
        """
        self.base_url = base_url
        self.search_url = "https://nlcbgames.com/play-whe-search-results/"
        self.output_dir = output_dir
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': base_url,
        })
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.results = []
        
        # Month and year mapping
        self.months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
    def _random_delay(self):
        """
        Implement a random delay between requests to avoid overloading the server
        """
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)
        
    def _fetch_page(self, url, params=None, max_retries=3):
        """
        Fetch a page with retry logic
        
        Args:
            url (str): URL to fetch
            params (dict): Optional parameters for GET request
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            BeautifulSoup object or None if all retries fail
        """
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Fetching {url} with params {params}")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup
            
            except requests.exceptions.RequestException as e:
                retries += 1
                logger.warning(f"Error fetching {url}: {e}. Retry {retries}/{max_retries}")
                
                if retries < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retries + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} retries")
                    return None
                    
        return None
    
    def _parse_month_results(self, soup, month, year):
        """
        Parse the Play Whe results for a specific month
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object of the month results page
            month (str): Month name (e.g., 'September')
            year (str): Year (e.g., '2016')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        results = []
        
        try:
            # Find all tables on the page
            tables = soup.find_all('table')
            
            if not tables:
                logger.warning(f"No tables found for {month} {year}")
                return results
                
            # Get the day headers (MON, TUE, etc.)
            day_headers = soup.find_all('h2', class_='elementor-heading-title')
            days = [h.get_text(strip=True) for h in day_headers if h.get_text(strip=True) in ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']]
            
            # Map abbreviated day to full day name
            day_mapping = {
                'MON': 'Monday',
                'TUE': 'Tuesday',
                'WED': 'Wednesday',
                'THU': 'Thursday',
                'FRI': 'Friday',
                'SAT': 'Saturday',
                'SUN': 'Sunday'
            }
            
            # Process each table (usually one table per day)
            for table_idx, table in enumerate(tables):
                # Get the draw times from the header row
                header_row = table.find('tr')
                if not header_row:
                    continue
                    
                time_cells = header_row.find_all('th')
                draw_times = [cell.get_text(strip=True) for cell in time_cells]
                
                if not draw_times:
                    continue
                
                # Get the day for this table
                day_of_week = day_mapping.get(days[table_idx] if table_idx < len(days) else 'Unknown', 'Unknown')
                
                # Process each row of results
                result_rows = table.find_all('tr')[1:]  # Skip header row
                
                for row_idx, row in enumerate(result_rows):
                    cells = row.find_all('td')
                    
                    if len(cells) != len(draw_times):
                        continue
                        
                    for col_idx, cell in enumerate(cells):
                        try:
                            # Extract the winning number using the correct class
                            win_number_el = cell.find('span', class_='archivewin')
                            if not win_number_el:
                                continue
                                
                            number = int(win_number_el.get_text(strip=True))
                            
                            # Extract the draw number using the correct class
                            draw_info_el = cell.find('span', class_='acrhivedraw')
                            if not draw_info_el:
                                continue
                                
                            draw_text = draw_info_el.get_text(strip=True)
                            draw_number_match = re.search(r'Draw #:(\d+)', draw_text)
                            draw_number = int(draw_number_match.group(1)) if draw_number_match else None
                            
                            # Check if there's an 'M' marker
                            has_m_marker = 'M' in cell.get_text(strip=True)
                            
                            # Calculate the date based on day of week, month, and year
                            # This is an approximation and may need adjustment
                            date_obj = self._approximate_date(day_of_week, month, year, row_idx)
                            
                            # Create result dictionary
                            result = {
                                'date': date_obj.strftime('%Y-%m-%d') if date_obj else f"{year}-{self._month_to_number(month):02d}-??",
                                'time': draw_times[col_idx],
                                'number': number,
                                'draw_number': draw_number,
                                'day_of_week': day_of_week,
                                'has_m_marker': has_m_marker,
                                'month': month,
                                'year': year
                            }
                            
                            results.append(result)
                            
                        except (ValueError, AttributeError, IndexError) as e:
                            logger.warning(f"Error parsing cell: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Error parsing month results: {e}")
            
        return results
    
    def _month_to_number(self, month):
        """
        Convert month name to month number
        
        Args:
            month (str): Month name (e.g., 'September')
            
        Returns:
            int: Month number (1-12)
        """
        try:
            return self.months.index(month) + 1
        except ValueError:
            return 0
    
    def _approximate_date(self, day_of_week, month, year, week_idx):
        """
        Approximate the date based on day of week, month, year, and week index
        
        Args:
            day_of_week (str): Day of week (e.g., 'Monday')
            month (str): Month name (e.g., 'September')
            year (str): Year (e.g., '2016')
            week_idx (int): Week index (0-based)
            
        Returns:
            datetime: Approximated date
        """
        try:
            year_int = int(year)
            month_int = self._month_to_number(month)
            
            # Get the first day of the month
            first_day = datetime(year_int, month_int, 1)
            
            # Find the first occurrence of the specified day in the month
            day_idx = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            target_day_idx = day_idx.get(day_of_week, 0)
            first_day_idx = first_day.weekday()
            
            # Calculate days to add to reach the first occurrence of the target day
            days_to_add = (target_day_idx - first_day_idx) % 7
            first_occurrence = first_day + timedelta(days=days_to_add)
            
            # Add weeks based on the week index
            target_date = first_occurrence + timedelta(days=7 * week_idx)
            
            # Check if the date is still in the correct month
            if target_date.month != month_int:
                return None
                
            return target_date
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Error approximating date: {e}")
            return None
    
    def _parse_month_results_alternative(self, soup, month, year):
        """
        Alternative parsing method that directly looks for specific classes
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object of the month results page
            month (str): Month name (e.g., 'September')
            year (str): Year (e.g., '2016')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        results = []
        
        try:
            # Find all cells with winning numbers
            cells = soup.find_all('td')
            
            # Get the time headers
            time_headers = []
            tables = soup.find_all('table')
            if tables:
                header_row = tables[0].find('tr')
                if header_row:
                    time_cells = header_row.find_all('th')
                    time_headers = [cell.get_text(strip=True) for cell in time_cells]
            
            if not time_headers:
                time_headers = ['10:30AM', '1:00PM', '4:00PM', '6:30PM']  # Default if not found
            
            # Process each cell
            for cell_idx, cell in enumerate(cells):
                try:
                    # Extract the winning number
                    win_number_el = cell.find('span', class_='archivewin')
                    if not win_number_el:
                        continue
                        
                    number = int(win_number_el.get_text(strip=True))
                    
                    # Extract the draw number
                    draw_info_el = cell.find('span', class_='acrhivedraw')
                    if not draw_info_el:
                        continue
                        
                    draw_text = draw_info_el.get_text(strip=True)
                    draw_number_match = re.search(r'Draw #:(\d+)', draw_text)
                    draw_number = int(draw_number_match.group(1)) if draw_number_match else None
                    
                    # Check if there's an 'M' marker
                    has_m_marker = 'M' in cell.get_text(strip=True)
                    
                    # Determine the time based on cell position
                    time_idx = cell_idx % len(time_headers)
                    draw_time = time_headers[time_idx]
                    
                    # Create result dictionary
                    result = {
                        'date': f"{year}-{self._month_to_number(month):02d}-??",  # Exact date unknown
                        'time': draw_time,
                        'number': number,
                        'draw_number': draw_number,
                        'day_of_week': 'Unknown',  # Day of week unknown
                        'has_m_marker': has_m_marker,
                        'month': month,
                        'year': year
                    }
                    
                    results.append(result)
                    
                except (ValueError, AttributeError, IndexError) as e:
                    logger.warning(f"Error parsing cell: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing month results with alternative method: {e}")
            
        return results
    
    def scrape_month(self, month, year):
        """
        Scrape Play Whe results for a specific month and year
        
        Args:
            month (str): Month name (e.g., 'September')
            year (str): Year (e.g., '2016')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        logger.info(f"Scraping Play Whe results for {month} {year}")
        
        # Prepare parameters for the request
        params = {
            'tmonth': month,
            'tyear': year
        }
        
        # Fetch the page
        soup = self._fetch_page(self.search_url, params=params)
        
        if not soup:
            logger.error(f"Failed to fetch results for {month} {year}")
            return []
            
        # Try the primary parsing method first
        month_results = self._parse_month_results(soup, month, year)
        
        # If no results, try the alternative method
        if not month_results:
            logger.info(f"Primary parsing method found no results for {month} {year}, trying alternative method")
            month_results = self._parse_month_results_alternative(soup, month, year)
            
        if month_results:
            logger.info(f"Extracted {len(month_results)} results for {m
(Content truncated due to size limit. Use line ranges to read in chunks)