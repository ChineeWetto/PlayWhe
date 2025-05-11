#!/usr/bin/env python3
"""
NLCB Play Whe Lottery Data Scraper

This script scrapes historical Play Whe lottery data from nlcbplaywhelotto.com
by systematically querying month by month from 2015 to present.

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
        logging.FileHandler("data/nlcb_scraper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("nlcb_play_whe_scraper")

class NLCBPlayWheScraper:
    """
    A class to scrape Play Whe lottery results from nlcbplaywhelotto.com
    """
    
    def __init__(self, base_url="https://www.nlcbplaywhelotto.com/nlcb-play-whe-results/", 
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
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        
    def _random_delay(self):
        """
        Implement a random delay between requests to avoid overloading the server
        """
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)
        
    def _fetch_page(self, url, params=None, data=None, method="GET", max_retries=3):
        """
        Fetch a page with retry logic
        
        Args:
            url (str): URL to fetch
            params (dict): Optional parameters for GET request
            data (dict): Optional data for POST request
            method (str): HTTP method (GET or POST)
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            BeautifulSoup object or None if all retries fail
        """
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Fetching {url} with method {method}")
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=30)
                else:
                    response = self.session.post(url, data=data, timeout=30)
                
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
            month (str): Month name (e.g., 'Jan')
            year (str): Year (e.g., '2015')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        results = []
        
        try:
            # Find all result sections
            result_sections = soup.find_all('div', class_='play-whe-result')
            
            if not result_sections:
                logger.warning(f"No result sections found for {month} {year}")
                return results
                
            # Process each result section
            for section in result_sections:
                try:
                    # Extract date
                    date_header = section.find('h2', class_='date-header')
                    if not date_header:
                        continue
                        
                    date_text = date_header.get_text(strip=True)
                    date_match = re.search(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', date_text)
                    
                    if not date_match:
                        continue
                        
                    day, month_abbr, year_short = date_match.groups()
                    full_year = f"20{year_short}"
                    date_str = f"{full_year}-{self._month_to_number(month_abbr):02d}-{int(day):02d}"
                    
                    # Find all draw sections
                    draw_sections = section.find_all('div', class_='draw-section')
                    
                    for draw_section in draw_sections:
                        # Extract draw time
                        time_header = draw_section.find('h3', class_='time-header')
                        if not time_header:
                            continue
                            
                        time_text = time_header.get_text(strip=True)
                        
                        # Extract draw number
                        draw_info = draw_section.find('div', class_='draw-info')
                        if not draw_info:
                            continue
                            
                        draw_text = draw_info.get_text(strip=True)
                        draw_match = re.search(r'Draw #(\d+)', draw_text)
                        
                        if not draw_match:
                            continue
                            
                        draw_number = int(draw_match.group(1))
                        
                        # Extract winning number
                        number_div = draw_section.find('div', class_='number')
                        if not number_div:
                            continue
                            
                        number_text = number_div.get_text(strip=True)
                        try:
                            number = int(number_text)
                        except ValueError:
                            continue
                            
                        # Create result dictionary
                        result = {
                            'date': date_str,
                            'time': time_text,
                            'number': number,
                            'draw_number': draw_number,
                            'day_of_week': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                            'month': month_abbr,
                            'year': full_year
                        }
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error parsing result section: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing month results: {e}")
            
        return results
    
    def _parse_search_results(self, soup):
        """
        Parse the Play Whe search results
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object of the search results page
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        results = []
        
        try:
            # Find all result containers
            result_containers = soup.find_all('div', class_='result-container')
            
            if not result_containers:
                # Try alternative structure
                result_containers = soup.find_all('div', class_='play-whe-result')
                
            if not result_containers:
                logger.warning("No result containers found")
                return results
                
            # Process each result container
            for container in result_containers:
                try:
                    # Extract date
                    date_element = container.find(['h2', 'div'], class_=['date', 'date-header'])
                    if not date_element:
                        continue
                        
                    date_text = date_element.get_text(strip=True)
                    date_match = re.search(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', date_text)
                    
                    if not date_match:
                        continue
                        
                    day, month_abbr, year_short = date_match.groups()
                    full_year = f"20{year_short}"
                    date_str = f"{full_year}-{self._month_to_number(month_abbr):02d}-{int(day):02d}"
                    
                    # Find all draw sections
                    draw_sections = container.find_all(['div', 'section'], class_=['draw', 'draw-section'])
                    
                    for draw_section in draw_sections:
                        # Extract draw time
                        time_element = draw_section.find(['h3', 'div'], class_=['time', 'time-header'])
                        if not time_element:
                            continue
                            
                        time_text = time_element.get_text(strip=True)
                        
                        # Extract draw number
                        draw_element = draw_section.find('div', class_=['draw-number', 'draw-info'])
                        if not draw_element:
                            continue
                            
                        draw_text = draw_element.get_text(strip=True)
                        draw_match = re.search(r'Draw #(\d+)', draw_text)
                        
                        if not draw_match:
                            continue
                            
                        draw_number = int(draw_match.group(1))
                        
                        # Extract winning number
                        number_element = draw_section.find(['div', 'span'], class_=['number', 'winning-number'])
                        if not number_element:
                            continue
                            
                        number_text = number_element.get_text(strip=True)
                        try:
                            number = int(number_text)
                        except ValueError:
                            continue
                            
                        # Create result dictionary
                        result = {
                            'date': date_str,
                            'time': time_text,
                            'number': number,
                            'draw_number': draw_number,
                            'day_of_week': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                            'month': month_abbr,
                            'year': full_year
                        }
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error parsing result container: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            
        return results
    
    def _month_to_number(self, month):
        """
        Convert month name to month number
        
        Args:
            month (str): Month name (e.g., 'Jan')
            
        Returns:
            int: Month number (1-12)
        """
        try:
            return self.months.index(month) + 1
        except ValueError:
            # Try to match partial month name
            for i, m in enumerate(self.months):
                if month.lower() in m.lower():
                    return i + 1
            return 0
    
    def scrape_month(self, month, year):
        """
        Scrape Play Whe results for a specific month and year
        
        Args:
            month (str): Month name (e.g., 'Jan')
            year (str): Year (e.g., '2015')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        logger.info(f"Scraping Play Whe results for {month} {year}")
        
        # Prepare data for the request
        data = {
            'month': month,
            'year': year,
            'search': 'SEARCH'
        }
        
        # Fetch the page
        soup = self._fetch_page(self.base_url, data=data, method="POST")
        
        if not soup:
            logger.error(f"Failed to fetch results for {month} {year}")
            return []
            
        # Parse the results
        month_results = self._parse_search_results(soup)
            
        if month_results:
            logger.info(f"Extracted {len(month_results)} results for {month} {year}")
        else:
            logger.warning(f"No results found for {month} {year}")
            
        # Add delay before the next request
        self._random_delay()
        
        return month_results
    
    def scrape_range(self, start_month, start_year, end_month=None, end_year=None):
        """
        Scrape Play Whe results for a range of months
        
        Args:
            start_month (str): Starting month name (e.g., 'Jan')
            start_year (str): Starting year (e.g., '2015')
            end_month (str, optional): Ending month name. If None, use current month.
            end_year (str, optional): Ending year. If None, use current year.
            
        Returns:
            pandas.DataFrame: DataFrame containing all scraped results
        """
        logger.info(f"Starting Play Whe data scraping from {start_month} {start_year}")
        
        # If end month/year not provided, use current month/year
        if end_month is None or end_year is None:
            now = datetime.now()
            end_month = self.months[now.month - 1] if end_month is None else end_month
            end_year = str(now.year) if end_year is None else end_year
            
        # Convert month names to indices
        start_month_idx = self.months.index(start_month)
        start_year_idx = int(start_year)
        end_month_idx = self.months.index(end_month)
        end_year_idx = int(end_year)
        
        # Iterate through each month in the range
        current_year_idx = start_year_idx
        current_month_idx = start_month_idx

        while (current_year_idx < end_year_idx) or (current_year_idx == end_year_idx and current_month_idx <= end_month_idx):
            # Get current month and year
            current_month = self.months[current_month_idx]
            current_year = str(current_year_idx)

            logger.info(f"Scraping {current_month} {current_year}...")

            # Scrape the current month
            month_results = self.scrape_month(current_month, current_year)

            # Add to results
            self.results.extend(month_results)

            # Save incremental results (in case of failure)
            if month_results:
                temp_df = pd.DataFrame(self.results)
                temp_path = os.path.join(self.output_dir, f'play_whe_results_temp.csv')
                temp_df.to_csv(temp_path, index=False)
                logger.info(f"Saved {len(self.results)} results so far to {temp_path}")

            # Move to next month
            current_month_idx += 1
            if current_month_idx >= len(self.months):
                current_month_idx = 0
                current_year_idx += 1

            # Add delay before the next request
            self._random_delay()

        # Convert results to DataFrame
        if self.results:
            df = pd.DataFrame(self.results)

            # Sort by date and draw number
            if 'date' in df.columns and 'draw_number' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by=['date', 'draw_number'])

            # Save final results
            csv_path = os.path.join(self.output_dir, f'play_whe_results_{start_month}_{start_year}_to_{end_month}_{end_year}.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} results to {csv_path}")

            return df
        else:
            logger.error("No results were found for the specified range")
            return None

    def save_results(self, file_name=None):
        """
        Save the scraped results to a CSV file

        Args:
            file_name (str, optional): Name of the output file. If None, a default name is used.

        Returns:
            str: Path to the saved file
        """
        if not self.results:
            logger.warning("No results to save")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Generate file name if not provided
        if file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'play_whe_results_{timestamp}.csv'

        # Save to CSV
        csv_path = os.path.join(self.output_dir, file_name)
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(df)} results to {csv_path}")

        return csv_path


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scrape Play Whe lottery results')
    parser.add_argument('--start-month', type=str, default='Jan', help='Starting month (e.g., Jan)')
    parser.add_argument('--start-year', type=str, default='2015', help='Starting year (e.g., 2015)')
    parser.add_argument('--end-month', type=str, help='Ending month (default: current month)')
    parser.add_argument('--end-year', type=str, help='Ending year (default: current year)')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--delay-min', type=float, default=1, help='Minimum delay between requests (seconds)')
    parser.add_argument('--delay-max', type=float, default=3, help='Maximum delay between requests (seconds)')

    args = parser.parse_args()

    # Create scraper instance
    scraper = NLCBPlayWheScraper(
        output_dir=args.output_dir,
        delay_min=args.delay_min,
        delay_max=args.delay_max
    )

    # Scrape results
    logger.info("Starting Play Whe data scraping")

    results_df = scraper.scrape_range(
        start_month=args.start_month,
        start_year=args.start_year,
        end_month=args.end_month,
        end_year=args.end_year
    )

    if results_df is not None:
        logger.info(f"Successfully scraped {len(results_df)} Play Whe results")
    else:
        logger.error("Failed to scrape Play Whe results")
(Content truncated due to size limit. Use line ranges to read in chunks)