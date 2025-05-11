#!/usr/bin/env python3
"""
NLCB Play Whe Lottery Data Scraper using Requests and BeautifulSoup

This script scrapes historical Play Whe lottery data from nlcbplaywhelotto.com
by systematically querying month by month from July 1994 to present.

The script handles form submissions, error recovery, and rate limiting to ensure reliable data collection.
"""

import os
import time
import random
import logging
import re
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/requests_scraper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("requests_play_whe_scraper")

class RequestsPlayWheScraper:
    """
    A class to scrape Play Whe lottery results from nlcbplaywhelotto.com using Requests and BeautifulSoup
    """
    
    def __init__(self, base_url="https://www.nlcbplaywhelotto.com/nlcb-play-whe-results/", 
                 output_dir="data", 
                 delay_min=2, 
                 delay_max=5):
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
        self.results = []
        self.session = requests.Session()
        
        # Set up headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': base_url,
            'Origin': 'https://www.nlcbplaywhelotto.com',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
        
    def _get_initial_page(self):
        """
        Get the initial page to extract form details and cookies
        
        Returns:
            BeautifulSoup: Parsed HTML of the initial page
        """
        try:
            logger.info(f"Getting initial page: {self.base_url}")
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting initial page: {e}")
            return None
    
    def _extract_form_details(self, soup):
        """
        Extract form details from the page
        
        Args:
            soup (BeautifulSoup): Parsed HTML of the page
            
        Returns:
            dict: Form details including action URL and hidden fields
        """
        form_details = {
            'action': self.base_url,  # Default to base URL
            'method': 'post',
            'hidden_fields': {}
        }
        
        try:
            # Find the search form
            form = soup.find('form')
            if form:
                # Get form action
                action = form.get('action')
                if action:
                    form_details['action'] = urljoin(self.base_url, action)
                
                # Get form method
                method = form.get('method', 'post')
                form_details['method'] = method.lower()
                
                # Get hidden fields
                for input_tag in form.find_all('input', type='hidden'):
                    name = input_tag.get('name')
                    value = input_tag.get('value')
                    if name:
                        form_details['hidden_fields'][name] = value
        except Exception as e:
            logger.error(f"Error extracting form details: {e}")
            
        return form_details
    
    def _parse_results(self, html_content):
        """
        Parse the Play Whe results from HTML content
        
        Args:
            html_content (str): HTML content of the results page
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        results = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        try:
            # Find all result sections
            result_sections = soup.find_all('div', class_=lambda c: c and ('play-whe-result' in c or 'result-container' in c))
            
            if not result_sections:
                # Try alternative selectors
                result_sections = soup.find_all('div', class_=lambda c: c and ('result' in c.lower() or 'draw' in c.lower()))
                
            if not result_sections:
                logger.warning("No result sections found")
                return results
                
            # Process each result section
            for section in result_sections:
                try:
                    # Extract date
                    date_header = section.find(['h2', 'div'], class_=lambda c: c and ('date' in c.lower() or 'header' in c.lower()))
                    if not date_header:
                        continue
                        
                    date_text = date_header.get_text(strip=True)
                    date_match = re.search(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', date_text)
                    
                    if not date_match:
                        continue
                        
                    day, month_abbr, year_short = date_match.groups()
                    full_year = f"20{year_short}" if int(year_short) >= 0 else f"19{year_short}"
                    date_str = f"{full_year}-{self._month_to_number(month_abbr):02d}-{int(day):02d}"
                    
                    # Find all draw sections
                    draw_sections = section.find_all(['div'], class_=lambda c: c and ('draw' in c.lower() or 'time' in c.lower()))
                    
                    for draw_section in draw_sections:
                        # Extract draw time
                        time_header = draw_section.find(['h3', 'div'], class_=lambda c: c and ('time' in c.lower() or 'header' in c.lower()))
                        if not time_header:
                            time_header = draw_section.find(text=lambda t: t and any(time_str in t.lower() for time_str in ['morning', 'midday', 'afternoon', 'evening']))
                            if not time_header:
                                continue
                                
                        time_text = time_header.get_text(strip=True)
                        
                        # Extract draw number
                        draw_info = draw_section.find(text=lambda t: t and 'draw #' in t.lower())
                        if not draw_info:
                            # Try to find it in a div
                            draw_div = draw_section.find(['div', 'span'], class_=lambda c: c and ('draw' in c.lower() or 'number' in c.lower()))
                            if draw_div:
                                draw_info = draw_div.get_text(strip=True)
                            
                        if draw_info:
                            draw_match = re.search(r'draw #?(\d+)', draw_info.lower())
                            if draw_match:
                                draw_number = int(draw_match.group(1))
                            else:
                                continue
                        else:
                            continue
                            
                        # Extract winning number
                        number_div = draw_section.find(['div', 'span'], class_=lambda c: c and ('number' in c.lower() or 'winning' in c.lower()))
                        if not number_div:
                            continue
                            
                        number_text = number_div.get_text(strip=True)
                        try:
                            number = int(re.search(r'\d+', number_text).group())
                        except (ValueError, AttributeError):
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
            logger.error(f"Error parsing results: {e}")
            
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
    
    def search_month_year(self, month, year):
        """
        Search for Play Whe results for a specific month and year
        
        Args:
            month (str): Month name (e.g., 'Jan')
            year (str): Year (e.g., '2015')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        logger.info(f"Searching for Play Whe results for {month} {year}")
        
        # Get initial page to extract form details and set cookies
        soup = self._get_initial_page()
        if not soup:
            logger.error("Failed to get initial page")
            return []
            
        # Extract form details
        form_details = self._extract_form_details(soup)
        
        # Prepare form data
        form_data = {
            'month': month,
            'year': year,
            'search': 'SEARCH'
        }
        
        # Add hidden fields
        form_data.update(form_details['hidden_fields'])
        
        try:
            # Submit the form
            logger.info(f"Submitting form to {form_details['action']} with method {form_details['method']}")
            if form_details['method'] == 'post':
                response = self.session.post(form_details['action'], data=form_data, timeout=30)
            else:
                response = self.session.get(form_details['action'], params=form_data, timeout=30)
                
            response.raise_for_status()
            
            # Parse the results
            month_results = self._parse_results(response.text)
            
            if month_results:
                logger.info(f"Extracted {len(month_results)} results for {month} {year}")
            else:
                # Try alternative approach - direct URL with query parameters
                alt_url = f"{self.base_url}?month={month}&year={year}&search=SEARCH"
                logger.info(f"No results found with form submission, trying alternative URL: {alt_url}")
                
                response = self.session.get(alt_url, timeout=30)
                response.raise_for_status()
                
                month_results = self._parse_results(response.text)
                
                if month_results:
                    logger.info(f"Extracted {len(month_results)} results with alternative URL for {month} {year}")
                else:
                    logger.warning(f"No results found for {month} {year}")
                
            # Add delay before the next request
            self._random_delay()
            
            return month_results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching for {month} {year}: {e}")
            return []
    
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
            current_month = self.months[current_month_idx]
            current_year = str(current_year_idx)
            
            # Scrape the current month
            month_results = self.search_month_year(current_month, current_year)
            self.results.extend(month_results)
            
            # Save intermediate results after each month
            self._save_intermediate_results(f"requests_{current_month}_{current_year}")
            
            # Move to the next month
            current_month_idx += 1
            if current_month_idx >= len(self.months):
                current_month_idx = 0
                current_year_idx += 1
                
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        if not df.empty:
            # Sort by date, time, and draw number
            df = df.sort_values(by=['date', 'time', 'draw_number'])
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, 'requests_play_whe_results.csv')
            df.to_csv(csv_path, index=False)

(Content truncated due to size limit. Use line ranges to read in chunks)