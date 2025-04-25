"""
Headspace: Analysis of the App’s Effectiveness

Scrape file

"""

import pandas as pd
import numpy as np
from app_store_scraper import AppStore


def main():
    headspace_data = AppStore(country='us', app_name='Headspace: Sleep & Meditation', app_id = '493145008')
    headspace_data.review(how_many=2000)

    hs_data_1 = pd.DataFrame(np.array(headspace_data.reviews),columns=['review'])
    hs_data_2 = hs_data_1.join(pd.DataFrame(hs_data_1.pop('review').tolist()))

    hs_data_2 = hs_data_2.sort_values('date', ascending=False)
    hs_data_2.head()

    headspace = hs_data_2[['date', 'userName', 'review', 'rating']]
    headspace.to_csv('headspace.csv', index=False)
    headspace.to_excel('headspace.xlsx', index=False)

if __name__ == "__main__":
    main()