# This file was used to scrape the publicly available data-points from the frag-einen-anwalt.de website.
# Please check about the interval to do not overload the server and set the random interval accordingly!
# The scraper was built specifically for the html version of the site when accessing it, no guarantees, that it is still working.

import re
import requests
from threading import Thread
import time
import json

from random import randint
from datetime import datetime
from tqdm import tqdm
from time import sleep

DETAIL_QA_PAGE_REGEX = "<a class=\\'\\' title=\\'[^\s]+\\' href=\\'([^\s]+)\\'"
TITLE_RE = "<h1 class=\"fea-question-head-title\"><span >([^</]+)"
CATEGORY_RE = "h2 class=\"fea-question-head-areaoflaw\">[^>]+>([^</]+)"
DATE_RE = "(\d{1,2}\. .* \d{4} \d{1,2}:\d{1,2}) <span class=\"medium-show-up\">"
QAs_RE = "<p>((.|\n)*?)<\/p>"
PRICE_RE = "Preis: <b>(\d*,\d*) €"
RATING_QUESTION_RE = "class=\"rating\">(\d(,\d)?)"
RATING_LAWYER_RE = "(\d,\d) von 5 Sterne"

def extract_detail_urls_from_overview_page(page_number: int, thread_name):
    'Gets the url for all detail pages of an overview page'
    url = BASE_URL + str(page_number)

    rnd = randint(34, 48) / 10
    time.sleep(rnd)

    request_result = requests.get(url)
    while request_result.status_code == 403:
        print(f"{thread_name}: BLOCKED ON WEBSITE AND WAITING...")
        time.sleep(120)
        request_result = requests.get(url)

    content = request_result.text
    matches = re.findall(DETAIL_QA_PAGE_REGEX, content)
    return matches


def extract_info_for_detail_page_string(url: str, thread_name):
    'Extracts and return the content for all the selected categories below'

    rnd = randint(34, 48) / 10
    time.sleep(rnd)

    request_result = requests.get(url)

    while request_result.status_code == 403:
        print(f"{thread_name}: BLOCKED ON WEBSITE AND WAITING...")
        time.sleep(120)
        request_result = requests.get(url)

    content = request_result.text
    content = content.replace("<br>", "")

    title = None
    category = None
    date = None
    price = None
    rating_lawyer = None
    rating_question = None
    question_text = None
    answer_text = None
    sub_question_text = None
    sub_answer_text = None

    try:
        title_m = re.search(TITLE_RE, content)
        if title_m is not None:
            title = title_m.group(1)

        category_m = re.search(CATEGORY_RE, content)
        if category_m is not None:
            category = category_m.group(1)

        date_m = re.search(DATE_RE, content)
        if date_m is not None:
            date = date_m[1]

        price_m = re.search(PRICE_RE, content)
        if price_m is not None:
            price = float(price_m.group(1).replace(",", "."))

        # if rating of question does exist, than this regex is not None
        rating_question_match = re.search(RATING_QUESTION_RE, content)

        if rating_question_match == None:
            rating_lawyer_match = re.search(RATING_LAWYER_RE, content)
            if rating_lawyer_match is not None:
                rating_lawyer = float(
                    rating_lawyer_match.group(1).replace(",", "."))
        else:
            rating_question = float(
                rating_question_match.group(1).replace(",", "."))
            rating_lawyer = float(re.findall(RATING_LAWYER_RE, content)[
                1].replace(",", "."))

        matches = re.findall(QAs_RE, content)

        # Offset necessary to prevent using the summary or add_info for some questions
        offset = 0
        add_on_info_question_offset = 0
        len_matches = len(matches)
        if "fea-question-head-summary" in content:
            offset = 1
        if "Eingrenzung vom Fragesteller" in content:
            add_on_info_question_offset = 1

        # Question
        question_text = remove_html_chars(matches[0+offset][0])
        # Answer
        if len_matches > 1 + offset + add_on_info_question_offset:
            answer_text = remove_html_chars(
                matches[1+offset + add_on_info_question_offset][0])
        # Subquestion
        if len_matches > 3 + offset + add_on_info_question_offset:
            sub_question_text = remove_html_chars(
                matches[3+offset+add_on_info_question_offset][0])
        # Subanswer
        if len_matches > 4 + offset + add_on_info_question_offset:
            sub_answer_text = remove_html_chars(
                matches[4+offset + add_on_info_question_offset][0])

    except:
        return None

    return {"Title": title, "Date": date, "Rating_lawyer": rating_lawyer, "Rating_question": rating_question,
            "Category": category, "Price": price, "Question_text": question_text, "Answer_text": answer_text,
            "Sub_question_text": sub_question_text, "Sub_answer_text": sub_answer_text, "url": url}


def remove_html_chars(text: str) -> str:
    replacers = {"\n": " ", "<p>": "", "\r": "", "\t": "",
                 "<": "", "&quot;": "'", "&lpar;": "(", "&rpar;": ")", "&#167;": "§"}

    for (char, sub) in replacers.items():
        text = text.replace(char, sub)

    return text


def run_thread(start_page_nr, end_page_nr, thread_name):
    total_pages = end_page_nr - start_page_nr
    pages_scraped = 1
    unscrapeable_data = 0

    with open(f"../data/data/2.{thread_name}.json", "w", encoding="UTF-8") as f:
        for page_nr in range(start_page_nr, end_page_nr):
            urls = extract_detail_urls_from_overview_page(page_nr, thread_name)
            pbar = tqdm(urls)
            pbar.set_description(
                f"{thread_name} - current page: {pages_scraped}/{total_pages}")

            for url in pbar:
                info = extract_info_for_detail_page_string(url, thread_name)
                if info is not None:
                    json.dump(info, f, indent=4)
                else:
                    unscrapeable_data += 1
                    print(
                        f"{thread_name} exception - unscrapeable data points: {unscrapeable_data}")

            pages_scraped += 1

OFFSET = 0
'can be used to skip pages at the beginning'

MAX_OVERVIEW_PAGES = 6981  
'maximum number of pages for getting detail pages'

BASE_URL = "https://www.frag-einen-anwalt.de/forum_forum.asp?forum_id=0&page="

NO_THREADS = 2
start_time = time.time()

threads = []
page_incr = int((MAX_OVERVIEW_PAGES-OFFSET)/NO_THREADS)
start_page_nr = 0 + OFFSET
end_page_nr = page_incr + OFFSET

# runs the script for the number of threads setup
for id in range(NO_THREADS):

    if (id < NO_THREADS - 1):
        print("Thread #", id, "Start-End", start_page_nr, end_page_nr)
        t = Thread(target=run_thread, args=[
                   start_page_nr, end_page_nr, "Thread_"+str(id)])
    else:
        print("Thread #", id, "Start-End", start_page_nr, MAX_OVERVIEW_PAGES)
        t = Thread(target=run_thread, args=[
                   start_page_nr, MAX_OVERVIEW_PAGES, "Thread_"+str(id)])

    print(f"Thread_{str(id)} started...")
    t.start()
    threads.append(t)

    start_page_nr += page_incr

    if id == NO_THREADS:
        end_page_nr = MAX_OVERVIEW_PAGES
    else:
        end_page_nr += page_incr

for t in threads:
    t.join()

print(f"Total runtime: {time.time() - start_time}")
