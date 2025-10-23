import time
import json
import logging
import time
import requests
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from tabulate import tabulate

logger = logging.getLogger(__name__)

def get_api_detail_info(api_cards, timeout: int=5, max_trial: int=3, save_file_path: str="api_cards_info.jsonl"):
    trial = 0
    # Set Chrome options (headless = run without GUI)
    options = Options()
    options.add_argument("--headless")  # remove this if you want to see the browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(timeout)
    
    for api_card in tqdm(api_cards):
        api_url = f"https://data.seoul.go.kr/dataList/{api_card['link']}"
        logger.info(api_url, driver.current_url)
        trial = 0
        
        # Iterate to get content safely
        while trial <= max_trial:
            trial += 1
            
            try:
                driver.get(api_url)
                wait = WebDriverWait(driver, 10)
                if "openapi" in api_card["data_types"]:
                    tab_container = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ui-tab-btns")))
                    buttons = tab_container.find_elements(By.TAG_NAME, "button")
            except TimeoutException:
                logger.warning(f"Timeout during driver.get. api_idx: {api_card['idx']}. Retry... {trial}")
                # driver.execute_script("window.stop()") # Error raised if driver is unresponsive
                try:
                    driver.current_url # Test if driver is still responsive
                except:
                    # Driver is unresponsive, recreate it
                    logger.warning(f"Driver unresponsive, recreating... api_idx: {api_card['idx']}")
                    try:
                        driver.quit()
                    except:
                        pass
                    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                    driver.set_page_load_timeout(timeout)
                time.sleep(300) # Wait until release of the request blocking
                continue
            except Exception as e:
                logger.warning(f"Unexpected error occurred. api_idx: {api_card['idx']}. Retry... {trial}")
                continue
            
            if "openapi" in api_card["data_types"]:
                for btn in buttons:
                    if "open api" in btn.text.lower():
                        btn.click()
                        break
                else:
                    continue
                
            if "openapi" in api_card["data_types"]:
                try:
                    # Wait to observe the result
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ui-tab-pnl")))
                except Exception as e:
                    logger.warning(f"Failed to load ajax. api_idx: {api_card['idx']}. Retry... {trial}")
                    continue
            
            # Find all accordion panels (each section)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Error case
            if "서비스 지연이 발생하고 있습니다. 다시 시도하여 주십시오.(103)" in str(soup):
                logger.warning("Error to get content. Retry...")
                time.sleep(300)
                continue
            break

        if soup.contents:
            markdown_sections = []
            detail_conts = soup.find_all("div", class_="detail-cont")
            for detail_cont in detail_conts:
                detail_cont_tit = detail_cont.select_one("div", class_="detail-cont-tit").h2
                
                if detail_cont.get("id") == "fileDownList":
                    # Even if the api_card does not have "file" in its data_types, it still exists the fileDownList detail_cont which is not visible.
                    # Skip it if it's the case.
                    if "file" in api_card["data_types"]:
                        # For getting the payload to download the file, we have to get to know what infSeq number is for this api.
                        input_tags = soup.select_one("#frmFile").find_all("input")
                        for input_tag in input_tags:
                            if input_tag.get("name") == "infSeq":
                                break
                        else:
                            input_tag = None
                        
                        section_title = detail_cont.select_one("div", class_="detail-cont-tit").h2.get_text(strip=True)
                        tables = detail_cont.find_all("table")
                        for table in tables:
                            headers = [th.get_text(strip=True) for th in table.find_all("th")]
                            rows = []
                            trs = table.find_all("tr")
                            for i, tr in enumerate(trs):
                                cells = []
                                for td in tr.find_all("td"):
                                    if download_link_tag := td.select_one("a"):
                                        seq = re.sub("[^0-9]", "", download_link_tag.get("href"))
                                        seq = int(seq) if seq.isdigit() else download_link_tag.get("href")
                                        payload = {
                                            "base_url": "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
                                            "infId": api_card["link"].split("/")[0],
                                            "seq": seq,
                                            "infSeq": input_tag.get("value") if input_tag else None,
                                        }
                                        cells += [json.dumps(payload)]
                                    else:
                                        cells += [td.get_text(strip=True)]
                                if cells:
                                    rows.append(cells)
                            download_links = [dict(zip(headers, row)) for row in rows]
                            api_card[section_title] = download_links
                    
                if detail_cont_tit:
                    if detail_cont_tit.get_text(strip=True) == "데이터 정보":
                        section_title = detail_cont_tit.get_text(strip=True)
                        tables = detail_cont.find_all("table")
                        
                        for table in tables:
                            headers = [th.get_text(strip=True) for th in table.find_all("th")]
                            rows = []
                            for tr in table.find_all("tr"):
                                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                                if cells:
                                    rows.append(cells)
                            if rows:
                                if any(len(row) != len(headers) for row in rows):
                                    # Case for horizontal table which has no header
                                    if len(headers) == len(rows):
                                        rows = [row[0] for row in rows]
                                        dict_table = dict(zip(headers, rows))
                                        api_card[section_title] = dict_table
                                        md_table = f"{json.dumps(dict_table, indent=4, ensure_ascii=False)}"
                                    else:
                                        continue
                                else:
                                    md_table = tabulate(rows, headers=headers, tablefmt="github")
                                
                                # Append to markdown_sections
                                # print(f"## {section_title}\n\n{md_table}\n"[:20])
                                markdown_sections.append(f"## {section_title}\n\n{md_table}\n")
                        
                    if detail_cont_tit.get_text(strip=True) == "연관데이터":
                        related = detail_cont.select_one("ul", class_="list-connection")
                        lst = related.find_all("li")
                        related_data = []
                        for element in lst:
                            category = element.span.get_text(strip=True)
                            api_name = element.a.get_text(strip=True)
                            api_link = element.a.get("href")
                            related_data += [{"category": category, "api_name": api_name, "api_link": api_link}]
                        api_card["연관데이터"] = related_data
            
            # For '미리보기'
            if "openapi" in api_card["data_types"]:
                panel = soup.select("#dataDiv")[0]
                tables = panel.find_all("table")
                for table in tables:
                    title_tag = table.find_previous("div", class_="detail-cont-tit")
                    table_title = title_tag.h2.get_text(strip=True) if title_tag else "Unknown Section"
                    headers = [th.get_text(strip=True) for th in table.find_all("th")]
                    rows = []
                    for tr in table.find_all("tr"):
                        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                        if cells:
                            rows.append(cells)
                    if rows:
                        if any(len(row) != len(headers) for row in rows):
                            # Case for horizontal table which has no header
                            if len(headers) == len(rows):
                                rows = [row[0] for row in rows]
                                md_table = f"{json.dumps(dict(zip(headers, rows)), indent=4, ensure_ascii=False)}"
                            else:
                                continue
                        else:
                            md_table = tabulate(rows, headers=headers, tablefmt="github")
                        
                        # Append to markdown_sections
                        markdown_sections.append(f"## {table_title}\n\n{md_table}\n")
                                
            api_card["api_info"] = "\n\n".join(markdown_sections)
        
        with open(save_file_path, "a") as f:
            f.write(json.dumps(api_card, ensure_ascii=False) + "\n")
        
        # Clear browser state
        try:
            driver.delete_all_cookies()
            driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
        except Exception as e:
            logger.warning(f"Failed to clear browser state: {e}")
        driver.delete_all_cookies()
        driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
            
    driver.quit()
    return api_cards

def create_page_payload(page_num: int):
    payload = {
        "pageIndex": page_num,   # <-- change this to move through pages
        "sortColBy": "",
        "searchValue": "",
        "resSearch": "",
        "resSearchValue": "",
        "mapId": "",
        "menuCd": "",
        "organCd": "",
        "organNm": "",
        "tagNm": "",
        "datasetKind": "",
        "searchFlag": "N",
        "basickwd": "",
        "exactkwd": "",
        "inkwd": "",
        "notkwd": "",
        "orgkwd": "",
        "syskwd": "",
        "detailkwd": "",
        "resSearchBbs": "",
        "resSearchMenu": "",
        "resDetailkwd": "",
        "resOrgSysYn": "",
        "searchKeyword": "",
        "basickwdval": "",
        "inkwdval": "",
        "notkwdval": "",
        "orgkwdval": "",
        "syskwdval": ""
    }
    return payload

# Step 1. Crawl api list
url = "https://data.seoul.go.kr/dataList/datasetList.do"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://data.seoul.go.kr/dataList/datasetList.do",
}
api_cards = []
max_page_num = 813

cnt = 0
for page_num in tqdm(range(1, max_page_num+1)):
    payload = create_page_payload(page_num=page_num)
    response = requests.post(url, data=payload, headers=headers)

    # parse it with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # get api_card and append to list
    for dl in soup.find_all("dl", class_="type-b"):
        api_category = dl.select_one("dt > em:nth-of-type(2)").get_text(strip=True)
        api_category = re.sub(r'[^가-힣A-Za-z0-9 ]', '', api_category) # remove special characters
        api_name = dl.select_one("dt > a > strong").get_text(strip=True).replace("\xa0", " ")
        is_updated = True if dl.select_one("dt > a > strong > img") else False
        api_description = dl.select_one("dd.list-statistics-info1").get_text(strip=True)
        modified_at = dl.select_one("dd.list-statistics-info2 span:nth-of-type(1)").get_text(strip=True).replace("수정일자 :", "").strip()
        provider = dl.select_one("dd.list-statistics-info2 span:nth-of-type(2)").get_text(strip=True).replace("제공기관 :", "").strip()
        provider_dept = dl.select_one("dd.list-statistics-info2 span:nth-of-type(3)").get_text(strip=True).replace("제공부서 :", "").strip()
        link = dl.select_one("dt > a")["data-rel"]
        data_types = [btn.get_text(strip=True).lower() for btn in dl.select("dd.list-statistics-info3 button")]

        api_cards += [
            {
                "idx": cnt,
                "api_name": api_name,
                "api_category": api_category,
                "is_updated": is_updated,
                "api_description": api_description,
                "modified_at": modified_at,
                "provider": provider,
                "provider_dept": provider_dept,
                "data_types": data_types,
                "link": link
            }
        ]
        cnt += 1

with open("./api_cards.jsonl", "a") as f:
    for api_card in api_cards:
        f.write(json.dumps(api_card, ensure_ascii=False) + "\n")

# Step 2. Render ajax contents through webdriver and crawl api detail information
api_cards = get_api_detail_info(api_cards, 5, 3, "api_cards_info.jsonl")