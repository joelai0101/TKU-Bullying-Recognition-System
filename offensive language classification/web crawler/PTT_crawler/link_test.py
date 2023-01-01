from pickle import TRUE
import time
import requests
import bs4

# 設定Header與Cookie

def getCurrentPageLink(URL):
    my_headers = {'cookie': 'over18=1;'}
    # 發送get 請求 到 ptt 八卦版
    response = requests.get(URL, headers=my_headers)
    time.sleep(1)
    #  把網頁程式碼(HTML) 丟入 bs4模組分析
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    ## 查找html 元素 抓出內容
    main_container = soup.find(id='main-container')

    divs = main_container.find_all('div', 'title')

    a_Set = []
    links = []

    for div in divs:
        a = div.find_all('a')
        # [<a href="/bbs/Gossiping/M.1648453620.A.ED8.html">[協尋] 3/25晚上9：30左右 國一北上往苗栗頭份路</a>]
        # print(a)
        # print(type(a))  # <class 'bs4.element.ResultSet'>
        a_Set.extend(a)


    # 取得當前頁數文章連結
    for a in a_Set:
        link = "https://www.ptt.cc/" + a.get('href')
        links.append(link)  
        print(link)
    return links

keyword = str(input("Enter the keyword to search:"))
URL = "https://www.ptt.cc/bbs/Gossiping/search?page=1&q=" + keyword
my_links = getCurrentPageLink(URL)
print(my_links)