from pickle import TRUE
import time
import requests
import bs4

# 設定Cookie
cookies = {
    'over18': '1'
}


def getCurrentPageLink(URL):
    # 發送get 請求 到 ptt 八卦版
    response = requests.get(URL, cookies=cookies)
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


def KeywordSearch(key):
    currentPage = 1
    all_links = []
    while(True):
        # 連結
        BASE_URL = "https://www.ptt.cc/bbs/Gossiping/search?page=" + str(currentPage) + "&q="
        # print(BASE_URL)
        URL = BASE_URL + key
        #print(URL)
        try:
            links = getCurrentPageLink(URL)
            time.sleep(2)
            all_links.extend(links)
            print('current page is {}'.format(currentPage))
            currentPage += 1
        except AttributeError:
            break
    return all_links


def getCommentsFromEveryLink(links):
    all_messages = []
    print('there are {} links'.format(len(links)))
    index = 1
    for link in links:

        response = requests.get(link, cookies=cookies)
        time.sleep(2)

        link_messages = []

        #將原始碼做整理
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        #使用find_all()找尋特定目標
        comments = soup.find_all('div', 'push')
        for comment in comments:
            message = comment.find(
                'span', 'f3 push-content').getText().replace(':', '').strip()
            link_messages.append(message)
        print('link {} is done.'.format(index))
        index += 1
        all_messages.extend(link_messages)

    return all_messages

keyword = str(input("Enter the keyword to search:"))
#User intput:
file_name = str(
    input("Enter the name you want for the new output file (Maybe the keyword):"))
my_links = KeywordSearch(keyword)

all_messages = getCommentsFromEveryLink(my_links)
print(all_messages)

#寫入檔案中
with open(file_name+'.txt', 'w') as f:
    for message in all_messages:
        f.write(message + "\n")
