from pickle import TRUE
import time
import requests
import bs4

cookies = {
    'over18': '1'
}

def getCommentsFromLink(link):
    response = requests.get(link, cookies=cookies)
    time.sleep(2)

    link_messages = []

    #將原始碼做整理
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    #使用find_all()找尋特定目標
    comments = soup.find_all('div', 'push')
    #sprint(comments)
    for comment in comments:
        message = comment.find(
            'span', 'f3 push-content').getText().replace(':', '').strip()
        link_messages.append(message)

    return link_messages


URL = "https://www.ptt.cc//bbs/Gossiping/M.1307762899.A.535.html"
messages = getCommentsFromLink(URL)

print(messages)