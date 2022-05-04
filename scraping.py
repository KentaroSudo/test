import urllib.request, urllib.error

page = urllib.request.urlopen('https://www.wunderground.com/history/daily/KBUF/date/2009-1-1')

from bs4 import BeautifulSoup

soup = BeautifulSoup(page,features="html.parser")
nobrs = soup.find_all(attrs={"class":"wu-value wu-value-to"})
print(nobrs)
