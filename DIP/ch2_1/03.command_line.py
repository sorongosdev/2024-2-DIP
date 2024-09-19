title = '서기 1년 1월 1일부터' \
        '오늘까지' \
        '일수 구하기'
months = [31, 28, 31, 30, 31,
          31, 30, 31, 30, 31]
year, month = 2020, 1
day = 7;
ratio = 365.2425

days = (year - 1) * ratio + \
       sum(months[:month - 1]) + day

print(title), print(' - 년:', year), print(' - 월:', month)
print(' - 일:', day);
print(' * 일수 총합:', int(days))
